"""Map statistics kernel → /ugv/map/stats (the web viewer's statistics widget).

Fed with RTAB-Map's pose graph (rtabmap_msgs/MapGraph) and per-step Info; answers one flat dict of JSON
scalars (the client rejects a status that holds an object or an array):

  keyframes          int           poses in the latest graph (landmarks, negative ids, are not keyframes)
  loop_closures      int           distinct unordered (from, to) node pairs among the latest graph's closure links
                                   (global, local-space, user closure); neighbour / local-time / merged / virtual /
                                   prior / landmark / gravity links are not closures
  path_length_m      float | None  polyline through the graph poses in id order, x/y only (None if not finite)
  db_bytes           int | None    size of the database file (the caller stats it; None = no regular file)
  last_update_age_s  float | None  seconds since the id list or any pose changed (None before the first graph, and
                                   always None in localize mode: the map is read-only there)
  mode               str           mapping | localize
  calibration_placeholder bool     the camera calibration in use is a stand-in (camera.calib `placeholder`)

"Graph changed" is a change of content, not the arrival of a message, so a republished unchanged graph does not
look like progress. (Measured on the real stack: RTAB-Map publishes no graph at all while the robot stands still,
so the age keeps growing; the content test keeps it right if that ever changes.)

Pure: the node feeds messages and a clock. No ROS imports. The only file access is the separate helper
regular_file_size(), which the node calls for db_bytes.
"""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Iterable, Sequence
from itertools import pairwise

from ugv_localization.common.checks import require_bool, require_finite
from ugv_localization.modes import Mode


def path_length(poses: Iterable[tuple[float, float]]) -> float:
    """Length of the polyline through (x, y) poses in the order given. Fewer than two poses: 0.0."""
    return float(sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in pairwise(poses)))


# rtabmap::Link::Type (core/Link.h, rtabmap 0.23): kNeighbor 0, kGlobalClosure 1, kLocalSpaceClosure 2,
# kLocalTimeClosure 3, kUserClosure 4, kVirtualClosure 5, kNeighborMerged 6, kPosePrior 7, kLandmark 8, kGravity 9.
# rtabmap_msgs/Link.type carries that integer.
CLOSURE_LINK_TYPES = frozenset({1, 2, 4})


def closure_pairs(links: Iterable[tuple[int, int, int]]) -> int:
    """Distinct unordered (from, to) pairs among links of a closure type. Links are (from_id, to_id, type)."""
    return len({(min(a, b), max(a, b)) for a, b, kind in links if kind in CLOSURE_LINK_TYPES})


def regular_file_size(path: str) -> int | None:
    """Size in bytes of a regular file; None for "" / missing / a directory / anything else."""
    if not path:
        return None
    try:
        st = os.stat(os.path.expanduser(path))
    except OSError:
        return None
    return int(st.st_size) if stat.S_ISREG(st.st_mode) else None


def _key(values: Sequence[float]) -> tuple[object, ...]:
    """Comparable form of a pose: NaN (which is never equal to itself) becomes a marker."""
    return tuple("nan" if math.isnan(v) else v for v in values)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _pose(values: Sequence[float]) -> tuple[float, ...]:
    pose = tuple(float(v) for v in values)
    if len(pose) < 2:
        raise ValueError(f"a pose needs at least x and y, got {len(pose)} value(s)")
    return pose


class MapStats:
    def __init__(self, mode: Mode, calibration_placeholder: bool = False) -> None:
        if type(mode) is not Mode:
            raise TypeError("mode must be a Mode")
        self._mode = mode
        self._placeholder = require_bool(calibration_placeholder, name="calibration_placeholder")
        self._graph: tuple[tuple[int, tuple[object, ...]], ...] | None = None  # id order; None = no graph yet
        self._changed_s: float | None = None
        self._keyframes = 0
        self._path_m = 0.0
        self._closures = 0

    # ----------------------------------------------------------------- inputs
    def on_graph(
        self,
        ids: Sequence[int],
        poses: Sequence[Sequence[float]],
        now_s: float,
        links: Iterable[tuple[int, int, int]] = (),
    ) -> None:
        """The latest optimized graph. poses[i] belongs to ids[i] and starts with x, y; any further values
        (z, quaternion) only take part in change detection. links are (from_id, to_id, type) of the graph.
        A malformed graph raises and changes nothing."""
        now_s = require_finite(now_s, name="now_s")
        if len(ids) != len(poses):
            raise ValueError(f"{len(ids)} ids but {len(poses)} poses")
        entries = [(int(i), _pose(p)) for i, p in zip(ids, poses)]
        links = [(int(a), int(b), int(k)) for a, b, k in links]
        ordered = sorted((e for e in entries if e[0] > 0), key=lambda e: e[0])
        graph = tuple((i, _key(p)) for i, p in ordered)
        self._closures = closure_pairs(links)  # the graph holds every link: a fresh database simply has fewer
        if graph == self._graph:
            return
        self._graph = graph
        self._changed_s = now_s
        self._keyframes = len(graph)
        self._path_m = path_length([(pose[0], pose[1]) for _, pose in ordered])

    # ----------------------------------------------------------------- output
    def snapshot(self, now_s: float, db_bytes: int | None) -> dict[str, object]:
        now_s = require_finite(now_s, name="now_s")
        if db_bytes is not None and (type(db_bytes) is not int or db_bytes < 0):
            raise ValueError(f"db_bytes must be an int >= 0 or None, got {db_bytes!r}")
        age = None if (self._changed_s is None or self._mode is Mode.LOCALIZE) else max(0.0, now_s - self._changed_s)
        return {
            "keyframes": self._keyframes,
            "loop_closures": self._closures,
            "path_length_m": _finite_or_none(self._path_m),
            "db_bytes": db_bytes,
            "last_update_age_s": None if age is None else _finite_or_none(age),
            "mode": self._mode.value,
            "calibration_placeholder": self._placeholder,
        }
