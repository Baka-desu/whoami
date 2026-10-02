"""Map statistics kernel → /ugv/map/stats (the web viewer's statistics widget).

Fed with RTAB-Map's pose graph (rtabmap_msgs/MapGraph) and per-step Info; answers one flat dict of JSON
scalars (the client rejects a status that holds an object or an array):

  keyframes          int           poses in the latest graph (landmarks, negative ids, are not keyframes)
  loop_closures      int           loop-closure / proximity-detection matches seen since start, each matched node once
  path_length_m      float | None  polyline through the graph poses in id order, x/y only (None if not finite)
  db_bytes           int | None    size of the database file (the caller stats it; None = no file)
  last_update_age_s  float | None  seconds since the id list or any pose changed (None before the first graph)
  mode               str           mapping | localize
  calibration_placeholder bool     the camera calibration in use is a stand-in (camera.calib `placeholder`)

"Graph changed" is a change of content, not the arrival of a message, so a republished unchanged graph does not
look like progress. (Measured on the real stack: RTAB-Map publishes no graph at all while the robot stands still,
so the age keeps growing; the content test keeps it right if that ever changes.)

Pure: the node feeds messages and a clock. No ROS imports, no file access.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import pairwise

from ugv_localization.common.checks import require_bool, require_finite
from ugv_localization.modes import Mode


def path_length(poses: Iterable[tuple[float, float]]) -> float:
    """Length of the polyline through (x, y) poses in the order given. Fewer than two poses: 0.0."""
    return float(sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in pairwise(poses)))


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
        # One entry per matched node and kind of match. RTAB-Map reports a match in every step it makes one, and a
        # robot that stands still re-matches the same node in each of them (measured: ~2 per second): the same
        # matched id again is the same event, not a new loop closure.
        self._matches: set[tuple[str, int]] = set()
        self._graph: tuple[tuple[int, tuple[float, ...]], ...] | None = None  # id order; None = no graph yet
        self._changed_s: float | None = None
        self._keyframes = 0
        self._path_m = 0.0

    # ----------------------------------------------------------------- inputs
    def on_graph(self, ids: Sequence[int], poses: Sequence[Sequence[float]], now_s: float) -> None:
        """The latest optimized graph. poses[i] belongs to ids[i] and starts with x, y; any further values
        (z, quaternion) only take part in change detection. A malformed graph raises and changes nothing."""
        now_s = require_finite(now_s, name="now_s")
        if len(ids) != len(poses):
            raise ValueError(f"{len(ids)} ids but {len(poses)} poses")
        entries = [(int(i), _pose(p)) for i, p in zip(ids, poses)]
        graph = tuple(sorted((e for e in entries if e[0] > 0), key=lambda e: e[0]))
        if graph == self._graph:
            return
        self._graph = graph
        self._changed_s = now_s
        self._keyframes = len(graph)
        self._path_m = path_length([(pose[0], pose[1]) for _, pose in graph])

    def on_info(self, loop_closure_id: int, proximity_id: int) -> None:
        """One RTAB-Map step: loop_closure_id / proximity_detection_id are the matched node (0 = none)."""
        if loop_closure_id > 0:
            self._matches.add(("loop", int(loop_closure_id)))
        if proximity_id > 0:
            self._matches.add(("proximity", int(proximity_id)))

    # ----------------------------------------------------------------- output
    def snapshot(self, now_s: float, db_bytes: int | None) -> dict[str, object]:
        now_s = require_finite(now_s, name="now_s")
        if db_bytes is not None and (type(db_bytes) is not int or db_bytes < 0):
            raise ValueError(f"db_bytes must be an int >= 0 or None, got {db_bytes!r}")
        age = None if self._changed_s is None else max(0.0, now_s - self._changed_s)
        return {
            "keyframes": self._keyframes,
            "loop_closures": len(self._matches),
            "path_length_m": _finite_or_none(self._path_m),
            "db_bytes": db_bytes,
            "last_update_age_s": None if age is None else _finite_or_none(age),
            "mode": self._mode.value,
            "calibration_placeholder": self._placeholder,
        }
