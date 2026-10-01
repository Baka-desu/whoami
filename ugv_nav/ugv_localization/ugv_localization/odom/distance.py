"""Distance travelled along the selected /odom → an odometry ESTIMATE, not a surveyed distance.

Input is odom_selector's /odom (continuous across source switches: the selector re-anchors).
With no wheel sensor that is RTAB-Map visual odometry on RGB + DA3 depth, so metric scale is
DA3's depth scale and every % of DA3 scale bias is a % of distance error (SENSOR_HONESTY.md).

  * Jitter floor: a step is counted only once the pose has left a min_step_m radius around the
    last counted point (the anchor). Slow motion still accumulates; standing still does not.
  * Jumps: a sample-to-sample move faster than max_speed_mps (+ min_step_m tolerance) is not
    driven distance (reset / relocalization artefact) → re-anchor there, count nothing.
  * Gaps: while visual odometry is lost, /odom stops; motion in that gap is not counted.
  * 2D (x, y): odometry is planar (Reg/Force3DoF), same as the dead-reckon budget.

Each counted metre is attributed to the source active at that time, so the label (basis) says
what the total is made of. No ROS imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ugv_localization.common.checks import NS_PER_S, require_positive_float, require_stamp
from ugv_localization.common.yamlio import load_yaml_mapping, require_exact_keys
from ugv_localization.odom.select import Source


class Basis(str, Enum):
    NONE = "none"  # nothing counted yet
    WHEEL = "wheel_odometry"  # every counted metre came from wheel odometry
    VISUAL = "visual_odometry_estimate"  # every counted metre came from visual odometry
    MIXED = "odometry_estimate"  # wheel + visual, or no source announced yet


@dataclass(frozen=True, slots=True)
class DistanceProfile:
    min_step_m: float
    max_speed_mps: float
    publish_rate_hz: float


def load_distance_profile(path: str | Path) -> DistanceProfile:
    data = load_yaml_mapping(path)
    names = ("min_step_m", "max_speed_mps", "publish_rate_hz")
    require_exact_keys(data, set(names), where=str(path))
    return DistanceProfile(**{n: require_positive_float(data[n], name=n) for n in names})


class DistanceTracker:
    def __init__(self, profile: DistanceProfile) -> None:
        if type(profile) is not DistanceProfile:
            raise TypeError("profile must be a DistanceProfile")
        self._p = profile
        self._source: Source | None = None
        self.reset()

    def reset(self) -> None:
        """Zero the total (reset service, or clock jumped backwards). Keeps the current source."""
        self._by_source: dict[Source | None, float] = {}
        self._anchor: tuple[float, float] | None = None
        self._prev: tuple[int, float, float] | None = None
        self.jumps_rejected = 0

    @property
    def total_m(self) -> float:
        return float(sum(self._by_source.values(), 0.0))  # always float: Float64 aborts on int

    @property
    def by_source_m(self) -> dict[Source | None, float]:
        return dict(self._by_source)

    @property
    def basis(self) -> Basis:
        counted = {s for s, m in self._by_source.items() if m > 0.0}
        if not counted:
            return Basis.NONE
        if counted == {Source.WHEEL}:
            return Basis.WHEEL
        if counted == {Source.VISUAL}:
            return Basis.VISUAL
        return Basis.MIXED

    def on_source(self, source: Source) -> None:
        """odom_selector's active source (/ugv/localization/odom_source)."""
        if type(source) is not Source:
            raise TypeError("source must be a Source")
        if self._source is None and None in self._by_source:
            # odom_selector announces its source before its first /odom, so metres counted before
            # the (latched) announcement reached us came from this first source (startup race).
            self._by_source[source] = self._by_source.get(source, 0.0) + self._by_source.pop(None)
        self._source = source

    def on_odom(self, stamp_ns: int, x: float, y: float) -> None:
        stamp_ns = require_stamp(stamp_ns, name="stamp_ns")
        if self._prev is not None and stamp_ns <= self._prev[0]:
            return  # duplicate / reordered sample
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        prev = self._prev
        self._prev = (stamp_ns, x, y)
        if prev is None or self._anchor is None:
            self._anchor = (x, y)
            return
        dt_s = (stamp_ns - prev[0]) / NS_PER_S
        if math.hypot(x - prev[1], y - prev[2]) > self._p.max_speed_mps * dt_s + self._p.min_step_m:
            self._anchor = (x, y)
            self.jumps_rejected += 1
            return
        d = math.hypot(x - self._anchor[0], y - self._anchor[1])
        if d >= self._p.min_step_m:
            self._by_source[self._source] = self._by_source.get(self._source, 0.0) + d
            self._anchor = (x, y)
