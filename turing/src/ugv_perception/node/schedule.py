"""Which frames get segmentation. Pure: no clock, no rclpy. The caller passes the time."""

from __future__ import annotations

import math

DEFAULT_MASK_PERIOD_S = 0.25
_NS = 1_000_000_000


def seg_due(
    now_ns: int,
    last_seg_ns: int | None,
    mask_period_s: float = DEFAULT_MASK_PERIOD_S,
) -> bool:
    """True when this frame should be segmented.

    Depth runs on every frame. Segmentation runs on the first frame, and after that on the first frame
    that arrives `mask_period_s` or more after the previous segmentation started. A period of 0 segments
    every frame. Both times come from the same monotonic clock; a clock that went backwards counts as due.
    """
    if type(now_ns) is not int:
        raise TypeError("now_ns must be a Python int")
    if last_seg_ns is not None and type(last_seg_ns) is not int:
        raise TypeError("last_seg_ns must be a Python int or None")
    if type(mask_period_s) not in (int, float):
        raise TypeError("mask_period_s must be a number")
    if not math.isfinite(mask_period_s) or mask_period_s < 0:
        raise ValueError("mask_period_s must be finite and >= 0")
    if last_seg_ns is None:
        return True
    elapsed_ns = now_ns - last_seg_ns
    return elapsed_ns < 0 or elapsed_ns >= round(mask_period_s * _NS)
