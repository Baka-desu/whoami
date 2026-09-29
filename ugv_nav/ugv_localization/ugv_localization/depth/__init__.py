"""Depth input gate (DA3 depth Image from Dev 1) + depth accuracy metrics. No ROS."""

from ugv_localization.depth.gate import (
    DepthFrame,
    DepthRejection,
    check_depth_frame,
    depth_coverage,
    depth_values,
)
from ugv_localization.depth.metrics import DepthErrorAccumulator

__all__ = [
    "DepthErrorAccumulator",
    "DepthFrame",
    "DepthRejection",
    "check_depth_frame",
    "depth_coverage",
    "depth_values",
]
