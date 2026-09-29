"""Depth frame gate for RGB-D SLAM input (mindmap D7/D8, interfaces.md "Depth input").

Dev 1 publishes Depth Anything 3 Metric depth as sensor_msgs/Image 32FC1, meters, NaN holes,
registered to the RGB camera (same frame_id, same size, source image stamp). This kernel decides
whether a frame honours that contract. It never repairs a frame — RTAB-Map gets the raw topic;
the verdict only feeds /ugv/pose_valid. No ROS imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ugv_localization.common.checks import NS_PER_S, require_stamp


class DepthRejection(str, Enum):
    NO_CAMERA_INFO = "no_camera_info"
    ENCODING = "encoding"
    FRAME_MISMATCH = "frame_mismatch"
    SIZE_MISMATCH = "size_mismatch"
    FUTURE_STAMP = "future_stamp"
    LOW_COVERAGE = "low_coverage"


@dataclass(frozen=True, slots=True)
class DepthFrame:
    stamp_ns: int
    frame_id: str
    encoding: str
    width: int
    height: int
    coverage: float  # fraction of (sampled) pixels with finite depth > 0


def check_depth_frame(
    frame: DepthFrame,
    *,
    camera_frame: str | None,
    camera_size: tuple[int, int] | None,
    now_ns: int,
    min_coverage: float,
    max_future_s: float,
) -> DepthRejection | None:
    stamp = require_stamp(frame.stamp_ns, name="frame.stamp_ns")
    now_ns = require_stamp(now_ns, name="now_ns")
    if camera_frame is None or camera_size is None:
        return DepthRejection.NO_CAMERA_INFO
    if frame.encoding != "32FC1":
        return DepthRejection.ENCODING
    if frame.frame_id != camera_frame:
        return DepthRejection.FRAME_MISMATCH
    if (frame.width, frame.height) != tuple(camera_size):
        return DepthRejection.SIZE_MISMATCH
    if stamp > now_ns + int(max_future_s * NS_PER_S):
        return DepthRejection.FUTURE_STAMP
    if not frame.coverage >= min_coverage:
        return DepthRejection.LOW_COVERAGE
    return None


def depth_values(
    data: bytes, *, width: int, height: int, step: int, is_bigendian: bool, stride: int
) -> np.ndarray:
    """Every stride-th pixel (rows and columns) of a 32FC1 image buffer, as float32."""
    if width <= 0 or height <= 0 or stride <= 0 or step < 4 * width:
        raise ValueError(f"bad depth geometry width={width} height={height} step={step} stride={stride}")
    if len(data) < step * height:
        raise ValueError(f"depth buffer has {len(data)} bytes, need {step * height}")
    dtype = np.dtype(">f4" if is_bigendian else "<f4")
    rows = np.frombuffer(data, dtype=np.uint8, count=step * height).reshape(height, step)
    img = rows[:, : 4 * width].copy().view(dtype).reshape(height, width)
    return img[::stride, ::stride].astype(np.float32).ravel()


def depth_coverage(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    ok = np.isfinite(values) & (values > 0.0)
    return float(np.count_nonzero(ok)) / float(values.size)
