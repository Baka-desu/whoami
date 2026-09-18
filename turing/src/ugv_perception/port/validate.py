"""Fail-closed port invariants I1–I10. No coercion. No K/D checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ugv_perception.port.header import FrameHeader
from ugv_perception.port.ids import CANONICAL

_V1_SCALE = 1.0


def assert_canonical(classes: NDArray[np.uint8]) -> None:
    """I1–I2: uint8 2-D, every pixel in {0,1,2}. Does not rewrite 0→1 (I10)."""
    if not isinstance(classes, np.ndarray):
        raise TypeError(f"classes must be a numpy ndarray, got {type(classes).__name__}")
    if classes.dtype != np.uint8:
        raise TypeError(f"classes dtype must be uint8, got {classes.dtype}")
    if classes.ndim != 2:
        raise ValueError(f"classes must be 2-D, got ndim={classes.ndim}")
    if classes.size == 0:
        raise ValueError("classes must be non-empty")
    # uint8 ≥ 0; values outside {0,1,2} are 3..255
    if np.any(classes > 2):
        raise ValueError("classes pixels must be in {0,1,2}")


def assert_confidence(
    confidence: NDArray[np.float32], hw: tuple[int, int]
) -> None:
    """I3: float32, same H×W as classes, finite, in [0,1]."""
    if not isinstance(confidence, np.ndarray):
        raise TypeError(
            f"confidence must be a numpy ndarray, got {type(confidence).__name__}"
        )
    if confidence.dtype != np.float32:
        raise TypeError(f"confidence dtype must be float32, got {confidence.dtype}")
    if confidence.shape != hw:
        raise ValueError(
            f"confidence shape {confidence.shape} must match classes HW {hw}"
        )
    if not np.isfinite(confidence).all():
        raise ValueError("confidence must be finite")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("confidence must be in [0,1]")


def assert_header(header: FrameHeader) -> None:
    """I4–I5: already enforced by FrameHeader.__post_init__; re-check identity."""
    if type(header) is not FrameHeader:
        raise TypeError(f"header must be FrameHeader, got {type(header).__name__}")
    if type(header.stamp_ns) is not int or header.stamp_ns <= 0:
        raise TypeError("stamp_ns must be a Python int > 0")
    if type(header.frame_id) is not str or header.frame_id == "":
        raise ValueError("frame_id must be a non-empty str")


def assert_age_s(age_s: float) -> None:
    """I4: informational age, finite and >= 0. Not a freshness decision."""
    if type(age_s) is bool or type(age_s) not in (int, float):
        raise TypeError(f"age_s must be int or float, got {type(age_s).__name__}")
    if not math.isfinite(age_s) or age_s < 0:
        raise ValueError("age_s must be finite and >= 0 (informational)")


def assert_scale(
    scale: float,
    mask_hw: tuple[int, int],
    source_hw: tuple[int, int] | None,
) -> None:
    """I6: v1 product scale is 1.0; isotropic if source HW is given."""
    if type(scale) is not float:
        raise TypeError(f"scale must be a Python float, got {type(scale).__name__}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and > 0")
    if scale != _V1_SCALE:
        raise ValueError("v1 product port requires scale == 1.0")
    if source_hw is None:
        return
    if type(source_hw) is not tuple or len(source_hw) != 2:
        raise TypeError("source_hw must be a (height, width) tuple")
    sh, sw = source_hw
    if type(sh) is not int or type(sw) is not int:
        raise TypeError("source_hw values must be Python ints")
    if sh <= 0 or sw <= 0:
        raise ValueError("source_hw must be positive")
    mh, mw = mask_hw
    if mh / sh != mw / sw:
        raise ValueError("scale must be isotropic (mask_h/src_h == mask_w/src_w)")
    if (mh, mw) != (sh, sw):
        raise ValueError("v1 scale 1.0 requires mask HW == source HW")
    if mh / sh != scale or mw / sw != scale:
        raise ValueError("axis ratios must equal scale")


def assert_aligned(
    classes: NDArray[np.uint8],
    confidence: NDArray[np.float32],
    header: FrameHeader,
    confidence_header: FrameHeader | None = None,
) -> None:
    """I7: same HW; optional second header must match stamp and frame_id."""
    assert_canonical(classes)
    assert_confidence(confidence, classes.shape)
    assert_header(header)
    if confidence_header is not None:
        assert_header(confidence_header)
        if (
            confidence_header.stamp_ns != header.stamp_ns
            or confidence_header.frame_id != header.frame_id
        ):
            raise ValueError("mask and confidence headers must be identical")


def assert_camera_info_pair(header: FrameHeader, camera_info: Any) -> None:
    """I9: frame_id identity only. Does not inspect K, D, size, or distortion."""
    assert_header(header)
    try:
        info_frame = camera_info.header.frame_id
    except AttributeError as exc:
        raise TypeError(
            "CameraInfo pairing requires .header.frame_id (no calibration fields)"
        ) from exc
    if type(info_frame) is not str or info_frame != header.frame_id:
        raise ValueError(
            "CameraInfo.header.frame_id must equal CanonicalMask.header.frame_id"
        )


def assert_producer_ok(producer_ok: bool) -> None:
    """I8: strict Python bool. No truthy coercion."""
    if type(producer_ok) is not bool:
        raise TypeError(
            f"producer_ok must be a Python bool, got {type(producer_ok).__name__}"
        )
