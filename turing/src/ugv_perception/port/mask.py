"""CanonicalMask and make_mask — the only public constructor for a legal port sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ugv_perception.port.header import FrameHeader
from ugv_perception.port.validate import (
    assert_age_s,
    assert_aligned,
    assert_camera_info_pair,
    assert_producer_ok,
    assert_scale,
)


@dataclass(frozen=True, slots=True)
class CanonicalMask:
    header: FrameHeader
    classes: NDArray[np.uint8]
    confidence: NDArray[np.float32]
    valid: bool
    age_s: float
    scale: float


def make_mask(
    *,
    stamp_ns: int,
    frame_id: str,
    classes: NDArray[np.uint8],
    confidence: NDArray[np.float32],
    producer_ok: bool,
    age_s: float,
    scale: float = 1.0,
    source_hw: tuple[int, int] | None = None,
    camera_info: Any | None = None,
    confidence_header: FrameHeader | None = None,
) -> CanonicalMask:
    """Build a CanonicalMask or raise. Does not coerce pixels. Does not copy arrays."""
    assert_producer_ok(producer_ok)
    header = FrameHeader(stamp_ns=stamp_ns, frame_id=frame_id)
    assert_age_s(age_s)
    assert_aligned(classes, confidence, header, confidence_header)
    assert_scale(scale, classes.shape, source_hw)
    if camera_info is not None:
        assert_camera_info_pair(header, camera_info)
    # valid = producer_ok with no bool() coercion — producer_ok is a real bool.
    return CanonicalMask(
        header=header,
        classes=classes,
        confidence=confidence,
        valid=producer_ok,
        age_s=age_s,
        scale=scale,
    )
