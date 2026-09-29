"""Depth frame gate: DA3 depth Image (32FC1, meters, NaN holes) → usable for RGB-D SLAM or not."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ugv_localization.depth import DepthFrame, DepthRejection, check_depth_frame, depth_coverage, depth_values

_NS = 1_000_000_000
_T0 = 100 * _NS


def _frame(**over) -> DepthFrame:
    base = dict(
        stamp_ns=_T0,
        frame_id="camera_optical",
        encoding="32FC1",
        width=4,
        height=2,
        coverage=0.9,
    )
    base.update(over)
    return DepthFrame(**base)


def _check(frame: DepthFrame, **over):
    kw = dict(camera_frame="camera_optical", camera_size=(4, 2), now_ns=_T0, min_coverage=0.5, max_future_s=0.05)
    kw.update(over)
    return check_depth_frame(frame, **kw)


def test_g1_good_frame_passes() -> None:
    assert _check(_frame()) is None


def test_g2_encoding_must_be_32fc1_meters() -> None:
    assert _check(_frame(encoding="16UC1")) is DepthRejection.ENCODING  # mm ints: not the contract
    assert _check(_frame(encoding="mono8")) is DepthRejection.ENCODING


def test_g3_frame_must_match_camera() -> None:
    assert _check(_frame(frame_id="base_link")) is DepthRejection.FRAME_MISMATCH


def test_g4_size_must_match_camera() -> None:
    assert _check(_frame(width=2, height=1)) is DepthRejection.SIZE_MISMATCH


def test_g5_future_stamp_rejected() -> None:
    assert _check(_frame(stamp_ns=_T0 + _NS)) is DepthRejection.FUTURE_STAMP


def test_g6_low_coverage_rejected() -> None:
    assert _check(_frame(coverage=0.2)) is DepthRejection.LOW_COVERAGE


def test_g7_unknown_camera_rejected() -> None:
    assert _check(_frame(), camera_frame=None, camera_size=None) is DepthRejection.NO_CAMERA_INFO


def test_g8_coverage_counts_finite_positive_only() -> None:
    v = np.array([1.0, math.nan, math.inf, -1.0, 0.0, 2.5, 3.0, math.nan], dtype=np.float32)
    assert depth_coverage(v) == pytest.approx(3 / 8)
    assert depth_coverage(np.array([], dtype=np.float32)) == 0.0


def test_g9_values_from_bytes_with_row_padding_and_stride() -> None:
    img = np.arange(12, dtype="<f4").reshape(3, 4)
    padded = np.zeros((3, 5), dtype="<f4")  # step = 5 floats: one padding float per row
    padded[:, :4] = img
    vals = depth_values(padded.tobytes(), width=4, height=3, step=20, is_bigendian=False, stride=1)
    assert vals.tolist() == img.ravel().tolist()
    sub = depth_values(padded.tobytes(), width=4, height=3, step=20, is_bigendian=False, stride=2)
    assert sub.tolist() == img[::2, ::2].ravel().tolist()


def test_g10_values_big_endian() -> None:
    img = np.array([[1.5, 2.5]], dtype=">f4")
    vals = depth_values(img.tobytes(), width=2, height=1, step=8, is_bigendian=True, stride=1)
    assert vals.tolist() == [1.5, 2.5]


def test_g11_truncated_buffer_raises() -> None:
    with pytest.raises(ValueError, match="bytes"):
        depth_values(b"\x00" * 7, width=2, height=1, step=8, is_bigendian=False, stride=1)
