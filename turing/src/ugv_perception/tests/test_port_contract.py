"""T01 contract tests — numeric arrays and headers, no camera, no RGB scenes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ugv_perception.port import (
    CANONICAL,
    CONF_ENCODING,
    HAZARD,
    MASK_ENCODING,
    TRAVERSABLE,
    UNKNOWN,
    CanonicalMask,
    FrameHeader,
    assert_aligned,
    assert_camera_info_pair,
    assert_canonical,
    make_mask,
)


def _classes(*rows: list[int]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8)


def _conf(hw: tuple[int, int], value: float = 1.0) -> np.ndarray:
    return np.full(hw, value, dtype=np.float32)


def _ok_mask(**kwargs) -> CanonicalMask:
    classes = kwargs.pop("classes", _classes([0, 1], [2, 0]))
    defaults = dict(
        stamp_ns=1_000,
        frame_id="camera_optical",
        classes=classes,
        confidence=kwargs.pop("confidence", _conf(classes.shape)),
        producer_ok=True,
        age_s=0.0,
        scale=1.0,
    )
    defaults.update(kwargs)
    return make_mask(**defaults)


class _CamInfo:
    """Pairing fixture: has .header.frame_id and garbage K so T01 must ignore K."""

    def __init__(self, frame_id: str) -> None:
        self.header = SimpleNamespace(frame_id=frame_id)
        self.k = [float("nan")] * 9
        self.d = [99.0]


# --- I1–I2 ---


def test_i1_i2_all_unknown_pass() -> None:
    m = _ok_mask(classes=_classes([0, 0], [0, 0]))
    assert set(np.unique(m.classes).tolist()) <= CANONICAL
    assert MASK_ENCODING == "mono8"


def test_i1_i2_all_traversable_pass() -> None:
    m = _ok_mask(classes=_classes([1, 1], [1, 1]))
    assert np.all(m.classes == TRAVERSABLE)


def test_i1_i2_all_hazard_pass() -> None:
    m = _ok_mask(classes=_classes([2, 2], [2, 2]))
    assert np.all(m.classes == HAZARD)


def test_i1_i2_mixed_pass() -> None:
    m = _ok_mask(classes=_classes([0, 1], [2, 1]))
    assert set(np.unique(m.classes).tolist()) == {0, 1, 2}


@pytest.mark.parametrize("bad", [3, 255])
def test_i1_i2_illegal_uint8_pixel_fails(bad: int) -> None:
    classes = _classes([0, 1], [2, bad])
    with pytest.raises(ValueError):
        _ok_mask(classes=classes)


def test_i1_i2_negative_pixel_fails_on_dtype() -> None:
    classes = np.array([[0, 1], [2, -1]], dtype=np.int16)
    with pytest.raises(TypeError):
        _ok_mask(classes=classes)


def test_i1_not_2d_fails() -> None:
    with pytest.raises(ValueError):
        assert_canonical(np.zeros((2, 2, 1), dtype=np.uint8))


# --- I3 ---


def test_i3_confidence_above_one_fails() -> None:
    classes = _classes([0, 1], [2, 0])
    conf = _conf(classes.shape, 1.0001)
    with pytest.raises(ValueError):
        _ok_mask(classes=classes, confidence=conf)


def test_i3_confidence_nan_fails() -> None:
    classes = _classes([0, 1], [2, 0])
    conf = _conf(classes.shape)
    conf[0, 0] = np.nan
    with pytest.raises(ValueError):
        _ok_mask(classes=classes, confidence=conf)


def test_i3_mismatched_hw_fails() -> None:
    classes = _classes([0, 1], [2, 0])
    with pytest.raises(ValueError):
        _ok_mask(classes=classes, confidence=_conf((3, 3)))


def test_i3_float64_conf_fails() -> None:
    classes = _classes([0, 1], [2, 0])
    conf = np.ones(classes.shape, dtype=np.float64)
    with pytest.raises(TypeError):
        _ok_mask(classes=classes, confidence=conf)


def test_i3_encoding_name() -> None:
    assert CONF_ENCODING == "32FC1"


# --- I4 ---


def test_i4_stamp_zero_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(stamp_ns=0)


def test_i4_stamp_negative_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(stamp_ns=-1)


@pytest.mark.parametrize("bad", [1.5, np.int64(1000), True])
def test_i4_stamp_non_python_int_fails(bad: object) -> None:
    with pytest.raises(TypeError):
        _ok_mask(stamp_ns=bad)


def test_i4_age_nan_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(age_s=float("nan"))


def test_i4_age_inf_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(age_s=float("inf"))


def test_i4_age_negative_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(age_s=-0.1)


# --- I5 ---


def test_i5_empty_frame_id_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(frame_id="")


# --- I6 ---


def test_i6_scale_zero_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(scale=0.0)


def test_i6_scale_negative_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(scale=-1.0)


def test_i6_scale_not_one_fails_v1() -> None:
    with pytest.raises(ValueError):
        _ok_mask(scale=0.5)


def test_i6_source_hw_mismatch_fails() -> None:
    classes = _classes([0, 1], [2, 0])
    with pytest.raises(ValueError):
        _ok_mask(classes=classes, source_hw=(4, 4))


def test_i6_source_hw_match_passes() -> None:
    classes = _classes([0, 1], [2, 0])
    m = _ok_mask(classes=classes, source_hw=classes.shape)
    assert m.scale == 1.0


def test_i6_non_uniform_ratios_fail() -> None:
    classes = _classes([0, 1], [2, 0])  # 2x2
    with pytest.raises(ValueError):
        _ok_mask(classes=classes, source_hw=(2, 4))


def test_i6_scale_int_one_fails_must_be_float() -> None:
    with pytest.raises(TypeError):
        _ok_mask(scale=1)


# --- I7 ---


def test_i7_mismatched_stamp_on_conf_header_fails() -> None:
    header = FrameHeader(stamp_ns=1000, frame_id="camera_optical")
    other = FrameHeader(stamp_ns=2000, frame_id="camera_optical")
    classes = _classes([0, 1], [2, 0])
    with pytest.raises(ValueError):
        assert_aligned(classes, _conf(classes.shape), header, other)


def test_i7_mismatched_frame_id_on_conf_header_fails() -> None:
    header = FrameHeader(stamp_ns=1000, frame_id="camera_optical")
    other = FrameHeader(stamp_ns=1000, frame_id="other_optical")
    classes = _classes([0, 1], [2, 0])
    with pytest.raises(ValueError):
        assert_aligned(classes, _conf(classes.shape), header, other)


def test_i7_matching_conf_header_passes() -> None:
    m = _ok_mask(
        confidence_header=FrameHeader(stamp_ns=1000, frame_id="camera_optical")
    )
    assert m.header.stamp_ns == 1000
    assert m.header.frame_id == "camera_optical"


# --- I8 ---


def test_i8_producer_ok_false_yields_valid_false() -> None:
    m = _ok_mask(producer_ok=False)
    assert m.valid is False
    assert type(m.valid) is bool


def test_i8_producer_ok_true_yields_valid_true() -> None:
    m = _ok_mask(producer_ok=True)
    assert m.valid is True
    assert type(m.valid) is bool


@pytest.mark.parametrize("bad", [1, "yes", np.True_])
def test_i8_truthy_producer_ok_raises_typeerror(bad: object) -> None:
    with pytest.raises(TypeError):
        _ok_mask(producer_ok=bad)


def test_i8_illegal_array_raises_not_valid_false() -> None:
    classes = _classes([0, 1], [2, 3])
    with pytest.raises(ValueError):
        make_mask(
            stamp_ns=1000,
            frame_id="camera_optical",
            classes=classes,
            confidence=_conf(classes.shape),
            producer_ok=False,
            age_s=0.0,
        )


# --- I9 ---


def test_i9_mismatched_camera_frame_fails() -> None:
    with pytest.raises(ValueError):
        _ok_mask(camera_info=_CamInfo("other_optical"))


def test_i9_matching_frame_passes_ignoring_k() -> None:
    info = _CamInfo("camera_optical")
    m = _ok_mask(camera_info=info)
    assert m.header.frame_id == "camera_optical"
    assert np.isnan(info.k[0])  # K was never a reason to fail


# --- I10 ---


def test_i10_unknown_not_rewritten_to_traversable() -> None:
    classes = _classes([0, 0], [0, 0])
    before = classes.copy()
    m = _ok_mask(classes=classes)
    assert np.array_equal(m.classes, before)
    assert not np.any(m.classes == TRAVERSABLE)
    assert_canonical(m.classes)
    assert np.array_equal(m.classes, before)


def test_i10_assert_canonical_does_not_mutate() -> None:
    classes = _classes([0, 1], [2, 0])
    snapshot = classes.copy()
    assert_canonical(classes)
    assert np.array_equal(classes, snapshot)


# --- construction extras ---


def test_make_mask_does_not_copy_arrays() -> None:
    classes = _classes([0, 1], [2, 0])
    conf = _conf(classes.shape)
    m = _ok_mask(classes=classes, confidence=conf)
    assert m.classes is classes
    assert m.confidence is conf


def test_port_meta_msg_has_required_fields_only() -> None:
    from pathlib import Path

    msg = Path(__file__).resolve().parents[1] / "port" / "port_meta.msg"
    fields = []
    for line in msg.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            fields.append(stripped)
    assert fields == [
        "std_msgs/Header header",
        "bool valid",
        "float32 age",
        "float32 scale",
    ]
