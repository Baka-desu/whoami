"""T02 decode_frame tests — message fixtures, no camera, no ROS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.ingest import CameraInfoView, ImageView, decode_frame

_H, _W = 2, 2
_K = (400.0, 0.0, 1.0, 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)
_I3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _rgb_bytes(
    pixels: list[tuple[int, int, int]], *, step: int | None = None, pad: int = 0
) -> bytearray:
    row_w = _W * 3
    stride = row_w if step is None else step
    buf = bytearray()
    for r in range(_H):
        row = bytearray()
        for c in range(_W):
            row.extend(pixels[r * _W + c])
        row.extend(bytes([pad]) * (stride - row_w))
        buf.extend(row)
    return buf


def _image(**kwargs) -> ImageView:
    pixels = [(10, 20, 30), (40, 50, 60), (70, 80, 90), (1, 2, 3)]
    defaults = dict(
        stamp_ns=1_000,
        frame_id="camera_optical",
        height=_H,
        width=_W,
        encoding="rgb8",
        step=_W * 3,
        data=_rgb_bytes(pixels),
    )
    defaults.update(kwargs)
    return ImageView(**defaults)


def _info(**kwargs) -> CameraInfoView:
    defaults = dict(
        stamp_ns=9_999,
        frame_id="camera_optical",
        height=_H,
        width=_W,
        k=_K,
    )
    defaults.update(kwargs)
    return CameraInfoView(**defaults)


def test_s1_stamp_zero_raises() -> None:
    with pytest.raises(TypeError):
        decode_frame(_image(stamp_ns=0), _info())


def test_s1_stamp_numpy_raises() -> None:
    with pytest.raises(TypeError):
        decode_frame(_image(stamp_ns=np.int64(1000)), _info())


def test_s1_stamp_true_raises() -> None:
    with pytest.raises(TypeError):
        decode_frame(_image(stamp_ns=True), _info())


def test_s2_mismatched_frame_id_raises() -> None:
    with pytest.raises(ValueError):
        decode_frame(_image(), _info(frame_id="other_optical"))


def test_s3_mono8_raises() -> None:
    with pytest.raises(ValueError):
        decode_frame(_image(encoding="mono8"), _info())


def test_s3_rgb8_ok() -> None:
    frame = decode_frame(_image(), _info())
    assert frame.rgb.shape == (_H, _W, 3)
    assert frame.rgb.dtype == np.uint8
    assert tuple(frame.rgb[0, 0]) == (10, 20, 30)


def test_s5_step_zero_raises() -> None:
    with pytest.raises(ValueError):
        decode_frame(_image(step=0), _info())


def test_s5_step_too_small_raises() -> None:
    with pytest.raises(ValueError):
        decode_frame(_image(step=_W * 3 - 1), _info())


def test_s5_padded_step_strips_pad() -> None:
    step = _W * 3 + 1
    data = _rgb_bytes(
        [(10, 20, 30), (40, 50, 60), (70, 80, 90), (1, 2, 3)], step=step, pad=99
    )
    frame = decode_frame(_image(step=step, data=data), _info())
    assert frame.rgb.shape == (_H, _W, 3)
    assert tuple(frame.rgb[0, 0]) == (10, 20, 30)
    assert tuple(frame.rgb[0, 1]) == (40, 50, 60)
    assert 99 not in frame.rgb.reshape(-1)
    assert frame.rgb.size == _H * _W * 3


def test_s6_missing_info_raises() -> None:
    with pytest.raises(TypeError):
        decode_frame(_image(), None)  # type: ignore[arg-type]


def test_s6_k_zeros_raises() -> None:
    with pytest.raises(ValueError):
        decode_frame(_image(), _info(k=(0.0,) * 9))


def test_s6_k_i3_raises() -> None:
    with pytest.raises(ValueError):
        decode_frame(_image(), _info(k=_I3))


def test_s6_fx_zero_raises() -> None:
    k = (0.0, 0.0, 1.0, 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        decode_frame(_image(), _info(k=k))


def test_s6_fy_zero_raises() -> None:
    k = (400.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        decode_frame(_image(), _info(k=k))


def test_s6_nonfinite_k_raises() -> None:
    k = (400.0, 0.0, float("nan"), 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        decode_frame(_image(), _info(k=k))


def test_s7_size_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        decode_frame(_image(), _info(height=480, width=640))


def test_s8_camera_info_stamp_ignored() -> None:
    frame = decode_frame(_image(stamp_ns=1_000), _info(stamp_ns=9_999))
    assert frame.stamp_ns == 1_000
    assert frame.frame_id == "camera_optical"


def test_s12_bgr8_swap() -> None:
    pixels = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
    data = _rgb_bytes(pixels)
    frame = decode_frame(_image(encoding="bgr8", data=data), _info())
    assert tuple(frame.rgb[0, 0]) == (3, 2, 1)


def test_s15_rgb_does_not_alias_data() -> None:
    data = _rgb_bytes([(10, 20, 30), (40, 50, 60), (70, 80, 90), (1, 2, 3)])
    frame = decode_frame(_image(data=data), _info())
    before = frame.rgb.copy()
    data[0] = 255
    assert np.array_equal(frame.rgb, before)
    assert frame.rgb[0, 0, 0] == 10


def test_s9_ingest_imports_clean() -> None:
    root = Path(__file__).resolve().parents[1] / "ingest"
    banned = ("rclpy", "cv2", "openvino", "adapter.pack", "compose")
    for py in root.glob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                for name in banned:
                    assert name not in stripped, f"{py.name}: {stripped}"


def test_s10_no_dummy_or_v4l2() -> None:
    root = Path(__file__).resolve().parents[1] / "ingest"
    for py in root.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "DummySource" not in text
        assert "V4L2" not in text
        assert "LiveCameraSource" not in text


def test_s14_subscribe_is_ros_bridge_not_v4l2() -> None:
    from ugv_perception.ingest.ros_bridge import image_msg_to_view
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Time

    msg = Image()
    t = Time()
    t.sec = 1
    t.nanosec = 0
    msg.header = Header(stamp=t, frame_id="camera_optical")
    msg.height = 1
    msg.width = 1
    msg.encoding = "rgb8"
    msg.step = 3
    msg.data = bytes([9, 8, 7])
    view = image_msg_to_view(msg)
    assert view.stamp_ns == 1_000_000_000
    assert view.data == bytes([9, 8, 7])
