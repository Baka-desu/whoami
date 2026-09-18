"""T02 ROS subscribe conversion — real sensor_msgs, no camera driver."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sensor_msgs")
pytest.importorskip("std_msgs")
pytest.importorskip("builtin_interfaces")

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

from ugv_perception.ingest.decode import decode_frame
from ugv_perception.ingest.ros_bridge import (
    camera_info_msg_to_view,
    image_msg_to_view,
    stamp_to_ns,
)

_K = (400.0, 0.0, 1.0, 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)


def _stamp(sec: int, nsec: int) -> Time:
    t = Time()
    t.sec = sec
    t.nanosec = nsec
    return t


def test_stamp_to_ns() -> None:
    t = _stamp(2, 500)
    assert stamp_to_ns(t) == 2_000_000_500
    assert type(stamp_to_ns(t)) is int


def test_stamp_true_raises() -> None:
    t = SimpleNamespace(sec=True, nanosec=0)
    try:
        stamp_to_ns(t)
        raise AssertionError("True sec must raise")
    except TypeError:
        pass


def test_image_msg_copied_and_decoded() -> None:
    msg = Image()
    msg.header = Header(stamp=_stamp(1, 0), frame_id="camera_optical")
    msg.height = 2
    msg.width = 2
    msg.encoding = "rgb8"
    msg.step = 6
    msg.data = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3])
    view = image_msg_to_view(msg)
    assert view.stamp_ns == 1_000_000_000
    assert view.frame_id == "camera_optical"
    msg.data = b"\x00" * 12
    info = CameraInfo()
    info.header = Header(stamp=_stamp(1, 0), frame_id="camera_optical")
    info.height = 2
    info.width = 2
    info.k = list(_K)
    civ = camera_info_msg_to_view(info)
    frame = decode_frame(view, civ)
    assert tuple(frame.rgb[0, 0]) == (10, 20, 30)
    assert type(frame.stamp_ns) is int
    assert frame.stamp_ns == 1_000_000_000
    assert isinstance(frame.rgb, np.ndarray)


def test_camera_info_k_tuple() -> None:
    info = CameraInfo()
    info.header = Header(stamp=_stamp(1, 0), frame_id="camera_optical")
    info.height = 2
    info.width = 2
    info.k = list(_K)
    civ = camera_info_msg_to_view(info)
    assert civ.k == _K
    assert type(civ.k[0]) is float
