"""Convert sensor_msgs Image/CameraInfo → ingest views. Copies buffers."""

from __future__ import annotations

from ugv_perception.ingest.msgs import CameraInfoView, ImageView

_NS = 1_000_000_000


def stamp_to_ns(stamp: object) -> int:
    sec = stamp.sec
    nsec = stamp.nanosec
    if type(sec) is not int or type(nsec) is not int:
        raise TypeError("ROS stamp sec/nanosec must be Python ints")
    if sec < 0 or nsec < 0:
        raise ValueError("ROS stamp must be non-negative")
    return sec * _NS + nsec


def image_msg_to_view(msg: object) -> ImageView:
    header = msg.header
    stamp_ns = stamp_to_ns(header.stamp)
    frame_id = header.frame_id
    data = bytes(msg.data)
    return ImageView(
        stamp_ns=stamp_ns,
        frame_id=frame_id,
        height=int(msg.height),
        width=int(msg.width),
        encoding=str(msg.encoding),
        step=int(msg.step),
        data=data,
    )


def camera_info_msg_to_view(msg: object) -> CameraInfoView:
    header = msg.header
    stamp_ns = stamp_to_ns(header.stamp)
    k_raw = list(msg.k)
    if len(k_raw) < 9:
        raise ValueError("CameraInfo.k must have 9 values")
    k = tuple(float(k_raw[i]) for i in range(9))
    return CameraInfoView(
        stamp_ns=stamp_ns,
        frame_id=header.frame_id,
        height=int(msg.height),
        width=int(msg.width),
        k=k,
    )
