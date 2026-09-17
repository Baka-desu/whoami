"""ROS-agnostic image / CameraInfo views. Not a camera driver."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageView:
    stamp_ns: int
    frame_id: str
    height: int
    width: int
    encoding: str
    step: int
    data: bytes | bytearray


@dataclass(frozen=True, slots=True)
class CameraInfoView:
    stamp_ns: int
    frame_id: str
    height: int
    width: int
    k: tuple[float, ...]
