"""Image + CameraInfo → shipped ImageFrame. Copies pixels; does not alias data."""

from __future__ import annotations

import math

import numpy as np

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.ingest.msgs import CameraInfoView, ImageView

_ENCODINGS = frozenset({"rgb8", "bgr8"})
_I3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_BPP = 3


def decode_frame(image: ImageView, camera_info: CameraInfoView) -> ImageFrame:
    if camera_info is None:
        raise TypeError("CameraInfoView is required")
    if type(image) is not ImageView:
        raise TypeError("image must be an ImageView")
    if type(camera_info) is not CameraInfoView:
        raise TypeError("camera_info must be a CameraInfoView")

    if type(image.stamp_ns) is not int or image.stamp_ns <= 0:
        raise TypeError("image stamp_ns must be a Python int > 0")
    if type(image.frame_id) is not str or image.frame_id == "":
        raise ValueError("image frame_id must be a non-empty str")
    if type(camera_info.frame_id) is not str or camera_info.frame_id != image.frame_id:
        raise ValueError("CameraInfo.frame_id must equal image frame_id")

    if type(image.height) is not int or type(image.width) is not int:
        raise TypeError("image height and width must be Python ints")
    if image.height <= 0 or image.width <= 0:
        raise ValueError("image height and width must be > 0")
    if camera_info.height != image.height or camera_info.width != image.width:
        raise ValueError("CameraInfo height/width must equal image height/width")

    if type(image.encoding) is not str or image.encoding not in _ENCODINGS:
        raise ValueError("encoding must be rgb8 or bgr8")
    if type(image.step) is not int or image.step <= 0:
        raise ValueError("step must be a Python int > 0")
    row_bytes = image.width * _BPP
    if image.step < row_bytes:
        raise ValueError("step must be >= width * 3")
    data = image.data
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("image data must be bytes or bytearray")
    if len(data) < image.step * image.height:
        raise ValueError("image data shorter than step * height")

    _require_k(camera_info.k)

    rgb = np.empty((image.height, image.width, 3), dtype=np.uint8)
    for row in range(image.height):
        start = row * image.step
        chunk = bytes(data[start : start + row_bytes])
        rgb[row] = np.frombuffer(chunk, dtype=np.uint8).reshape(image.width, 3)
    if image.encoding == "bgr8":
        rgb = rgb[:, :, ::-1].copy()

    return ImageFrame(
        rgb=rgb,
        stamp_ns=image.stamp_ns,
        frame_id=image.frame_id,
    )


def _require_k(k: object) -> None:
    if type(k) is not tuple or len(k) != 9:
        raise TypeError("k must be a tuple of 9 finite values")
    vals: list[float] = []
    for i, item in enumerate(k):
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("k values must not be bool")
        try:
            v = float(item)
        except (TypeError, ValueError) as exc:
            raise TypeError("k values must be numeric") from exc
        if not math.isfinite(v):
            raise ValueError("k values must be finite")
        vals.append(v)
    if all(v == 0.0 for v in vals):
        raise ValueError("k must not be all-zero")
    if tuple(vals) == _I3:
        raise ValueError("k must not be I_3 (reserved placeholder)")
    if vals[0] <= 0.0 or vals[4] <= 0.0:
        raise ValueError("fx and fy must be > 0")
