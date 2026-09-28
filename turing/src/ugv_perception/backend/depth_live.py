"""Startup compile of the DA3 IR. Absent IR means T08 stays off."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ugv_perception.adapter.output import AdapterError
from ugv_perception.backend.openvino_gpu import OpenVinoGpuTensorBackend
from ugv_perception.depth.geometry import (
    backproject,
    focal_model,
    hole_safe_resize,
    k_model,
    meters_from_raw,
    model_hw,
    preprocess_nchw,
)


class DepthChannel:
    def __init__(self, backend: OpenVinoGpuTensorBackend) -> None:
        self._backend = backend

    def points(self, rgb: np.ndarray, k: tuple[float, ...] | np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise TypeError("rgb must be uint8 HWC")
        k_cam = np.asarray(k, dtype=np.float64).reshape(3, 3)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        k_m, (mh, mw) = k_model(k_cam, (height, width))
        if model_hw(height, width) != (mh, mw):
            raise AdapterError("model size disagrees with K_model")
        self._backend.ensure_hw(mh, mw)
        blob, sized = preprocess_nchw(rgb)
        if sized != (mh, mw):
            raise AdapterError("preprocess size disagrees with K_model")
        outputs = self._backend.run_all(blob)
        if len(outputs) < 2:
            raise AdapterError("DA3 IR must return depth_raw and sky")
        raw = np.squeeze(outputs[0])
        sky = np.squeeze(outputs[1])
        depth_m = meters_from_raw(raw, focal_model(k_m))
        on_camera = hole_safe_resize(depth_m, sky, (height, width))
        return backproject(on_camera, k_cam)


def build_depth_channel(root: Path) -> DepthChannel | None:
    xml = root / "weights" / "da3metric-large.xml"
    if not xml.is_file():
        return None
    backend = OpenVinoGpuTensorBackend()
    backend.load(str(xml), input_hw=None)
    return DepthChannel(backend)
