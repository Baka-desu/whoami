"""Startup load of DA3. Absent IR and safetensors means T08 stays off."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np

from ugv_perception.adapter.output import AdapterError
from ugv_perception.backend.device import pick_tensor_backend
from ugv_perception.depth.geometry import (
    backproject,
    focal_model,
    hole_safe_resize,
    k_model,
    meters_from_raw,
    model_hw,
    preprocess_nchw,
)


def _no_timing(name: str) -> AbstractContextManager[None]:
    return nullcontext()


class DepthChannel:
    def __init__(self, backend: object) -> None:
        self._backend = backend

    def maps(
        self,
        rgb: np.ndarray,
        k: tuple[float, ...] | np.ndarray,
        stage: Callable[[str], AbstractContextManager[None]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Camera-sized meters (NaN holes) and unorganized XYZ. One infer.

        `stage(name)` is an optional timing hook returning a context manager. It sees
        "depth_infer" (backend sizing, preprocess, model run) and "depth_post"
        (meters, hole-safe resize, back-projection). It does not change the result.

        A backend with `run_depth_metres` (CUDA) does the preprocess and the meters/resize on its own
        device and returns the camera-sized depth; every other backend takes the numpy path below.
        """
        span = stage if stage is not None else _no_timing
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise TypeError("rgb must be uint8 HWC")
        k_cam = np.asarray(k, dtype=np.float64).reshape(3, 3)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        k_m, (mh, mw) = k_model(k_cam, (height, width))
        if model_hw(height, width) != (mh, mw):
            raise AdapterError("model size disagrees with K_model")
        on_device = getattr(self._backend, "run_depth_metres", None)
        if callable(on_device):
            on_camera = on_device(rgb, focal_model(k_m), (mh, mw), (height, width), span)
            with span("depth_post"):
                return on_camera, backproject(on_camera, k_cam)
        with span("depth_infer"):
            self._backend.ensure_hw(mh, mw)
            blob, sized = preprocess_nchw(rgb)
            if sized != (mh, mw):
                raise AdapterError("preprocess size disagrees with K_model")
            outputs = self._backend.run_all(blob)
        if len(outputs) < 2:
            raise AdapterError("DA3 must return depth_raw and sky")
        with span("depth_post"):
            raw = np.squeeze(outputs[0])
            sky = np.squeeze(outputs[1])
            depth_m = meters_from_raw(raw, focal_model(k_m))
            on_camera = hole_safe_resize(depth_m, sky, (height, width)).astype(np.float32)
            return on_camera, backproject(on_camera, k_cam)

    def points(self, rgb: np.ndarray, k: tuple[float, ...] | np.ndarray) -> np.ndarray:
        return self.maps(rgb, k)[1]


def build_depth_channel(root: Path, *, fp16: bool = False) -> DepthChannel | None:
    """DA3 on the best backend found, FP32 unless `fp16` (the node's `da3_fp16` opt-in) is set."""
    xml = root / "weights" / "da3metric-large.xml"
    folder = root / "weights" / "da3metric-large"
    backend = pick_tensor_backend(ir_xml=xml, safetensors_dir=folder, kind="da3", fp16=fp16)
    if backend is None:
        return None
    return DepthChannel(backend)
