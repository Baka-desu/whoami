"""OpenVINO 2026.4.0 GPU backend. Import openvino only inside load/run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ugv_perception.adapter.output import AdapterError, Instance
from ugv_perception.backend.instances import instances_from_engine

_DEVICE = "GPU"


class OpenVinoGpuBackend:
    id = "openvino_gpu"

    def __init__(self) -> None:
        self._compiled = None
        self._prompts: tuple[str, ...] | None = None

    def load(self, weights_path: str, **engine_args: object) -> None:
        prompts = engine_args.get("prompts")
        if type(prompts) is not tuple or len(prompts) == 0:
            raise TypeError("load(..., prompts=tuple[str, ...]) is required")
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"OpenVINO IR missing: {path}")
        try:
            import openvino as ov
        except ImportError as exc:
            raise AdapterError("openvino is not installed") from exc
        core = ov.Core()
        devices = list(core.available_devices)
        if not any(str(d).startswith(_DEVICE) for d in devices):
            raise AdapterError(f"OpenVINO {_DEVICE} not available; devices={devices}")
        try:
            model = core.read_model(str(path))
            self._compiled = core.compile_model(model, _DEVICE)
        except Exception as exc:
            raise AdapterError(f"OpenVINO compile on {_DEVICE} failed") from exc
        self._prompts = prompts

    def run(self, rgb: NDArray[np.uint8]) -> tuple[Instance, ...]:
        if self._compiled is None or self._prompts is None:
            raise AdapterError("OpenVinoGpuBackend.load() was not called")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3:
            raise TypeError("rgb must be uint8 HWC")
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        try:
            compiled = self._compiled
            inp = compiled.inputs[0]
            shape = list(inp.shape)
            # NCHW or NHWC; stretch rgb to model spatial size if present
            blob = _rgb_to_input(rgb, shape)
            result = compiled([blob])
            class_indices, scores, masks = _parse_ov_result(result, rgb_hw=(h, w))
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("OpenVINO GPU run failed") from exc
        return instances_from_engine(
            rgb_hw=(h, w),
            prompts=self._prompts,
            class_indices=class_indices,
            scores=scores,
            masks=masks,
        )


def _rgb_to_input(rgb: np.ndarray, shape: list[object]) -> np.ndarray:
    """Stretch to model H/W if the static shape exposes them; NCHW float32 [0,1]."""
    img = rgb.astype(np.float32) / 255.0
    dims = [int(x) if str(x).isdigit() or isinstance(x, (int, np.integer)) else -1 for x in shape]
    if len(dims) == 4 and dims[1] in (1, 3):
        mh, mw = dims[2], dims[3]
        if mh > 0 and mw > 0 and (mh, mw) != (img.shape[0], img.shape[1]):
            img = _stretch_hwc(img, mh, mw)
        return np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    if len(dims) == 4 and dims[-1] in (1, 3):
        mh, mw = dims[1], dims[2]
        if mh > 0 and mw > 0 and (mh, mw) != (img.shape[0], img.shape[1]):
            img = _stretch_hwc(img, mh, mw)
        return img[np.newaxis, ...]
    return np.transpose(img, (2, 0, 1))[np.newaxis, ...]


def _stretch_hwc(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from ugv_perception.backend.instances import _bilinear_float

    chans = [ _bilinear_float(img[:, :, c], h, w).astype(np.float32) for c in range(img.shape[2]) ]
    return np.stack(chans, axis=2)


def _parse_ov_result(result: object, rgb_hw: tuple[int, int]) -> tuple[list, list, list]:
    """Best-effort parse. Unknown layouts raise; do not invent instances."""
    tensors = []
    if hasattr(result, "values"):
        tensors = list(result.values())
    elif isinstance(result, (list, tuple)):
        tensors = list(result)
    else:
        raise AdapterError("unrecognized OpenVINO result type")
    arrays = [np.asarray(t) for t in tensors]
    # Expected: scores [N], classes [N], masks [N,H,W] — otherwise refuse
    masks_n = [a for a in arrays if a.ndim == 3]
    vecs = [a for a in arrays if a.ndim == 1]
    if len(masks_n) == 1 and len(vecs) >= 2:
        masks = [np.asarray(masks_n[0][i]) for i in range(masks_n[0].shape[0])]
        class_indices = [int(x) for x in np.asarray(vecs[0]).tolist()]
        scores = list(np.asarray(vecs[1]).tolist())
        if len(masks) == len(class_indices) == len(scores):
            return class_indices, scores, masks
    if not arrays or (len(arrays) == 1 and arrays[0].size == 0):
        return [], [], []
    raise AdapterError("OpenVINO IR outputs do not match instance (class, score, mask) layout")
