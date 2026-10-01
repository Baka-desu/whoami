"""RUGD SegFormer-B5 dense logits → RawSemOutput. OpenVINO stays in T12."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import AdapterError, RawSemOutput

ADAPTER_ID = "rugd"
CLASS_NAMES: tuple[str, ...] = (
    "void",
    "dirt",
    "sand",
    "grass",
    "tree",
    "pole",
    "water",
    "sky",
    "vehicle",
    "container/generic-object",
    "asphalt",
    "gravel",
    "building",
    "mulch",
    "rock-bed",
    "log",
    "bicycle",
    "person",
    "fence",
    "bush",
    "sign",
    "rock",
    "bridge",
    "concrete",
    "picnic-table",
)
N_CLASSES = len(CLASS_NAMES)


def load_rugd_config(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adapter YAML missing: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if type(data) is not dict or data.get("adapter_id") != ADAPTER_ID:
        raise ValueError("adapter_id must be rugd")
    backend = data.get("backend")
    if backend not in ("openvino_gpu", "cuda_pytorch"):
        raise ValueError("backend must be openvino_gpu or cuda_pytorch")
    weights = data.get("weights")
    if type(weights) is not str or weights == "":
        raise ValueError("weights must be a local path str")
    hw = data.get("input_hw")
    if (
        type(hw) is not list
        or len(hw) != 2
        or type(hw[0]) is not int
        or type(hw[1]) is not int
        or hw[0] < 1
        or hw[1] < 1
    ):
        raise ValueError("input_hw must be [height, width] ints")
    mean = _triple(data.get("mean"), "mean")
    std = _triple(data.get("std"), "std")
    return {
        "adapter_id": ADAPTER_ID,
        "backend": backend,
        "weights": weights,
        "input_hw": (hw[0], hw[1]),
        "mean": mean,
        "std": std,
    }


def _triple(value: object, field: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise ValueError(f"{field} must be three numbers")
    out: list[float] = []
    for item in value:
        if type(item) not in (int, float) or isinstance(item, bool):
            raise ValueError(f"{field} entries must be numbers")
        out.append(float(item))
    return (out[0], out[1], out[2])


def preprocess_rgb(
    rgb: NDArray[np.uint8],
    *,
    input_hw: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> NDArray[np.float32]:
    """Stretch to input_hw, scale to [0,1], ImageNet norm, NCHW."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError("rgb must be uint8 HWC")
    in_h, in_w = input_hw
    resized = np.moveaxis(
        _resize_map(np.moveaxis(rgb.astype(np.float32), 2, 0), in_h, in_w), 0, 2
    )
    norm = (resized / 255.0 - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return np.transpose(norm, (2, 0, 1))[None].astype(np.float32)


def decode_rugd_logits(logits: np.ndarray, frame: ImageFrame) -> RawSemOutput:
    """(1, 25, h, w) or (25, h, w) logits → argmax softmax on camera HW."""
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3 or a.shape[0] != N_CLASSES:
        raise AdapterError(f"RUGD logits must be (25, h, w), got {tuple(a.shape)}")
    if np.any(~np.isfinite(a)):
        raise AdapterError("RUGD logits are not finite")
    rh, rw = int(frame.rgb.shape[0]), int(frame.rgb.shape[1])
    if (a.shape[1], a.shape[2]) != (rh, rw):
        a = _resize_map(a, rh, rw)
    shifted = a - a.max(axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    prob = exp / exp.sum(axis=0, keepdims=True)
    labels = prob.argmax(axis=0).astype(np.int32)
    scores = np.take_along_axis(prob, labels[None, ...], axis=0)[0].astype(np.float32)
    if np.any(~np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise AdapterError("RUGD scores are not finite and in [0,1]")
    return RawSemOutput(
        adapter_id=ADAPTER_ID,
        label_ids=labels,
        raw_scores=scores,
        id_to_name={i: CLASS_NAMES[i] for i in range(N_CLASSES)},
        stamp_ns=frame.stamp_ns,
        frame_id=frame.frame_id,
        hw=(rh, rw),
    )


class RugdSegformerAdapter:
    def __init__(
        self,
        backend: object,
        *,
        input_hw: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self._backend = backend
        self._input_hw = input_hw
        self._mean = mean
        self._std = std

    def infer(self, frame: ImageFrame) -> RawSemOutput:
        try:
            blob = preprocess_rgb(
                frame.rgb, input_hw=self._input_hw, mean=self._mean, std=self._std
            )
            hw = (int(frame.rgb.shape[0]), int(frame.rgb.shape[1]))
            fast = getattr(self._backend, "run_decoded", None)
            if fast is not None:
                # Backend decodes on its own device (CUDA): same maths as decode_rugd_logits.
                labels, scores = fast(blob, hw)
            else:
                logits = self._backend.run(blob)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("RUGD backend failed") from exc
        if fast is None:
            return decode_rugd_logits(logits, frame)
        if labels.shape != hw or scores.shape != hw:
            raise AdapterError("RUGD backend decoded to the wrong size")
        if np.any(~np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise AdapterError("RUGD scores are not finite and in [0,1]")
        return RawSemOutput(
            adapter_id=ADAPTER_ID,
            label_ids=labels,
            raw_scores=scores,
            id_to_name={i: CLASS_NAMES[i] for i in range(N_CLASSES)},
            stamp_ns=frame.stamp_ns,
            frame_id=frame.frame_id,
            hw=hw,
        )


def _axis_weights(n_src: int, n_dst: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half-pixel bilinear sample positions along one axis: (i0, i1, weight of i1)."""
    pos = np.clip((np.arange(n_dst) + 0.5) * n_src / n_dst - 0.5, 0.0, n_src - 1)
    i0 = np.floor(pos).astype(np.intp)
    return i0, np.minimum(i0 + 1, n_src - 1), pos - i0


def _resize_map(src: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize of the last two axes (half-pixel centres, edge clamp), float64.

    Separable: rows first, then columns, with the sample positions computed once for every leading
    channel. Same result as the 2D four-tap form; that form rebuilt full-image index grids per class
    map and cost ~0.8 s per 640x480 frame for the 25 RUGD logit maps.
    """
    src_f = np.asarray(src).astype(np.float64, copy=False)
    mh, mw = src_f.shape[-2:]
    if (mh, mw) == (h, w):
        return src_f
    y0, y1, wy = _axis_weights(mh, h)
    x0, x1, wx = _axis_weights(mw, w)
    rows = src_f[..., y0, :] * (1.0 - wy)[:, None] + src_f[..., y1, :] * wy[:, None]
    return rows[..., x0] * (1.0 - wx) + rows[..., x1] * wx
