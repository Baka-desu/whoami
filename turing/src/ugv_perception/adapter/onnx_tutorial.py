"""Tutorial ONNX eval adapter. Dense logits → RawSemOutput. No OpenVINO import."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import AdapterError, RawSemOutput

ADAPTER_ID = "onnx"
CLASS_NAMES: tuple[str, ...] = ("background", "sidewalk", "grass")


def decode_onnx_logits(
    logits: np.ndarray,
    frame: ImageFrame,
    names: tuple[str, ...] = CLASS_NAMES,
) -> RawSemOutput:
    """Resize logits to camera HW, softmax, argmax / max-prob in [0,1]."""
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3 or a.shape[0] < 1:
        raise AdapterError(f"ONNX logits must be (C, h, w), got {tuple(a.shape)}")
    if np.any(~np.isfinite(a)):
        raise AdapterError("ONNX logits are not finite")
    n_cls = int(a.shape[0])
    if n_cls != len(names):
        raise AdapterError("ONNX class count must match id_to_name")
    rh, rw = int(frame.rgb.shape[0]), int(frame.rgb.shape[1])
    if (a.shape[1], a.shape[2]) != (rh, rw):
        a = np.stack([_resize_map(a[c], rh, rw) for c in range(n_cls)], axis=0)
    shifted = a - a.max(axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    prob = exp / exp.sum(axis=0, keepdims=True)
    labels = prob.argmax(axis=0).astype(np.int32)
    scores = np.take_along_axis(prob, labels[None, ...], axis=0)[0].astype(np.float32)
    if np.any(~np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise AdapterError("ONNX scores are not finite and in [0,1]")
    return RawSemOutput(
        adapter_id=ADAPTER_ID,
        label_ids=labels,
        raw_scores=scores,
        id_to_name={int(i): names[i] for i in range(n_cls)},
        stamp_ns=frame.stamp_ns,
        frame_id=frame.frame_id,
        hw=(rh, rw),
    )


class OnnxTutorialAdapter:
    def __init__(self, logits_fn: object, names: tuple[str, ...] = CLASS_NAMES) -> None:
        self._logits_fn = logits_fn
        self._names = names

    def infer(self, frame: ImageFrame) -> RawSemOutput:
        try:
            logits = self._logits_fn(frame)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("ONNX backend failed") from exc
        return decode_onnx_logits(logits, frame, self._names)


def _resize_map(src: np.ndarray, h: int, w: int) -> np.ndarray:
    mh, mw = src.shape
    src_f = src.astype(np.float64, copy=False)
    if (mh, mw) == (h, w):
        return src_f
    ys = np.clip((np.arange(h) + 0.5) * mh / h - 0.5, 0.0, mh - 1)
    xs = np.clip((np.arange(w) + 0.5) * mw / w - 0.5, 0.0, mw - 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    y0 = np.floor(yy).astype(np.intp)
    x0 = np.floor(xx).astype(np.intp)
    y1 = np.minimum(y0 + 1, mh - 1)
    x1 = np.minimum(x0 + 1, mw - 1)
    wy = yy - y0
    wx = xx - x0
    return (
        src_f[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + src_f[y0, x1] * (1.0 - wy) * wx
        + src_f[y1, x0] * wy * (1.0 - wx)
        + src_f[y1, x1] * wy * wx
    )
