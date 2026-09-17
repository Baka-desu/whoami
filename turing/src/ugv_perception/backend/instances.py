"""Engine arrays → Instance tuples. Only place that may resize/threshold masks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ugv_perception.adapter.output import Instance


def _nearest_bool(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    mh, mw = mask.shape
    if (mh, mw) == (h, w):
        return mask
    y_idx = np.minimum((np.arange(h) * mh) // h, mh - 1)
    x_idx = np.minimum((np.arange(w) * mw) // w, mw - 1)
    return np.ascontiguousarray(mask[np.ix_(y_idx, x_idx)])


def _bilinear_float(src: np.ndarray, h: int, w: int) -> np.ndarray:
    mh, mw = src.shape
    src_f = src.astype(np.float64, copy=False)
    if (mh, mw) == (h, w):
        return src_f
    ys = np.linspace(0.0, mh - 1, h, dtype=np.float64) if h > 1 else np.zeros(h, dtype=np.float64)
    xs = np.linspace(0.0, mw - 1, w, dtype=np.float64) if w > 1 else np.zeros(w, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    y0 = np.floor(yy).astype(np.intp)
    x0 = np.floor(xx).astype(np.intp)
    y1 = np.minimum(y0 + 1, mh - 1)
    x1 = np.minimum(x0 + 1, mw - 1)
    wy = yy - y0
    wx = xx - x0
    Ia = src_f[y0, x0]
    Ib = src_f[y0, x1]
    Ic = src_f[y1, x0]
    Id = src_f[y1, x1]
    return Ia * (1.0 - wy) * (1.0 - wx) + Ib * (1.0 - wy) * wx + Ic * wy * (1.0 - wx) + Id * wy * wx


def _mask_to_rgb_hw(mask: np.ndarray, rgb_hw: tuple[int, int]) -> np.ndarray:
    h, w = rgb_hw
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("engine mask must be a non-empty 2-D array")
    if mask.dtype == np.bool_:
        out = _nearest_bool(mask, h, w)
        return np.asarray(out, dtype=np.bool_)
    if np.issubdtype(mask.dtype, np.floating):
        if not np.isfinite(mask).all() or np.any(mask < 0.0) or np.any(mask > 1.0):
            raise ValueError("float engine mask must be finite and in [0, 1]")
        resized = _bilinear_float(mask, h, w)
        return np.asarray(resized >= 0.5, dtype=np.bool_)
    raise TypeError(f"engine mask dtype must be bool or float, got {mask.dtype}")


def instances_from_engine(
    *,
    rgb_hw: tuple[int, int],
    prompts: tuple[str, ...],
    class_indices: Sequence[object],
    scores: Sequence[object],
    masks: Sequence[np.ndarray],
) -> tuple[Instance, ...]:
    if type(rgb_hw) is not tuple or len(rgb_hw) != 2:
        raise TypeError("rgb_hw must be a (H, W) tuple")
    h, w = rgb_hw
    if type(h) is not int or type(w) is not int or h <= 0 or w <= 0:
        raise ValueError("rgb_hw must be positive Python ints")
    if type(prompts) is not tuple or len(prompts) == 0:
        raise TypeError("prompts must be a non-empty tuple")
    n = len(prompts)
    if not (len(class_indices) == len(scores) == len(masks)):
        raise ValueError("class_indices, scores, and masks must have the same length")
    if len(class_indices) == 0:
        return ()

    out: list[Instance] = []
    for class_index, score, mask in zip(class_indices, scores, masks, strict=True):
        idx = int(class_index)
        if idx < 0 or idx >= n:
            raise ValueError(f"class_index {idx} not in 0..{n - 1}")
        prompt_id = idx + 1
        if isinstance(score, (bool, np.bool_)):
            raise TypeError("engine score must not be bool")
        try:
            val = float(score)
        except (TypeError, ValueError) as exc:
            raise TypeError("engine score must be numeric") from exc
        if not math.isfinite(val) or val < 0.0 or val > 1.0:
            raise ValueError("engine score must be finite and in [0, 1] (no clip)")
        bool_mask = _mask_to_rgb_hw(mask, rgb_hw)
        out.append(Instance(prompt_id=prompt_id, score=val, mask=bool_mask))
    return tuple(out)
