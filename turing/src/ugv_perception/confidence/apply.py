"""Normalize then gate. CPU numpy. Does not set degraded. Does not hardcode τ."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ugv_perception.confidence.profile import GateProfile
from ugv_perception.port.ids import UNKNOWN
from ugv_perception.port.validate import assert_canonical


def _normalize(profile: GateProfile, raw: NDArray[np.float32]) -> NDArray[np.float32]:
    if profile.normalizer == "identity":
        if not np.isfinite(raw).all() or np.any(raw < 0.0) or np.any(raw > 1.0):
            raise ValueError("identity requires every score finite and in [0, 1] (no clip)")
        return raw
    center = profile.sigmoid_center
    scale = profile.sigmoid_scale
    if center is None or scale is None:
        raise ValueError("sigmoid profile missing center/scale")
    conf = 1.0 / (1.0 + np.exp(-(raw.astype(np.float32, copy=False) - center) / scale))
    return np.asarray(conf, dtype=np.float32)


def apply(
    profile: GateProfile,
    *,
    adapter_id: str,
    classes: NDArray[np.uint8],
    raw_scores: NDArray[np.float32],
    runner_up: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], float, bool]:
    if type(profile) is not GateProfile:
        raise TypeError("profile must be a GateProfile")
    if type(adapter_id) is not str:
        raise TypeError(f"adapter_id must be a Python str, got {type(adapter_id).__name__}")
    if adapter_id != profile.adapter_id:
        raise ValueError(
            f"adapter_id {adapter_id!r} does not match profile {profile.adapter_id!r}"
        )

    assert_canonical(classes)
    if not isinstance(raw_scores, np.ndarray):
        raise TypeError("raw_scores must be a numpy ndarray")
    if raw_scores.dtype != np.float32:
        raise TypeError(f"raw_scores dtype must be float32, got {raw_scores.dtype}")
    if raw_scores.shape != classes.shape:
        raise ValueError("raw_scores shape must match classes")

    conf = _normalize(profile, raw_scores)
    classes_out = classes.copy()
    classes_out[conf < profile.tau_min] = UNKNOWN
    classes_out[(classes_out == 1) & (conf < profile.tau_trav)] = UNKNOWN
    classes_out[(classes_out == 2) & (conf < profile.tau_haz)] = UNKNOWN

    if runner_up is not None:
        if not isinstance(runner_up, np.ndarray):
            raise TypeError("runner_up must be a numpy ndarray")
        if runner_up.dtype != np.float32:
            raise TypeError(f"runner_up dtype must be float32, got {runner_up.dtype}")
        if runner_up.shape != classes.shape:
            raise ValueError("runner_up shape must match classes")
        conf2 = _normalize(profile, runner_up)
        classes_out[(classes_out != UNKNOWN) & ((conf - conf2) < profile.kappa)] = UNKNOWN

    known_fraction = float((classes_out != UNKNOWN).mean())
    collapse_candidate = bool(known_fraction < profile.min_known_fraction)
    assert_canonical(classes_out)
    return classes_out, conf, known_fraction, collapse_candidate
