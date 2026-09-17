"""Instances → dense label_ids / raw_scores. Copies Instance.score; no clip."""

from __future__ import annotations

import math

import numpy as np

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import (
    ADAPTER_ID,
    UNLABELED_ID,
    UNLABELED_NAME,
    Instance,
    RawSemOutput,
)


def _require_frame(frame: ImageFrame) -> tuple[int, int]:
    if type(frame) is not ImageFrame:
        raise TypeError("frame must be an ImageFrame")
    if type(frame.stamp_ns) is not int or frame.stamp_ns <= 0:
        raise TypeError("stamp_ns must be a Python int > 0")
    if type(frame.frame_id) is not str or frame.frame_id == "":
        raise ValueError("frame_id must be a non-empty str")
    rgb = frame.rgb
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise TypeError("rgb must be uint8 ndarray")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("rgb must be non-empty H,W,3")
    return int(rgb.shape[0]), int(rgb.shape[1])


def _require_mask(mask: object, hw: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("Instance.mask must be a numpy ndarray")
    if mask.dtype != np.bool_:
        raise TypeError(f"Instance.mask dtype must be bool, got {mask.dtype}")
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Instance.mask must be a non-empty 2-D bool array")
    if mask.shape != hw:
        raise ValueError(f"Instance.mask shape {mask.shape} must match rgb HW {hw}")
    return mask


def _require_score(score: object) -> float:
    if type(score) is not float:
        raise TypeError(f"Instance.score must be a Python float, got {type(score).__name__}")
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("Instance.score must be finite and in [0, 1] (no clip)")
    return score


def pack(
    frame: ImageFrame,
    instances: tuple[Instance, ...] | list[Instance],
    prompts: tuple[str, ...],
) -> RawSemOutput:
    hw = _require_frame(frame)
    if type(prompts) is not tuple or len(prompts) == 0:
        raise TypeError("prompts must be a non-empty tuple of str")
    n_prompts = len(prompts)
    for name in prompts:
        if type(name) is not str or name == "" or name == UNLABELED_NAME:
            raise ValueError("invalid prompt name")

    label_ids = np.full(hw, UNLABELED_ID, dtype=np.int32)
    raw_scores = np.zeros(hw, dtype=np.float32)

    seq: list[Instance] = []
    for inst in instances:
        if type(inst) is not Instance:
            raise TypeError("instances must be Instance")
        seq.append(inst)
    ordered = sorted(seq, key=lambda inst: inst.score, reverse=True)
    for inst in ordered:
        if type(inst.prompt_id) is not int:
            raise TypeError("prompt_id must be a Python int")
        if inst.prompt_id == UNLABELED_ID or inst.prompt_id < 1 or inst.prompt_id > n_prompts:
            raise ValueError("prompt_id must be in 1..N; id 0 is unlabeled only")
        score = _require_score(inst.score)
        mask = _require_mask(inst.mask, hw)
        update = mask & (score > raw_scores)
        label_ids[update] = inst.prompt_id
        raw_scores[update] = np.float32(score)

    id_to_name: dict[int, str] = {int(UNLABELED_ID): UNLABELED_NAME}
    for i, name in enumerate(prompts, start=1):
        id_to_name[int(i)] = name
    if id_to_name[UNLABELED_ID] != UNLABELED_NAME:
        raise RuntimeError("id 0 must remain unlabeled")

    return RawSemOutput(
        adapter_id=ADAPTER_ID,
        label_ids=label_ids,
        raw_scores=raw_scores,
        id_to_name=id_to_name,
        stamp_ns=frame.stamp_ns,
        frame_id=frame.frame_id,
        hw=hw,
    )
