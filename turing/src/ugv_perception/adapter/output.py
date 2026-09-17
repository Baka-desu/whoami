"""Adapter language. Must not leak past T03 remap."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

UNLABELED_ID = 0
UNLABELED_NAME = "unlabeled"
ADAPTER_ID = "yoloe"


@dataclass(frozen=True, slots=True)
class Instance:
    prompt_id: int
    score: float
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RawSemOutput:
    adapter_id: str
    label_ids: NDArray[np.int32]
    raw_scores: NDArray[np.float32]
    id_to_name: dict[int, str]
    stamp_ns: int
    frame_id: str
    hw: tuple[int, int]


class AdapterError(Exception):
    """T06 failed. T07 sets adapter_error; T06 does not publish a fake mask."""
