"""Vectorized name→{0,1,2} LUT. Does not copy or write raw_scores."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ugv_perception.port.validate import assert_canonical
from ugv_perception.remap.table import RemapTable


def apply(
    table: RemapTable,
    *,
    adapter_id: str,
    label_ids: NDArray[np.int32],
    id_to_name: dict[int, str],
    raw_scores: NDArray[np.float32],
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    if type(table) is not RemapTable:
        raise TypeError("table must be a RemapTable")
    if type(adapter_id) is not str:
        raise TypeError(f"adapter_id must be a Python str, got {type(adapter_id).__name__}")
    if adapter_id != table.adapter_id:
        raise ValueError(
            f"adapter_id {adapter_id!r} does not match table {table.adapter_id!r}"
        )

    if not isinstance(label_ids, np.ndarray):
        raise TypeError("label_ids must be a numpy ndarray")
    if label_ids.dtype != np.int32:
        raise TypeError(f"label_ids dtype must be int32, got {label_ids.dtype}")
    if label_ids.ndim != 2 or label_ids.size == 0:
        raise ValueError("label_ids must be a non-empty 2-D array")
    if np.any(label_ids < 0):
        raise ValueError("label_ids must be >= 0")

    if type(id_to_name) is not dict:
        raise TypeError("id_to_name must be a dict")
    for key, name in id_to_name.items():
        if type(key) is not int or key < 0:
            raise TypeError(
                f"id_to_name keys must be Python int >= 0, got {type(key).__name__}"
            )
        if type(name) is not str or name == "":
            raise TypeError("id_to_name values must be non-empty Python str")

    if not isinstance(raw_scores, np.ndarray):
        raise TypeError("raw_scores must be a numpy ndarray")
    if raw_scores.dtype != np.float32:
        raise TypeError(f"raw_scores dtype must be float32, got {raw_scores.dtype}")
    if raw_scores.shape != label_ids.shape:
        raise ValueError("raw_scores shape must match label_ids")

    unique_ids = np.unique(label_ids)
    for uid in unique_ids:
        py_id = int(uid)
        if py_id not in id_to_name:
            raise ValueError(
                f"label id {py_id} is missing from id_to_name (adapter unnamed id)"
            )

    # max(id_to_name) is safe only after R9 (>=0) and R11 (every pixel id is named).
    max_id = max(id_to_name)
    lut = np.full(max_id + 1, table.default, dtype=np.uint8)
    for i, name in id_to_name.items():
        lut[i] = table.name_to_class.get(name, table.default)

    classes = lut[label_ids]
    assert_canonical(classes)
    return classes, raw_scores
