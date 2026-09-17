"""Load gate YAML. Fail closed. minmax is illegal. τ numbers are not compiled in."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from ugv_perception.confidence.profile import GateProfile

_ALLOWED = frozenset({"identity", "sigmoid"})
_TAU_FIELDS = (
    "tau_min",
    "tau_trav",
    "tau_haz",
    "kappa",
    "min_known_fraction",
)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: yaml.Loader, node: yaml.nodes.MappingNode, deep: bool = False):
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def _require_unit_float(value: object, *, field: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field} must be a Python float in [0, 1], got {type(value).__name__}")
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1], got {value}")
    return value


def _require_finite_float(value: object, *, field: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field} must be a Python float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value}")
    return value


def load_gates(path: str | Path) -> GateProfile:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"gate YAML missing: {path}")

    with path.open(encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=_UniqueKeyLoader)

    if type(data) is not dict:
        raise TypeError("gate YAML root must be a mapping")

    adapter_id = data.get("adapter_id")
    if type(adapter_id) is not str or adapter_id == "":
        raise ValueError("adapter_id must be a non-empty str")
    if path.stem != adapter_id:
        raise ValueError(
            f"repo convention C1: filename stem {path.stem!r} must equal adapter_id {adapter_id!r}"
        )

    normalizer = data.get("normalizer")
    if type(normalizer) is not str:
        raise TypeError("normalizer must be a str")
    if normalizer == "minmax":
        raise ValueError("normalizer minmax is forbidden in v1")
    if normalizer not in _ALLOWED:
        raise ValueError(f"normalizer must be identity or sigmoid, got {normalizer!r}")

    taus = {name: _require_unit_float(data.get(name), field=name) for name in _TAU_FIELDS}

    has_center = "sigmoid_center" in data
    has_scale = "sigmoid_scale" in data
    if normalizer == "identity":
        if has_center or has_scale:
            raise ValueError("identity profile must not set sigmoid_center or sigmoid_scale")
        center = None
        scale = None
    else:
        if not has_center or not has_scale:
            raise ValueError("sigmoid requires sigmoid_center and sigmoid_scale")
        center = _require_finite_float(data["sigmoid_center"], field="sigmoid_center")
        scale = _require_finite_float(data["sigmoid_scale"], field="sigmoid_scale")
        if scale <= 0.0:
            raise ValueError("sigmoid_scale must be > 0")

    return GateProfile(
        adapter_id=adapter_id,
        normalizer=normalizer,
        tau_min=taus["tau_min"],
        tau_trav=taus["tau_trav"],
        tau_haz=taus["tau_haz"],
        kappa=taus["kappa"],
        min_known_fraction=taus["min_known_fraction"],
        sigmoid_center=center,
        sigmoid_scale=scale,
    )
