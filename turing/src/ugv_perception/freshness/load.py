"""Load port freshness YAML. perception_max_age is not compiled into evaluate."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from ugv_perception.freshness.profile import FreshnessProfile


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


def load_freshness(path: str | Path) -> FreshnessProfile:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"freshness YAML missing: {path}")

    with path.open(encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=_UniqueKeyLoader)

    if type(data) is not dict:
        raise TypeError("freshness YAML root must be a mapping")

    value = data.get("perception_max_age")
    if type(value) is not float:
        raise TypeError(
            f"perception_max_age must be a Python float > 0, got {type(value).__name__}"
        )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"perception_max_age must be finite and > 0, got {value}")

    return FreshnessProfile(perception_max_age=value)
