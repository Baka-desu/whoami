"""Load remap YAML. Fail closed. Filename stem == adapter_id is repo convention C1."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import yaml

from ugv_perception.port.ids import CANONICAL, UNKNOWN
from ugv_perception.remap.table import RemapTable


class _UniqueKeyLoader(yaml.SafeLoader):
    """R7: duplicate keys are an error, not last-wins."""


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


def _require_canonical_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a Python int in {{0,1,2}}, got {type(value).__name__}")
    if value not in CANONICAL:
        raise ValueError(f"{field} must be in {{0,1,2}}, got {value}")
    return value


def load_remap(path: str | Path) -> RemapTable:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"remap YAML missing: {path}")

    with path.open(encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=_UniqueKeyLoader)

    if type(data) is not dict:
        raise TypeError("remap YAML root must be a mapping")

    adapter_id = data.get("adapter_id")
    if type(adapter_id) is not str or adapter_id == "":
        raise ValueError("adapter_id must be a non-empty str")

    if path.stem != adapter_id:
        raise ValueError(
            f"repo convention C1: filename stem {path.stem!r} must equal adapter_id {adapter_id!r}"
        )

    raw_map = data.get("map")
    if raw_map is None:
        raw_map = {}
    if type(raw_map) is not dict:
        raise TypeError("map must be a mapping of names to {0,1,2}")

    name_to_class: dict[str, int] = {}
    for name, klass in raw_map.items():
        if type(name) is not str or name == "":
            raise TypeError("map keys must be non-empty str")
        if name == "cautious":
            raise ValueError("canonical class 'cautious' is not in v1")
        name_to_class[name] = _require_canonical_int(klass, field=f"map[{name!r}]")

    if "default" not in data:
        raise ValueError("default is required")
    default = _require_canonical_int(data["default"], field="default")
    if default != UNKNOWN:
        raise ValueError("v1 default must be 0 (unknown)")

    return RemapTable(
        adapter_id=adapter_id,
        name_to_class=MappingProxyType(name_to_class),
        default=default,
    )
