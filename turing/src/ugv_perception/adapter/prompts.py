"""Load YOLOE prompts. Names must be a subset of the ontology map keys."""

from __future__ import annotations

from pathlib import Path

import yaml

from ugv_perception.adapter.output import ADAPTER_ID, UNLABELED_NAME


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


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"YAML missing: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=_UniqueKeyLoader)
    if type(data) is not dict:
        raise TypeError("YAML root must be a mapping")
    return data


def load_prompts(path: str | Path, ontology_path: str | Path) -> tuple[str, ...]:
    path = Path(path)
    data = _load_yaml(path)
    adapter_id = data.get("adapter_id")
    if type(adapter_id) is not str or adapter_id != ADAPTER_ID:
        raise ValueError(f"adapter_id must be {ADAPTER_ID!r}")
    raw = data.get("prompts")
    if type(raw) is not list or len(raw) == 0:
        raise ValueError("prompts must be a non-empty list of names")
    prompts: list[str] = []
    seen: set[str] = set()
    for name in raw:
        if type(name) is not str or name == "":
            raise TypeError("prompt names must be non-empty str")
        if name == UNLABELED_NAME:
            raise ValueError("unlabeled must not appear in the prompt list")
        if name in seen:
            raise ValueError(f"duplicate prompt: {name!r}")
        seen.add(name)
        prompts.append(name)

    onto = _load_yaml(Path(ontology_path))
    raw_map = onto.get("map")
    if type(raw_map) is not dict:
        raise TypeError("ontology map must be a mapping")
    allowed = set(raw_map.keys())
    unknown = [n for n in prompts if n not in allowed]
    if unknown:
        raise ValueError(f"prompts not in ontology map: {unknown}")
    return tuple(prompts)
