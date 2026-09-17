"""Immutable name → {0,1,2} table. Values are Python ints; default v1 is 0."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class RemapTable:
    adapter_id: str
    name_to_class: MappingProxyType
    default: int
