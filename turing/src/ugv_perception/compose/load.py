"""Refuse to start without remap / gates / freshness YAML."""

from __future__ import annotations

from pathlib import Path

from ugv_perception.confidence.load import load_gates
from ugv_perception.confidence.profile import GateProfile
from ugv_perception.freshness.load import load_freshness
from ugv_perception.freshness.profile import FreshnessProfile
from ugv_perception.remap.load import load_remap
from ugv_perception.remap.table import RemapTable


def load_compose_configs(
    *,
    remap_path: str | Path,
    gates_path: str | Path,
    freshness_path: str | Path,
) -> tuple[RemapTable, GateProfile, FreshnessProfile]:
    remap_path = Path(remap_path)
    if not remap_path.is_file():
        raise FileNotFoundError(f"remap YAML missing: {remap_path}")
    return (
        load_remap(remap_path),
        load_gates(gates_path),
        load_freshness(freshness_path),
    )
