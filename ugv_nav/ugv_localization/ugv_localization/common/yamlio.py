"""Strict YAML loading: file must exist, top level must be a mapping, keys must match exactly."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_mapping(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return data


def require_exact_keys(data: dict, expected: set[str], *, where: str) -> None:
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing:
        raise KeyError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)}")


def merge_overlay(base: dict, overlay: dict, *, where: str) -> dict:
    """Base mapping with an overlay's values on top (nested mappings merged key by key; base not modified).

    An overlay may only change keys the base already has, so a typo fails loudly instead of doing nothing.
    """
    out = dict(base)
    for key, value in overlay.items():
        if key not in base:
            raise KeyError(f"{where}: overlay key {key!r} is not in the base profile")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{where}: {key} must be a mapping")
            value = merge_overlay(base[key], value, where=f"{where}:{key}")
        out[key] = value
    return out


def load_yaml_profile(path: str | Path, overlay_path: str | Path | None = None) -> dict:
    """load_yaml_mapping(path), with an optional overlay file (e.g. a *_laptop.yaml timing profile) on top."""
    data = load_yaml_mapping(path)
    if overlay_path:
        data = merge_overlay(data, load_yaml_mapping(overlay_path), where=str(overlay_path))
    return data
