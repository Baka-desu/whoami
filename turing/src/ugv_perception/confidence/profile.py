"""Per-adapter gate profile. τ values live in YAML, not in this module."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateProfile:
    adapter_id: str
    normalizer: str
    tau_min: float
    tau_trav: float
    tau_haz: float
    kappa: float
    min_known_fraction: float
    sigmoid_center: float | None
    sigmoid_scale: float | None
