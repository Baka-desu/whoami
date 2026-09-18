"""Freshness config. Max age lives in YAML, not in evaluate."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FreshnessProfile:
    perception_max_age: float


@dataclass(frozen=True, slots=True)
class FreshnessResult:
    age_s: float
    is_fresh: bool
    time_degraded: bool


@dataclass(frozen=True, slots=True)
class PublishDecision:
    valid: bool
    degraded: bool
    publish_mask: bool
