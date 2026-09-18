"""Clock + degraded policy. Does not publish. Does not catch T06."""

from __future__ import annotations

from ugv_perception.freshness.profile import (
    FreshnessProfile,
    FreshnessResult,
    PublishDecision,
)

_NS_PER_S = 1_000_000_000


def _require_stamp(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int > 0, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a Python bool, got {type(value).__name__}")
    return value


def evaluate(
    profile: FreshnessProfile, stamp_ns: int, now_ns: int
) -> FreshnessResult:
    if type(profile) is not FreshnessProfile:
        raise TypeError("profile must be a FreshnessProfile")
    stamp_ns = _require_stamp(stamp_ns, name="stamp_ns")
    now_ns = _require_stamp(now_ns, name="now_ns")
    age_s = (now_ns - stamp_ns) / _NS_PER_S
    is_fresh = (age_s >= 0.0) and (age_s <= profile.perception_max_age)
    time_degraded = not is_fresh
    return FreshnessResult(
        age_s=age_s,
        is_fresh=is_fresh,
        time_degraded=time_degraded,
    )


def combine_degraded(
    time_degraded: bool,
    collapse_candidate: bool,
    adapter_error: bool,
) -> bool:
    time_degraded = _require_bool(time_degraded, name="time_degraded")
    collapse_candidate = _require_bool(collapse_candidate, name="collapse_candidate")
    adapter_error = _require_bool(adapter_error, name="adapter_error")
    return bool(time_degraded or collapse_candidate or adapter_error)


def decide_publish(
    result: FreshnessResult,
    collapse_candidate: bool,
    adapter_error: bool,
) -> PublishDecision:
    if type(result) is not FreshnessResult:
        raise TypeError("result must be a FreshnessResult")
    collapse_candidate = _require_bool(collapse_candidate, name="collapse_candidate")
    adapter_error = _require_bool(adapter_error, name="adapter_error")
    degraded = combine_degraded(
        result.time_degraded, collapse_candidate, adapter_error
    )
    valid = bool(result.is_fresh and not collapse_candidate and not adapter_error)
    return PublishDecision(valid=valid, degraded=degraded, publish_mask=valid)
