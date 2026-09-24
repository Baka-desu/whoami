"""Timeout monitor: tracks last-seen arrival time per watch, trips on silence.

Deliberately arrival-based, not value-based: a watch trips both when its
publisher goes silent (node crash) *and* when it publishes but goes stale
relative to its own claimed timeout. Dev 1's /ugv/perception_degraded and
Dev 2's /ugv/pose_valid are honesty flags for *their* state; this table is
Dev 5's independent check that those nodes are alive and talking at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from ugv_safety.watchdog.table import WatchdogProfile


@dataclass(frozen=True, slots=True)
class WatchdogResult:
    tripped: bool
    reasons: tuple[str, ...]
    ages_s: dict[str, float]


class WatchdogMonitor:
    def __init__(self, profile: WatchdogProfile) -> None:
        self._profile = profile
        self._last_seen_ns: dict[str, int] = {}

    def touch(self, name: str, now_ns: int) -> None:
        if name not in self._profile.timeouts_s:
            raise KeyError(f"unknown watch {name!r}; not in safety_timeouts.yaml")
        self._last_seen_ns[name] = now_ns

    def evaluate(self, now_ns: int) -> WatchdogResult:
        reasons: list[str] = []
        ages: dict[str, float] = {}
        for name, max_age in self._profile.timeouts_s.items():
            last = self._last_seen_ns.get(name)
            if last is None:
                reasons.append(f"{name}:never_seen")
                ages[name] = float("inf")
                continue
            age_s = (now_ns - last) / 1_000_000_000
            ages[name] = age_s
            if age_s > max_age:
                reasons.append(f"{name}:stale({age_s:.2f}s>{max_age:.2f}s)")
        return WatchdogResult(tripped=bool(reasons), reasons=tuple(reasons), ages_s=ages)
