"""Load config/safety/safety_timeouts.yaml into a WatchdogProfile.

Authority: architecture.md §12 (system health -> safe stop), dev.md Dev 5 task 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class WatchdogProfile:
    timeouts_s: dict[str, float]


def load_watchdog_profile(path: Path) -> WatchdogProfile:
    data = yaml.safe_load(Path(path).read_text()) or {}
    timeouts = data.get("timeouts_s")
    if not timeouts:
        raise ValueError(f"no timeouts_s table in {path}")
    return WatchdogProfile(timeouts_s={str(k): float(v) for k, v in timeouts.items()})
