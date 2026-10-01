"""Safety arbiter decision kernel (no ROS). Sole authority over the base command.

architecture.md §3.1, highest wins:
  L1  e-stop asserted                                   -> zero twist, immediately
  L2  system-health fail (camera / perception / localization / TF / Nav2 silent or down)
                                                        -> zero twist
  L3  perception degraded, or invalid pose              -> zero twist (hold)
  L4  Nav2 candidate (/cmd_vel_nav2), fresh             -> forwarded, clamped

Fail closed: a signal that was never received is as bad as a stale one, so a fresh arbiter holds
at L2 until every watched source has spoken. Time is always passed in, so decisions are
deterministic and unit-testable.

The status text (published on /ugv/safety_status, whose type dev.md leaves undefined) is
`L<level> <NAME>[: reason,...]`, e.g. `L4 FORWARD`, `L3 HOLD: pose_invalid`, `L2 HEALTH: nav2_stale`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import yaml

NS_PER_S = 1_000_000_000
_MAX_DT_S = 0.5  # a long scheduling gap must not turn into one huge ramp step


class Level(IntEnum):
    ESTOP = 1
    HEALTH = 2
    DEGRADED = 3
    NAV2 = 4


_NAME = {Level.ESTOP: "ESTOP", Level.HEALTH: "HEALTH", Level.DEGRADED: "HOLD"}


class ConfigError(ValueError):
    """The safety config is missing a value or has a nonsensical one."""


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    publish_rate_hz: float
    perception_s: float
    localization_s: float
    nav2_s: float
    camera_s: float
    candidate_s: float
    max_linear: float
    max_angular: float
    ramp_on_hold: bool
    decel_linear: float
    decel_angular: float

    def __post_init__(self) -> None:
        for name in (
            "publish_rate_hz", "perception_s", "localization_s", "nav2_s", "camera_s", "candidate_s",
            "max_linear", "max_angular", "decel_linear", "decel_angular",
        ):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0.0:
                raise ConfigError(f"{name} must be a finite number > 0, got {v!r}")
        if not isinstance(self.ramp_on_hold, bool):
            raise ConfigError(f"ramp_on_hold must be true/false, got {self.ramp_on_hold!r}")


_TIMEOUT_KEYS = {"perception_s", "localization_s", "nav2_s", "camera_s", "candidate_s"}
_LIMIT_KEYS = {"max_linear", "max_angular", "ramp_on_hold", "decel_linear", "decel_angular"}


def load_config(path: str | Path) -> SafetyConfig:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read safety config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    timeouts, limits = data.get("timeouts"), data.get("limits")
    if not isinstance(timeouts, dict) or not isinstance(limits, dict):
        raise ConfigError(f"{path}: needs `timeouts` and `limits` mappings")
    for block, keys, label in ((timeouts, _TIMEOUT_KEYS, "timeouts"), (limits, _LIMIT_KEYS, "limits")):
        missing, extra = keys - set(block), set(block) - keys
        if missing or extra:
            raise ConfigError(f"{path}: {label} missing {sorted(missing)} unexpected {sorted(extra)}")
    if "publish_rate_hz" not in data:
        raise ConfigError(f"{path}: missing publish_rate_hz")
    return SafetyConfig(publish_rate_hz=data["publish_rate_hz"], **timeouts, **limits)


@dataclass(frozen=True, slots=True)
class Decision:
    level: Level
    reasons: tuple[str, ...]
    linear: float
    angular: float

    @property
    def status(self) -> str:
        if self.level is Level.NAV2:
            if self.reasons:
                return f"L4 IDLE: {','.join(self.reasons)}"
            return "L4 FORWARD"
        return f"L{int(self.level)} {_NAME[self.level]}: {','.join(self.reasons)}"


class SafetyArbiter:
    def __init__(self, cfg: SafetyConfig) -> None:
        self._cfg = cfg
        self._estop = False  # latched until an explicit release
        self._pose: tuple[bool, int] | None = None
        self._perception: tuple[bool, int] | None = None
        self._nav2: tuple[bool, int] | None = None
        self._camera: tuple[int, int] | None = None  # (header stamp ns, arrival ns)
        self._candidate: tuple[float, float, int] | None = None
        self._out_lin = 0.0
        self._out_ang = 0.0
        self._last_step_ns: int | None = None

    # ---- inputs ------------------------------------------------------------------------------
    @property
    def estop_asserted(self) -> bool:
        return self._estop

    def on_estop(self, asserted: bool, now_ns: int) -> None:
        # No message is not an assertion (nobody could ever drive), but an asserted e-stop stays
        # asserted if the operator's link drops: only an explicit False releases it.
        self._estop = bool(asserted)

    def on_pose_valid(self, valid: bool, now_ns: int) -> None:
        self._pose = (bool(valid), now_ns)

    def on_perception_degraded(self, degraded: bool, now_ns: int) -> None:
        self._perception = (bool(degraded), now_ns)

    def on_nav2_heartbeat(self, alive: bool, now_ns: int) -> None:
        self._nav2 = (bool(alive), now_ns)

    def on_camera(self, stamp_ns: int, now_ns: int) -> None:
        self._camera = (stamp_ns, now_ns)

    def on_candidate(self, linear: float, angular: float, now_ns: int) -> None:
        self._candidate = (linear, angular, now_ns)

    # ---- decision ----------------------------------------------------------------------------
    def _stale(self, arrived_ns: int | None, limit_s: float, now_ns: int) -> bool:
        return arrived_ns is None or (now_ns - arrived_ns) > limit_s * NS_PER_S

    def _health_reasons(self, now_ns: int) -> list[str]:
        c = self._cfg
        out: list[str] = []
        cam = self._camera
        if cam is None or self._stale(cam[1], c.camera_s, now_ns) or (now_ns - cam[0]) > c.camera_s * NS_PER_S:
            out.append("camera_stale")
        if self._stale(self._perception[1] if self._perception else None, c.perception_s, now_ns):
            out.append("perception_stale")
        if self._stale(self._pose[1] if self._pose else None, c.localization_s, now_ns):
            out.append("localization_stale")
        if self._stale(self._nav2[1] if self._nav2 else None, c.nav2_s, now_ns):
            out.append("nav2_stale")
        elif self._nav2 is not None and not self._nav2[0]:
            out.append("nav2_down")
        return out

    def _ramp(self, target: float, current: float, decel: float, dt: float) -> float:
        """Move `current` toward `target` (zero) by at most decel*dt."""
        step = decel * dt
        if current > target:
            return max(target, current - step)
        return min(target, current + step)

    def step(self, now_ns: int) -> Decision:
        c = self._cfg
        dt = 0.0 if self._last_step_ns is None else min(max((now_ns - self._last_step_ns) / NS_PER_S, 0.0), _MAX_DT_S)
        self._last_step_ns = now_ns

        if self._estop:
            return self._emit(Level.ESTOP, ("e_stop",), 0.0, 0.0)

        health = self._health_reasons(now_ns)
        if health:
            return self._hold(Level.HEALTH, tuple(health), c, dt)

        degraded: list[str] = []
        if self._pose is not None and not self._pose[0]:
            degraded.append("pose_invalid")
        if self._perception is not None and self._perception[0]:
            degraded.append("perception_degraded")
        if degraded:
            return self._hold(Level.DEGRADED, tuple(degraded), c, dt)

        cand = self._candidate
        if cand is None:
            return self._emit(Level.NAV2, ("no_candidate",), 0.0, 0.0)
        lin, ang, at = cand
        if not (math.isfinite(lin) and math.isfinite(ang)):
            return self._emit(Level.NAV2, ("candidate_invalid",), 0.0, 0.0)
        if self._stale(at, c.candidate_s, now_ns):
            return self._emit(Level.NAV2, ("candidate_stale",), 0.0, 0.0)
        lin = max(-c.max_linear, min(c.max_linear, lin))
        ang = max(-c.max_angular, min(c.max_angular, ang))
        return self._emit(Level.NAV2, (), lin, ang)

    def _hold(self, level: Level, reasons: tuple[str, ...], c: SafetyConfig, dt: float) -> Decision:
        if not c.ramp_on_hold:
            return self._emit(level, reasons, 0.0, 0.0)
        lin = self._ramp(0.0, self._out_lin, c.decel_linear, dt)
        ang = self._ramp(0.0, self._out_ang, c.decel_angular, dt)
        return self._emit(level, reasons, lin, ang)

    def _emit(self, level: Level, reasons: tuple[str, ...], lin: float, ang: float) -> Decision:
        self._out_lin, self._out_ang = lin, ang
        return Decision(level, reasons, lin, ang)
