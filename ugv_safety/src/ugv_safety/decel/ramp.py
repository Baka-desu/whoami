"""Rate-limited twist ramp (dev.md Dev 5 task 3).

The arbiter decides a *target* twist per tick (zero on hold, Nav2's candidate
otherwise). This module smooths the *published* twist toward that target so a
safety stop is a controlled deceleration, not a discontinuous jump that could
skid or tip an outdoor UGV. Accel and decel limits are independent because a
safety stop should be allowed to brake harder than it accelerates.
"""

from __future__ import annotations

from dataclasses import dataclass

from ugv_safety.arbiter.types import Twist2D


@dataclass(frozen=True, slots=True)
class RampLimits:
    max_linear_accel: float
    max_linear_decel: float
    max_angular_accel: float
    max_angular_decel: float


def _slew(current: float, target: float, dt_s: float, max_accel: float, max_decel: float) -> float:
    decelerating = abs(target) < abs(current)
    limit = max_decel if decelerating else max_accel
    max_step = abs(limit) * dt_s
    delta = target - current
    if delta > max_step:
        return current + max_step
    if delta < -max_step:
        return current - max_step
    return target


class TwistRamp:
    """Stateful: call step() once per control tick with the elapsed dt."""

    def __init__(self, limits: RampLimits) -> None:
        self._limits = limits
        self._current = Twist2D(0.0, 0.0)

    @property
    def current(self) -> Twist2D:
        return self._current

    def reset(self, twist: Twist2D = Twist2D(0.0, 0.0)) -> None:
        self._current = twist

    def step(self, target: Twist2D, dt_s: float) -> Twist2D:
        if dt_s <= 0:
            return self._current
        lin = _slew(
            self._current.linear_x, target.linear_x, dt_s,
            self._limits.max_linear_accel, self._limits.max_linear_decel,
        )
        ang = _slew(
            self._current.angular_z, target.angular_z, dt_s,
            self._limits.max_angular_accel, self._limits.max_angular_decel,
        )
        self._current = Twist2D(lin, ang)
        return self._current
