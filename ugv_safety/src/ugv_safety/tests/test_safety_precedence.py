"""Definition of Done #4 (architecture.md §13): estop / stale perception /
invalid pose each independently drive published /cmd_vel to zero, and the
zero is reached through the decel ramp rather than by construction alone.
"""

from ugv_safety.arbiter.decide import decide
from ugv_safety.arbiter.types import ArbiterInputs, Twist2D
from ugv_safety.decel.ramp import RampLimits, TwistRamp

LIMITS = RampLimits(
    max_linear_accel=2.0, max_linear_decel=2.0, max_angular_accel=2.0, max_angular_decel=2.0
)
MOVING = Twist2D(1.0, 0.3)


def _settle(scenario: dict[str, object], *, ticks: int = 50, dt: float = 0.05) -> Twist2D:
    base = dict(
        estop=False,
        watchdog_tripped=False,
        watchdog_reasons=(),
        perception_degraded=False,
        pose_valid=True,
        candidate=MOVING,
    )
    base.update(scenario)
    ramp = TwistRamp(LIMITS)
    ramp.reset(MOVING)
    out = ramp.current
    for _ in range(ticks):
        decision = decide(ArbiterInputs(**base))
        out = ramp.step(decision.output, dt)
    return out


def test_estop_drives_to_zero() -> None:
    assert _settle({"estop": True}) == Twist2D(0.0, 0.0)


def test_stale_perception_drives_to_zero() -> None:
    assert _settle({"perception_degraded": True}) == Twist2D(0.0, 0.0)


def test_invalid_pose_drives_to_zero() -> None:
    assert _settle({"pose_valid": False}) == Twist2D(0.0, 0.0)


def test_watchdog_trip_drives_to_zero() -> None:
    assert _settle({"watchdog_tripped": True, "watchdog_reasons": ("camera:stale",)}) == Twist2D(0.0, 0.0)


def test_all_clear_holds_candidate() -> None:
    assert _settle({}) == MOVING
