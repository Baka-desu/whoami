from ugv_safety.arbiter.types import Twist2D
from ugv_safety.decel.ramp import RampLimits, TwistRamp

LIMITS = RampLimits(
    max_linear_accel=1.0,
    max_linear_decel=2.0,
    max_angular_accel=1.0,
    max_angular_decel=2.0,
)


def test_ramp_reaches_target_within_time() -> None:
    ramp = TwistRamp(LIMITS)
    out = None
    for _ in range(20):
        out = ramp.step(Twist2D(1.0, 0.0), dt_s=0.1)
    assert out == Twist2D(1.0, 0.0)


def test_decel_faster_than_accel() -> None:
    ramp = TwistRamp(LIMITS)
    ramp.reset(Twist2D(1.0, 0.0))
    out = ramp.step(Twist2D(0.0, 0.0), dt_s=0.1)
    # decel limit 2.0 m/s^2 * 0.1s = 0.2 m/s step
    assert abs(out.linear_x - 0.8) < 1e-9


def test_zero_dt_is_noop() -> None:
    ramp = TwistRamp(LIMITS)
    ramp.reset(Twist2D(0.3, 0.0))
    out = ramp.step(Twist2D(1.0, 0.0), dt_s=0.0)
    assert out == Twist2D(0.3, 0.0)


def test_estop_style_hold_converges_to_zero_not_instant() -> None:
    ramp = TwistRamp(LIMITS)
    ramp.reset(Twist2D(1.0, 0.5))
    first = ramp.step(Twist2D(0.0, 0.0), dt_s=0.05)
    assert first.linear_x > 0.0
    assert first.angular_z > 0.0
    for _ in range(50):
        out = ramp.step(Twist2D(0.0, 0.0), dt_s=0.05)
    assert out == Twist2D(0.0, 0.0)
