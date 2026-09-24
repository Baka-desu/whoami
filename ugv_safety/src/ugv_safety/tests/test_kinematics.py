import pytest

from ugv_safety.arbiter.types import Twist2D
from ugv_safety.motor.kinematics import DiffDriveGeometry, twist_to_wheel_speeds

GEOM = DiffDriveGeometry(wheel_radius_m=0.1, track_width_m=0.4)


def test_pure_forward() -> None:
    wheels = twist_to_wheel_speeds(Twist2D(1.0, 0.0), GEOM)
    assert wheels.left_rad_s == pytest.approx(10.0)
    assert wheels.right_rad_s == pytest.approx(10.0)


def test_pure_rotation_in_place() -> None:
    wheels = twist_to_wheel_speeds(Twist2D(0.0, 1.0), GEOM)
    assert wheels.left_rad_s == pytest.approx(-2.0)
    assert wheels.right_rad_s == pytest.approx(2.0)


def test_zero_twist_zero_wheels() -> None:
    wheels = twist_to_wheel_speeds(Twist2D(0.0, 0.0), GEOM)
    assert wheels.left_rad_s == 0.0
    assert wheels.right_rad_s == 0.0


def test_rejects_zero_wheel_radius() -> None:
    with pytest.raises(ValueError):
        twist_to_wheel_speeds(Twist2D(1.0, 0.0), DiffDriveGeometry(wheel_radius_m=0.0, track_width_m=0.4))
