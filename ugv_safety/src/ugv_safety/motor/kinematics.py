"""Differential-drive twist -> wheel angular velocity kinematics.

Authority: dev.md Dev 5 task 3 ("motor driver interface"). This is the pure
math seam; ugv_safety.node.motor_driver_node wraps it in rclpy and is where a
real serial/CAN wheel driver would plug in. The Gazebo sim profile does not
use this node — the diff_drive plugin in ugv_robot_description consumes
/cmd_vel directly, matching the "Base Wheels / Sim" consumer row in dev.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass

from ugv_safety.arbiter.types import Twist2D


@dataclass(frozen=True, slots=True)
class DiffDriveGeometry:
    wheel_radius_m: float
    track_width_m: float


@dataclass(frozen=True, slots=True)
class WheelCommand:
    left_rad_s: float
    right_rad_s: float


def twist_to_wheel_speeds(twist: Twist2D, geometry: DiffDriveGeometry) -> WheelCommand:
    if geometry.wheel_radius_m <= 0:
        raise ValueError("wheel_radius_m must be > 0")
    half_track = geometry.track_width_m / 2.0
    left_v = twist.linear_x - twist.angular_z * half_track
    right_v = twist.linear_x + twist.angular_z * half_track
    return WheelCommand(
        left_rad_s=left_v / geometry.wheel_radius_m,
        right_rad_s=right_v / geometry.wheel_radius_m,
    )
