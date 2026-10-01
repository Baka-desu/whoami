"""URDF for the camera mount. Pure Python so it is testable without ROS.

Frames (REP 103 / 105): base_link on the ground plane under the robot centre, x forward, y left, z up;
camera_link at the measured mount, pitched down; camera_optical_frame x right, y down, z forward (the
frame_id the camera driver stamps, which Dev 1, Dev 2 and Dev 3 all resolve through TF).

No default mount: a wrong height or pitch silently moves every projected costmap cell and RTAB-Map's
geometry, so the values must be measured on the robot (tape measure + inclinometer or phone level).
"""

from __future__ import annotations

import math


class MountError(ValueError):
    pass


def _num(name: str, text: str) -> float:
    if text is None or str(text).strip() == "":
        raise MountError(f"{name} is required: measure it on the robot (no default)")
    try:
        v = float(text)
    except ValueError as exc:
        raise MountError(f"{name} must be a number, got {text!r}") from exc
    if not math.isfinite(v):
        raise MountError(f"{name} must be finite")
    return v


def camera_mount(x: str, y: str, z: str, pitch_deg: str) -> tuple[float, float, float, float]:
    """Validated (x, y, z, pitch_rad). z is the lens height above the ground; pitch is positive downward."""
    cx, cy, cz, p = _num("camera_x", x), _num("camera_y", y), _num("camera_z", z), _num("camera_pitch_deg", pitch_deg)
    if cz <= 0.0:
        raise MountError("camera_z must be > 0 (lens height above the ground, metres)")
    if not -10.0 <= p <= 80.0:
        raise MountError("camera_pitch_deg must be in -10..80 (positive = looking down)")
    return cx, cy, cz, math.radians(p)


def robot_urdf(x: float, y: float, z: float, pitch_rad: float,
               camera_frame: str = "camera_optical_frame") -> str:
    return f"""<?xml version="1.0"?>
<robot name="ugv">
  <link name="base_link"/>
  <link name="camera_link"/>
  <link name="{camera_frame}"/>
  <joint name="camera_mount" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="0 {pitch_rad:.6f} 0"/>
  </joint>
  <joint name="camera_optical" type="fixed">
    <parent link="camera_link"/>
    <child link="{camera_frame}"/>
    <origin xyz="0 0 0" rpy="{-math.pi / 2:.6f} 0 {-math.pi / 2:.6f}"/>
  </joint>
</robot>
"""
