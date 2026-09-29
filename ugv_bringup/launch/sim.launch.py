"""sim profile: outdoor Gazebo world + spawned UGV (dev.md Dev 5 task 5).
Camera and diff-drive wheels come from the plugins in
ugv_robot_description/urdf/ugv.urdf.xacro, not from ugv_safety's
motor_driver_node or camera.launch.py -- see that file's header comment for
why sim is a different "Base Wheels / Sim" consumer of /cmd_vel.

    ros2 launch ugv_bringup sim.launch.py
"""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORLD = _REPO_ROOT / "ugv_bringup" / "worlds" / "outdoor.world"
_XACRO = _REPO_ROOT / "ugv_robot_description" / "urdf" / "ugv.urdf.xacro"


def generate_launch_description() -> LaunchDescription:
    gz_sim = ExecuteProcess(
        cmd=["gz", "sim", "-r", str(_WORLD)],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": _xacro_command()}],
    )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        name="ugv_spawn",
        output="screen",
        arguments=["-topic", "robot_description", "-name", "ugv", "-z", "0.3"],
    )

    return LaunchDescription([gz_sim, robot_state_publisher, spawn_entity])


def _xacro_command() -> str:
    # Evaluated at launch-description-generation time via the xacro CLI so
    # robot_state_publisher gets a plain URDF string, matching the standard
    # ROS 2 pattern (Command(["xacro", path]) would also work but requires
    # launch.substitutions.Command's subprocess semantics; this keeps the
    # dependency explicit and easy to swap for a real calibrated model).
    import subprocess

    result = subprocess.run(
        ["xacro", str(_XACRO)], check=True, capture_output=True, text=True
    )
    return result.stdout
