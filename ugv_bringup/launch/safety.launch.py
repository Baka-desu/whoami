"""Safety authority stack: sole /cmd_vel publisher, always launched regardless
of runtime profile (dev.md Dev 5 tasks 1-3, architecture.md §3.1/§12).

    ros2 launch ugv_bringup safety.launch.py
"""

from __future__ import annotations

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    arbiter = Node(
        package="ugv_safety",
        executable="arbiter_node",
        name="ugv_safety_arbiter",
        output="screen",
    )
    motor_driver = Node(
        package="ugv_safety",
        executable="motor_driver_node",
        name="ugv_motor_driver",
        output="screen",
    )
    return LaunchDescription([arbiter, motor_driver])
