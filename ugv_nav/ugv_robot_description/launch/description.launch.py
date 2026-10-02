"""ros2 launch ugv_robot_description description.launch.py camera_x:=.. camera_y:=.. camera_z:=.. camera_pitch_deg:=..

Publishes base_link -> camera_link -> camera_optical_frame (robot_state_publisher, static). All four mount
values are required and measured on the robot: camera_z = lens height above the ground (m), camera_x / camera_y
= lens offset from the robot centre (m, forward / left), camera_pitch_deg = downward tilt (deg).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ugv_robot_description.urdf import camera_mount, robot_urdf

_ARGS = ("camera_x", "camera_y", "camera_z", "camera_pitch_deg")


def _setup(context, *args, **kwargs):
    vals = [LaunchConfiguration(a).perform(context) for a in _ARGS]
    x, y, z, pitch = camera_mount(*vals)
    frame = LaunchConfiguration("camera_frame").perform(context)
    return [Node(package="robot_state_publisher", executable="robot_state_publisher",
                 name="robot_state_publisher", output="screen",
                 parameters=[{"robot_description": robot_urdf(x, y, z, pitch, frame)}])]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        *[DeclareLaunchArgument(a, default_value="", description="measured camera mount (required)") for a in _ARGS],
        DeclareLaunchArgument("camera_frame", default_value="camera_optical_frame"),
        OpaqueFunction(function=_setup),
    ])
