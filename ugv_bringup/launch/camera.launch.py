"""live_cam profile camera driver (dev.md Dev 5 task 5): the shared vision
sensor. Publishes Image + CameraInfo for Dev 1 and Dev 2's dual fan-out
(architecture.md §5) -- neither of them opens the device.

    ros2 launch ugv_bringup camera.launch.py
    ros2 launch ugv_bringup camera.launch.py camera_info_url:=file:///path/to/calibration.yaml

Uses the standard v4l2_camera ROS 2 package. camera_info_url is intentionally
left empty by default: Dev 2 owns config/cameras/ and real calibration. An
uncalibrated CameraInfo (all-zero K) is honest about that -- do not treat it
as calibrated (architecture.md §8.5 / kill list: mono USB marketed as
outdoor-meter-ready).
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    device_arg = DeclareLaunchArgument("video_device", default_value="/dev/video0")
    frame_id_arg = DeclareLaunchArgument("camera_frame_id", default_value="camera_optical_frame")
    camera_info_url_arg = DeclareLaunchArgument("camera_info_url", default_value="")

    camera_node = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="ugv_camera_driver",
        output="screen",
        parameters=[{
            "video_device": LaunchConfiguration("video_device"),
            "camera_frame_id": LaunchConfiguration("camera_frame_id"),
            "camera_info_url": LaunchConfiguration("camera_info_url"),
        }],
        remappings=[
            ("image_raw", "/camera/image_raw"),
            ("camera_info", "/camera/camera_info"),
        ],
    )

    return LaunchDescription([device_arg, frame_id_arg, camera_info_url_arg, camera_node])
