"""ros2 launch ugv_bringup camera.launch.py calibration_file:=/path/to/cam.yaml [device:=/dev/video0]
ros2 launch ugv_bringup camera.launch.py calibration_mode:=true [width:=640 height:=480]   # to calibrate

calibration_file has no default on purpose: a camera YAML must come from a real calibration, never be
invented. Calibration mode publishes raw images only (no CameraInfo) so `camera_calibration` can produce it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name -> (default, python type). Types are explicit so `device:=0` stays a string.
_ARGS: dict[str, tuple[str, type]] = {
    "device": ("/dev/video0", str),
    "frame_id": ("camera_optical_frame", str),
    "fps": ("30.0", float),
    "image_topic": ("/camera/image_raw", str),
    "info_topic": ("/camera/camera_info", str),
    "compressed_topic": ("/image_raw/compressed", str),
    "ui_info_topic": ("/camera_info", str),
    "compressed_rate_hz": ("5.0", float),
    "jpeg_quality": ("80", int),
    "calibration_mode": ("false", bool),
    "width": ("640", int),
    "height": ("480", int),
}


def generate_launch_description() -> LaunchDescription:
    params = {"calibration_file": ParameterValue(LaunchConfiguration("calibration_file"), value_type=str)}
    params.update({k: ParameterValue(LaunchConfiguration(k), value_type=t) for k, (_, t) in _ARGS.items()})
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "calibration_file", default_value="",
                description="camera_info_manager YAML from a real calibration (not needed with calibration_mode:=true)",
            ),
            *[DeclareLaunchArgument(k, default_value=d) for k, (d, _) in _ARGS.items()],
            Node(
                package="ugv_bringup",
                executable="camera_driver",
                name="camera_driver",
                output="screen",
                parameters=[params],
            ),
        ]
    )
