"""Master launch: dev.md Dev 5 task 5 -- runtime profiles live_cam|sim|bag.
Safety stack (arbiter + motor driver) always launches; the profile only
decides where Image + CameraInfo come from.

    ros2 launch ugv_bringup bringup.launch.py profile:=live_cam
    ros2 launch ugv_bringup bringup.launch.py profile:=sim
    ros2 launch ugv_bringup bringup.launch.py profile:=bag bag_path:=/path/to/bag
"""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

_LAUNCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LAUNCH_DIR.parents[1]
_SAFETY_LAUNCH = _REPO_ROOT / "ugv_safety" / "launch" / "safety.launch.py"
_VALID_PROFILES = ("live_cam", "sim", "bag")


def _include(path: Path) -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(path))
    )


def _launch_setup(context, *args, **kwargs):
    profile = LaunchConfiguration("profile").perform(context)
    if profile not in _VALID_PROFILES:
        raise RuntimeError(
            f"profile:={profile!r} is not one of {_VALID_PROFILES} (architecture.md §4)"
        )

    actions = [_include(_SAFETY_LAUNCH)]

    if profile == "live_cam":
        actions.append(_include(_LAUNCH_DIR / "camera.launch.py"))
    elif profile == "sim":
        actions.append(_include(_LAUNCH_DIR / "sim.launch.py"))
    elif profile == "bag":
        bag_path = LaunchConfiguration("bag_path").perform(context)
        if not bag_path:
            raise RuntimeError("profile:=bag requires bag_path:=<path to a recorded outdoor bag>")
        actions.append(ExecuteProcess(
            cmd=["ros2", "bag", "play", bag_path],
            output="screen",
        ))

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="live_cam", description="live_cam|sim|bag"),
        DeclareLaunchArgument("bag_path", default_value="", description="required when profile:=bag"),
        OpaqueFunction(function=_launch_setup),
    ])
