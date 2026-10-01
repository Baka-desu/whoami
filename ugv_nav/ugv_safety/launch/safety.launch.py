"""ros2 launch ugv_safety safety.launch.py [config_path:=...] [use_sim_time:=true]"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    default_cfg = str(Path(get_package_share_directory("ugv_safety")) / "config" / "safety" / "safety_timeouts.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_cfg),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("estop_state_path", default_value="~/.ros/ugv/estop_latched"),
            Node(
                package="ugv_safety",
                executable="safety_arbiter",
                name="safety_arbiter",
                output="screen",
                parameters=[
                    {
                        "config_path": ParameterValue(LaunchConfiguration("config_path"), value_type=str),
                        "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
                        "estop_state_path": ParameterValue(LaunchConfiguration("estop_state_path"), value_type=str),
                    }
                ],
            ),
        ]
    )
