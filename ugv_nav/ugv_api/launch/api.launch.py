"""Dev 5 operator gateway: REST /api/v1 + SSE (ugv_api).

    ros2 launch ugv_api api.launch.py
    ros2 launch ugv_api api.launch.py host:=0.0.0.0 port:=8080 use_sim_time:=true

Starts only the gateway. It reads the §12 inputs, publishes /ugv/e_stop, calls Nav2's
/navigate_to_pose and RTAB-Map's mode services. It never publishes /cmd_vel*.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params = LaunchConfiguration("params_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file", default_value=PathJoinSubstitution([FindPackageShare("ugv_api"), "config", "api.yaml"])),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="8080"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="ugv_api",
            executable="api_gateway",
            name="ugv_api",
            output="screen",
            parameters=[
                params,
                {
                    "host": LaunchConfiguration("host"),
                    "port": ParameterValue(LaunchConfiguration("port"), value_type=int),
                    "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
                },
            ],
        ),
    ])
