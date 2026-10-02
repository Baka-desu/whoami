"""ros2 launch ugv_costmap semantic_costmap.launch.py [params_file:=...]"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default = str(Path(get_package_share_directory("ugv_costmap")) / "config" / "semantic_costmap.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default),
        Node(package="ugv_costmap", executable="semantic_costmap_node", name="semantic_costmap",
             output="screen", parameters=[LaunchConfiguration("params_file")]),
    ])
