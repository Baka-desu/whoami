"""Offline evaluation: replay a recorded bag through the Dev 2 stack (profile:=bag).

    ros2 launch ugv_localization bag_eval.launch.py bag:=eval_bags/sim_loop1 mode:=mapping fresh_db:=true

The bag must NOT contain /tf odom->base_link or map->odom (we regenerate them), nor /clock
(`ros2 bag play --clock` publishes it; a recorded /clock would fight it). Record with
scripts/record_eval_bag.sh, which keeps only /tf_static for sensor extrinsics.
Replay DA3 (depth_input:=cloud, default) or the sim ground-truth depth camera
(depth_input:=image depth_topic:=<gt depth topic>) from the same bag.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("ugv_localization")
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "localization.launch.py")),
        launch_arguments={
            "profile": "bag",
            "mode": LaunchConfiguration("mode"),
            "odom_source": LaunchConfiguration("odom_source"),
            "database_path": LaunchConfiguration("database_path"),
            "fresh_db": LaunchConfiguration("fresh_db"),
            "image_topic": LaunchConfiguration("image_topic"),
            "camera_info_topic": LaunchConfiguration("camera_info_topic"),
            "depth_input": LaunchConfiguration("depth_input"),
            "depth_cloud_topic": LaunchConfiguration("depth_cloud_topic"),
            "depth_topic": LaunchConfiguration("depth_topic"),
            "wheel_odom_topic": LaunchConfiguration("wheel_odom_topic"),
        }.items(),
    )
    play = ExecuteProcess(
        cmd=["ros2", "bag", "play", LaunchConfiguration("bag"), "--clock", "--rate", LaunchConfiguration("rate")],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("bag", description="path to rosbag2 directory"),
            DeclareLaunchArgument("rate", default_value="1.0"),
            DeclareLaunchArgument("mode", default_value="mapping"),
            DeclareLaunchArgument("odom_source", default_value="auto", description="auto | wheel | visual"),
            DeclareLaunchArgument("database_path", default_value="~/.ros/ugv/eval_rtabmap.db"),
            DeclareLaunchArgument("fresh_db", default_value="true"),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument("depth_input", default_value="cloud", description="cloud | image"),
            DeclareLaunchArgument("depth_cloud_topic", default_value="/perception/depth_cloud"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            DeclareLaunchArgument("wheel_odom_topic", default_value="/wheel/odom"),
            localization,
            # Give nodes time to subscribe before the first message is played.
            TimerAction(period=3.0, actions=[play]),
        ]
    )
