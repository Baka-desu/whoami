"""Dev 2 localization stack: RGB + DA3 depth → RTAB-Map RGB-D, switchable odometry, pose validity.

    ros2 launch ugv_localization localization.launch.py mode:=mapping profile:=sim
    ros2 launch ugv_localization localization.launch.py mode:=localize profile:=sim odom_source:=wheel
    ros2 launch ugv_localization localization.launch.py mode:=mapping fresh_db:=true odom_source:=visual

Nodes:
  cloud_to_depth Dev 1 DA3 cloud + CameraInfo → /rtabmap/depth/image      (depth_input:=cloud, default)
  rgbd_sync      camera RGB + depth image + CameraInfo → /rtabmap/rgbd_image (exact stamps)
  rgbd_odometry  visual odometry on rgbd_image → /rtabmap/odom_visual   (odom_source auto|visual)
  odom_selector  wheel | visual | auto → /odom + TF odom->base_link     (the only publisher)
  rtabmap        RGB-D SLAM → TF map->odom, /map (occupancy from depth), /rtabmap/info
  pose_validity  /ugv/pose_valid heartbeat

TF chain owned here (mindmap D6). Camera driver + /wheel/odom come from Dev 5 bringup; DA3 depth
comes from Dev 1 perception as a point cloud (interfaces.md "Depth input"). This file never starts
either. depth_input:=image skips the conversion and takes a depth image on depth_topic directly
(sim ground-truth depth camera for bring-up / DA3 benchmarking).
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ugv_localization.modes import Mode, parse_mode, plan_mode
from ugv_localization.odom import needs_visual_odometry, parse_odom_source, rtabmap_subscribes_odom_info

# profile -> default use_sim_time (architecture §4)
_PROFILES = {"live_cam": False, "sim": True, "bag": True}
_RGBD = "/rtabmap/rgbd_image"
_ODOM_VISUAL = "/rtabmap/odom_visual"
_ODOM_INFO = "/rtabmap/odom_info"
_DEPTH_FROM_CLOUD = "/rtabmap/depth/image"
_DEPTH_INPUTS = ("cloud", "image")


def _to_bool(text: str, name: str) -> bool:
    low = text.strip().lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    raise RuntimeError(f"{name} must be true/false, got {text!r}")


def _setup(context, *args, **kwargs):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    profile = arg("profile")
    if profile not in _PROFILES:
        raise RuntimeError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")
    ust = arg("use_sim_time")
    use_sim_time = _PROFILES[profile] if ust == "auto" else _to_bool(ust, "use_sim_time")
    policy = parse_odom_source(arg("odom_source"))

    mode = parse_mode(arg("mode"))
    db = os.path.expanduser(arg("database_path"))
    if mode is Mode.MAPPING:
        Path(db).parent.mkdir(parents=True, exist_ok=True)
    plan = plan_mode(mode, db, fresh=_to_bool(arg("fresh_db"), "fresh_db"))  # fails fast

    cfg = os.path.join(get_package_share_directory("ugv_localization"), "config")
    image_topic = arg("image_topic")
    info_topic = arg("camera_info_topic")
    depth_input = arg("depth_input")
    if depth_input not in _DEPTH_INPUTS:
        raise RuntimeError(f"depth_input must be one of {list(_DEPTH_INPUTS)}, got {depth_input!r}")
    depth_topic = _DEPTH_FROM_CLOUD if depth_input == "cloud" else arg("depth_topic")
    depth_src = arg("depth_cloud_topic") if depth_input == "cloud" else depth_topic
    common = {"use_sim_time": use_sim_time}

    actions = [
        LogInfo(
            msg=f"[ugv_localization] profile={profile} mode={mode.value} odom_source={policy.value} "
            f"db={plan.database_path} fresh={bool(plan.rtabmap_args)} use_sim_time={use_sim_time} "
            f"depth_input={depth_input} ({depth_src})"
        ),
    ]
    if depth_input == "cloud":
        actions.append(
            Node(
                package="rtabmap_util",
                executable="pointcloud_to_depthimage",
                name="cloud_to_depth",
                namespace="rtabmap",
                output="screen",
                parameters=[os.path.join(cfg, "cloud_to_depth.yaml"), common],
                remappings=[
                    ("cloud", depth_src),
                    ("camera_info", info_topic),
                    ("image", _DEPTH_FROM_CLOUD),
                    ("image_raw", "/rtabmap/depth/image_raw"),
                ],
            )
        )
    actions += [
        Node(
            package="rtabmap_sync",
            executable="rgbd_sync",
            name="rgbd_sync",
            namespace="rtabmap",
            output="screen",
            parameters=[os.path.join(cfg, "rgbd_sync.yaml"), common],
            remappings=[
                ("rgb/image", image_topic),
                ("depth/image", depth_topic),
                ("rgb/camera_info", info_topic),
                ("rgbd_image", _RGBD),
            ],
        ),
    ]
    if needs_visual_odometry(policy):
        actions.append(
            Node(
                package="rtabmap_odom",
                executable="rgbd_odometry",
                name="rgbd_odometry",
                namespace="rtabmap",
                output="screen",
                parameters=[os.path.join(cfg, "rgbd_odometry.yaml"), common],
                remappings=[("rgbd_image", _RGBD), ("odom", _ODOM_VISUAL), ("odom_info", _ODOM_INFO)],
            )
        )
    actions += [
        Node(
            package="ugv_localization",
            executable="odom_selector",
            name="odom_selector",
            output="screen",
            parameters=[
                {
                    **common,
                    "profile_path": os.path.join(cfg, "odom_select.yaml"),
                    "odom_source": policy.value,
                }
            ],
            remappings=[
                ("wheel/odom", arg("wheel_odom_topic")),
                ("odom_visual", _ODOM_VISUAL),
                ("odom", "/odom"),
            ],
        ),
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            namespace="rtabmap",
            output="screen",
            parameters=[
                os.path.join(cfg, "rtabmap_rgbd.yaml"),
                {
                    **common,
                    "database_path": plan.database_path,
                    "subscribe_odom_info": rtabmap_subscribes_odom_info(policy),
                    **plan.rtabmap_params(),
                },
            ],
            arguments=list(plan.rtabmap_args),
            remappings=[
                ("rgbd_image", _RGBD),
                ("odom", "/odom"),
                ("odom_info", _ODOM_INFO),
                ("map", "/map"),
            ],
        ),
        Node(
            package="ugv_localization",
            executable="pose_validity_node",
            name="pose_validity",
            output="screen",
            parameters=[
                {
                    **common,
                    "profile_path": os.path.join(cfg, "pose_validity.yaml"),
                    "mode": plan.mode.value,
                }
            ],
            remappings=[
                ("odom", "/odom"),
                ("camera_info", info_topic),
                ("depth", depth_topic),
                ("rtabmap/info", "/rtabmap/info"),
            ],
        ),
    ]
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="mapping", description="mapping | localize"),
            DeclareLaunchArgument(
                "odom_source",
                default_value="auto",
                description="auto (wheel when alive, else visual) | wheel | visual",
            ),
            DeclareLaunchArgument("database_path", default_value="~/.ros/ugv/rtabmap.db"),
            DeclareLaunchArgument("fresh_db", default_value="false", description="mapping only: delete db at start"),
            DeclareLaunchArgument("profile", default_value="live_cam", description="live_cam | sim | bag"),
            DeclareLaunchArgument("use_sim_time", default_value="auto", description="auto = from profile"),
            # Topic names pending Dev 5 / Dev 1 confirmation (docs/localization/interfaces.md).
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument(
                "depth_input",
                default_value="cloud",
                description="cloud (Dev 1 DA3 PointCloud2, converted here) | image (32FC1 depth image)",
            ),
            DeclareLaunchArgument("depth_cloud_topic", default_value="/perception/depth_cloud"),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/depth/image_raw",
                description="depth_input:=image only — e.g. the sim ground-truth depth camera",
            ),
            DeclareLaunchArgument("wheel_odom_topic", default_value="/wheel/odom"),
            OpaqueFunction(function=_setup),
        ]
    )
