"""Dev 2 localization stack: RGB + DA3 depth → RTAB-Map RGB-D, switchable odometry, pose validity.

    ros2 launch ugv_localization localization.launch.py mode:=mapping profile:=sim
    ros2 launch ugv_localization localization.launch.py mode:=localize profile:=sim odom_source:=auto
    ros2 launch ugv_localization localization.launch.py mode:=mapping fresh_db:=true   # visual odometry only
    ros2 launch ugv_localization localization.launch.py map_assembler:=true    # 3D map for the web viewer

Nodes:
  cloud_to_depth Dev 1 DA3 cloud + CameraInfo → /rtabmap/depth/image      (depth_input:=cloud, fallback only)
  rgbd_sync      camera RGB + depth image + CameraInfo → /rtabmap/rgbd_image (exact stamps)
  rgbd_odometry  visual odometry on rgbd_image → /rtabmap/odom_visual   (odom_source auto|visual)
  odom_selector  wheel | visual | auto → /odom + TF odom->base_link     (the only publisher)
  rtabmap        RGB-D SLAM → TF map->odom, /map (occupancy from depth), /rtabmap/info, mapPath, mapData
  map_assembler  /rtabmap/mapData → /rtabmap/cloud_map (the 3D map, assembled outside the SLAM step).
                 Opt-in, map_assembler:=true (mindmap D24): it keeps every node's data and never gives it back,
                 so it is off by default for long missions; without it nothing publishes /rtabmap/cloud_map
                 (the web viewer's cloud layer stays empty) and SLAM, TF and pose validity are unchanged
  pose_validity  /ugv/pose_valid heartbeat
  distance_tracker /ugv/localization/distance_travelled (odometry estimate) + distance_basis label
  map_stats      /ugv/map/stats JSON (keyframes, loop closures, path length, db size, calibration placeholder)

TF chain owned here (mindmap D6). Camera driver + /wheel/odom come from Dev 5 bringup; DA3 depth
comes from Dev 1 perception as a 32FC1 depth image on /perception/depth/image (depth_input:=image,
default; interfaces.md "Depth input"). depth_input:=cloud instead converts Dev 1's
/perception/depth_cloud. For the sim ground-truth depth camera: depth_topic:=<its topic>.
This file never starts the camera or DA3.
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
_CLOUD_MAP = "/rtabmap/cloud_map"  # the viewer's 3D map, from map_assembler
_SLAM_CLOUD_MAP = "/rtabmap/slam/cloud_map"  # rtabmap's own copy, moved out of the way


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
    # timing:=laptop selects *_laptop.yaml timing profiles (slow GPU laptop: DA3 depth ~1 Hz, ~0.7 s old).
    timing = arg("timing")
    if timing not in ("default", "laptop"):
        raise RuntimeError(f"timing must be default or laptop, got {timing!r}")
    suffix = "" if timing == "default" else "_laptop"
    image_topic = arg("image_topic")
    info_topic = arg("camera_info_topic")
    depth_input = arg("depth_input")
    if depth_input not in _DEPTH_INPUTS:
        raise RuntimeError(f"depth_input must be one of {list(_DEPTH_INPUTS)}, got {depth_input!r}")
    depth_topic = _DEPTH_FROM_CLOUD if depth_input == "cloud" else arg("depth_topic")
    depth_src = arg("depth_cloud_topic") if depth_input == "cloud" else depth_topic
    common = {"use_sim_time": use_sim_time}
    with_assembler = _to_bool(arg("map_assembler"), "map_assembler")

    actions = [
        LogInfo(
            msg=f"[ugv_localization] profile={profile} mode={mode.value} odom_source={policy.value} "
            f"db={plan.database_path} fresh={bool(plan.rtabmap_args)} use_sim_time={use_sim_time} "
            f"depth_input={depth_input} ({depth_src}) map_assembler={with_assembler}"
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
            parameters=[os.path.join(cfg, f"rgbd_sync{suffix}.yaml"), common],
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
                    "profile_path": os.path.join(cfg, f"odom_select{suffix}.yaml"),
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
                # Its own cloud_map is assembled inside the SLAM step: never subscribe this one (map_assembler serves the real one)
                ("cloud_map", _SLAM_CLOUD_MAP),
            ],
        ),
    ]
    if with_assembler:
        # The 3D map for the viewer, assembled in its own process from /rtabmap/mapData (Task 8 review I1). rtabmap builds
        # cloud_map inside its SLAM callback, and the first step after a late subscriber (the gateway, on demand) attaches
        # assembles the whole map: 0.17-0.35 s at 275-415 nodes, growing with the map (docs/mapping/baseline.md "Task 8 fix
        # round 1"). Same Grid/* and map params as rtabmap (the YAML is /**); its other map topics stay under
        # /rtabmap/assembler/. map_cleanup false: its cache survives the viewer closing, so a re-attach adds only the new
        # nodes (0.2-0.4 s instead of 1.7-3.0 s) and its depth-1 mapData subscription drops nothing meanwhile. true would not
        # bound memory: the peak while a viewer is open is the same (~2 MB/node at 640x480) and grows with the map either
        # way (baseline.md "Final review I2"); for a long mission use map_assembler:=false (docs/mapping/README.md).
        actions.append(
            Node(
                package="rtabmap_util",
                executable="map_assembler",
                name="map_assembler",
                namespace="rtabmap/assembler",
                output="screen",
                parameters=[os.path.join(cfg, "rtabmap_rgbd.yaml"), {**common, "rtabmap": "/rtabmap/rtabmap", "map_cleanup": False}],
                remappings=[("mapData", "/rtabmap/mapData"), ("cloud_map", _CLOUD_MAP)],
            )
        )
    actions += [
        Node(
            package="ugv_localization",
            executable="pose_validity_node",
            name="pose_validity",
            output="screen",
            parameters=[
                {
                    **common,
                    "profile_path": os.path.join(cfg, f"pose_validity{suffix}.yaml"),
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
        Node(
            package="ugv_localization",
            executable="distance_tracker",
            name="distance_tracker",
            output="screen",
            parameters=[{**common, "profile_path": os.path.join(cfg, "distance.yaml")}],
            remappings=[("odom", "/odom")],
        ),
        Node(
            package="ugv_localization",
            executable="map_stats_node",
            name="map_stats",
            output="screen",
            parameters=[
                {
                    **common,
                    "mode": plan.mode.value,
                    "database_path": plan.database_path,
                    "calibration_file": arg("calibration_file"),
                }
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
                default_value="visual",
                description="visual (camera + DA3 depth only, default: no wheel sensor yet) | auto (wheel when alive, else visual) | wheel",
            ),
            DeclareLaunchArgument("database_path", default_value="~/.ros/ugv/rtabmap.db"),
            DeclareLaunchArgument("timing", default_value="default"),
            DeclareLaunchArgument(
                "calibration_file",
                default_value="",
                description="camera calibration YAML in use; only its `placeholder` flag is read, for /ugv/map/stats ('' = none)",
            ),
            DeclareLaunchArgument("fresh_db", default_value="false", description="mapping only: delete db at start"),
            DeclareLaunchArgument(
                "map_assembler",
                default_value="false",
                description="true: map_assembler serves /rtabmap/cloud_map to the web viewer (memory grows with the map) | "
                "false (default): no /rtabmap/cloud_map at all; SLAM, TF and pose validity are unchanged",
            ),
            DeclareLaunchArgument("profile", default_value="live_cam", description="live_cam | sim | bag"),
            DeclareLaunchArgument("use_sim_time", default_value="auto", description="auto = from profile"),
            # Topic names pending Dev 5 / Dev 1 confirmation (docs/localization/interfaces.md).
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument(
                "depth_input",
                default_value="image",
                description="image (32FC1 depth image, default) | cloud (Dev 1 DA3 PointCloud2, converted here)",
            ),
            DeclareLaunchArgument("depth_cloud_topic", default_value="/perception/depth_cloud"),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/perception/depth/image",
                description="depth_input:=image: Dev 1 DA3 depth (default) or the sim ground-truth depth camera",
            ),
            DeclareLaunchArgument("wheel_odom_topic", default_value="/wheel/odom"),
            OpaqueFunction(function=_setup),
        ]
    )
