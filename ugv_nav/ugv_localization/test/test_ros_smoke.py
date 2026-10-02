"""ROS-side smoke tests. Skipped when ROS 2 (rclpy / launch_ros / rtabmap_msgs) is not installed."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "ugv_localization.nodes.odom_selector",
        "ugv_localization.nodes.pose_validity_node",
        "ugv_localization.nodes.distance_tracker",
        "ugv_localization.nodes.tf_rate_check",
        "ugv_localization.nodes.drift_eval",
        "ugv_localization.nodes.depth_eval",
        "ugv_localization.nodes.mode_cli",
        "ugv_localization.nodes.camera_info_to_yaml",
    ],
)
def test_s1_nodes_import(module: str) -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("rtabmap_msgs")
    mod = importlib.import_module(module)
    assert callable(mod.main)


def test_s3_no_removed_logger_warn_alias() -> None:
    # Lyrical rclpy removed RcutilsLogger.warn (use .warning); it crashed pose_validity_node at runtime.
    offenders = [
        p.name
        for p in (_PKG / "ugv_localization" / "nodes").glob("*.py")
        if ".warn(" in p.read_text(encoding="utf-8") or "get_logger().warn\n" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.mark.parametrize("name", ["localization.launch.py", "bag_eval.launch.py"])
def test_s2_launch_descriptions_build(name: str) -> None:
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), _PKG / "launch" / name)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    ld = mod.generate_launch_description()
    assert ld.entities


@pytest.mark.parametrize("depth_input", ["cloud", "image"])
@pytest.mark.parametrize(
    "odom_source, expect_vo",
    [("auto", True), ("wheel", False), ("visual", True)],
)
def test_s4_launch_nodes_per_odom_source(odom_source: str, expect_vo: bool, depth_input: str, tmp_path: Path) -> None:
    pytest.importorskip("launch_ros")
    ament = pytest.importorskip("ament_index_python.packages")
    try:
        ament.get_package_share_directory("ugv_localization")
    except Exception:  # noqa: BLE001
        pytest.skip("ugv_localization not installed (run under colcon test)")
    from launch import LaunchContext
    from launch_ros.actions import Node

    spec = importlib.util.spec_from_file_location("loc_launch", _PKG / "launch" / "localization.launch.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    ctx = LaunchContext()
    ctx.launch_configurations.update(
        {
            "mode": "mapping",
            "odom_source": odom_source,
            "database_path": str(tmp_path / "db" / "rtabmap.db"),
            "fresh_db": "false",
            "profile": "sim",
            "use_sim_time": "auto",
            "image_topic": "/camera/image_raw",
            "camera_info_topic": "/camera/camera_info",
            "depth_input": depth_input,
            "depth_cloud_topic": "/perception/depth_cloud",
            "depth_topic": "/camera/depth/image_raw",
            "wheel_odom_topic": "/wheel/odom",
            "timing": "default",
            "map_assembler": "false",
            "calibration_file": "",
        }
    )
    execs = [a.node_executable for a in mod._setup(ctx) if isinstance(a, Node)]
    expected = {"rgbd_sync", "odom_selector", "rtabmap", "pose_validity_node", "distance_tracker", "map_stats_node"}
    expected |= {"rgbd_odometry"} if expect_vo else set()
    expected |= {"pointcloud_to_depthimage"} if depth_input == "cloud" else set()
    assert set(execs) == expected
    assert len(execs) == len(set(execs))  # one of each; the TF-owning selector is never duplicated


def test_s5_no_logger_severity_switch_on_one_call_site() -> None:
    # rclpy raises "Logger severity cannot be changed between calls" when one call site alternates
    # info/warning (it crashed pose_validity_node the first time the pose became valid).
    import re

    pattern = re.compile(r"=\s*self\.get_logger\(\)\.\w+\s+if\s+.*else\s+self\.get_logger\(\)")
    offenders = [
        p.name for p in (_PKG / "ugv_localization" / "nodes").glob("*.py") if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_s6_rtabmap_3d_map_outputs_pinned() -> None:
    # Task 8: the 3D map outputs are on (Grid/3D) and the node parameters behind them are explicit, so a rtabmap upgrade
    # cannot change what the viewer and the elevation mapper receive. The real stack is test_ros_stack.py::test_x4.
    import yaml

    params = yaml.safe_load((_PKG / "config" / "rtabmap_rgbd.yaml").read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    assert params["Grid/3D"] == "true"  # library params are strings
    assert params["cloud_output_voxelized"] is True  # node params are plain values
    assert params["cloud_subtract_filtering"] is False
    assert params["map_always_update"] is False
    assert params["latch"] is True
