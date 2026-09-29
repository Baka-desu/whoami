"""End-to-end endpoint test: the real launch file + real RTAB-Map nodes, fed synthetic sensors.

Proves the wiring between every endpoint in localization.launch.py (topic names, remaps, QoS,
exact-stamp sync, TF ownership) — not RTAB-Map accuracy. Inputs mimic the team contracts:
  Dev 5  /camera/image_raw rgb8 + /camera/camera_info per frame (BEST-EFFORT, like many drivers),
         /wheel/odom (no TF), /tf_static base_link -> camera_optical
  Dev 1  /perception/depth_cloud PointCloud2 (x/y/z float32 m, unorganized, sky dropped, back-projected
         with the raw K; source image stamp + optical frame) — exactly turing node/cloud.py's format.
         depth_input:=image variant: a 32FC1 depth image (sim ground-truth depth camera path).
Skipped unless ROS 2 + rtabmap_ros + an installed ugv_localization are available (colcon test).
The scene is a static random texture at 3 m: synthetic test input, not a product calibration.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("rtabmap_msgs")
ament = pytest.importorskip("ament_index_python.packages")
try:
    ament.get_package_share_directory("ugv_localization")
    ament.get_package_share_directory("rtabmap_slam")
except Exception:  # noqa: BLE001
    pytest.skip("ugv_localization / rtabmap_slam not installed (run under colcon test)", allow_module_level=True)

import numpy as np  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from rtabmap_msgs.msg import Info, RGBDImage  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

_W, _H, _FRAME = 320, 240, "camera_optical"
_K = [260.0, 0.0, 160.0, 0.0, 260.0, 120.0, 0.0, 0.0, 1.0]
_COV = [0.001 if i in (0, 7, 14, 21, 28, 35) else 0.0 for i in range(36)]
_LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
_BEST_EFFORT = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
_RNG = np.random.default_rng(7)
_TEXTURE = (_RNG.integers(0, 255, size=(_H // 4, _W // 4, 3), dtype=np.uint8)).repeat(4, 0).repeat(4, 1)
_DEPTH = np.full((_H, _W), 3.0, dtype="<f4")
_DEPTH[: _H // 6, :] = np.nan  # "sky" band


def _backproject(depth: np.ndarray) -> np.ndarray:
    """Same math as Dev 1 depth/geometry.backproject: optical frame, holes dropped."""
    v, u = np.nonzero(np.isfinite(depth))
    z = depth[v, u].astype(np.float64)
    x = (u - _K[2]) * z / _K[0]
    y = (v - _K[5]) * z / _K[4]
    return np.stack([x, y, z], axis=1).astype("<f4")


_POINTS = _backproject(_DEPTH)


class Stack:
    def __init__(self, tmp: Path, odom_source: str, depth_input: str) -> None:
        self.depth_input = depth_input
        self.domain = 30 + (os.getpid() % 30)
        self.ctx = rclpy.Context()
        rclpy.init(context=self.ctx, domain_id=self.domain)
        self.node = rclpy.create_node("stack_probe", context=self.ctx)
        self.ex = rclpy.executors.SingleThreadedExecutor(context=self.ctx)
        self.ex.add_node(self.node)
        env = {**os.environ, "ROS_DOMAIN_ID": str(self.domain)}
        self.proc = subprocess.Popen(
            [
                "ros2", "launch", "ugv_localization", "localization.launch.py",
                "mode:=mapping", "fresh_db:=true", f"odom_source:={odom_source}", "profile:=live_cam",
                f"database_path:={tmp / 'rtabmap.db'}", f"depth_input:={depth_input}",
                "depth_topic:=/camera/depth/image_raw",
            ],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        n = self.node
        self.pub_img = n.create_publisher(Image, "/camera/image_raw", _BEST_EFFORT)
        self.pub_info = n.create_publisher(CameraInfo, "/camera/camera_info", _BEST_EFFORT)
        self.pub_depth = n.create_publisher(Image, "/camera/depth/image_raw", 5)
        self.pub_cloud = n.create_publisher(PointCloud2, "/perception/depth_cloud", 10)  # Dev 1: reliable, depth 10
        self.pub_wheel = n.create_publisher(Odometry, "/wheel/odom", 20)
        static = n.create_publisher(TFMessage, "/tf_static", _LATCHED)
        t = TransformStamped()
        t.header.stamp = n.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "base_link", _FRAME
        t.transform.translation.z = 0.5
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = -0.5, 0.5, -0.5, 0.5
        static.publish(TFMessage(transforms=[t]))
        self.seen: dict[str, list] = {k: [] for k in ("rgbd", "info", "odom", "vo", "map", "valid", "status", "source", "tf")}
        n.create_subscription(RGBDImage, "/rtabmap/rgbd_image", self.seen["rgbd"].append, qos_profile_sensor_data)
        n.create_subscription(Info, "/rtabmap/info", self.seen["info"].append, 10)
        n.create_subscription(Odometry, "/odom", self.seen["odom"].append, 20)
        n.create_subscription(Odometry, "/rtabmap/odom_visual", self.seen["vo"].append, 20)
        n.create_subscription(OccupancyGrid, "/map", self.seen["map"].append, _LATCHED)
        n.create_subscription(Bool, "/ugv/pose_valid", lambda m: self.seen["valid"].append(m.data), 50)
        n.create_subscription(String, "/ugv/localization_status", lambda m: self.seen["status"].append(m.data), 10)
        n.create_subscription(String, "/ugv/localization/odom_source", lambda m: self.seen["source"].append(m.data), _LATCHED)
        n.create_subscription(
            TFMessage, "/tf", lambda m: self.seen["tf"].extend((x.header.frame_id, x.child_frame_id) for x in m.transforms), 100
        )

    def publish(self, *, wheel: bool, depth: bool) -> None:
        stamp = self.node.get_clock().now().to_msg()
        img = Image()
        img.header.stamp, img.header.frame_id = stamp, _FRAME
        img.height, img.width, img.encoding, img.step = _H, _W, "rgb8", 3 * _W
        img.data = _TEXTURE.tobytes()
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, _FRAME
        info.width, info.height, info.distortion_model = _W, _H, "plumb_bob"
        info.d, info.k = [0.0] * 5, list(_K)
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [_K[0], 0.0, _K[2], 0.0, 0.0, _K[4], _K[5], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_img.publish(img)
        self.pub_info.publish(info)
        if depth and self.depth_input == "cloud":
            c = PointCloud2()
            c.header.stamp, c.header.frame_id = stamp, _FRAME  # source image stamp (Dev 1 contract)
            c.height, c.width = 1, len(_POINTS)
            c.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
            c.is_bigendian, c.point_step, c.is_dense = False, 12, False
            c.row_step = 12 * len(_POINTS)
            c.data = _POINTS.tobytes()
            self.pub_cloud.publish(c)
        elif depth:
            d = Image()
            d.header.stamp, d.header.frame_id = stamp, _FRAME
            d.height, d.width, d.encoding, d.step = _H, _W, "32FC1", 4 * _W
            d.data = _DEPTH.tobytes()
            self.pub_depth.publish(d)
        if wheel:
            o = Odometry()
            o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, "odom", "base_link"
            o.pose.pose.orientation.w = 1.0
            o.pose.covariance = list(_COV)
            self.pub_wheel.publish(o)

    def run(self, seconds: float, *, wheel: bool = True, depth: bool = True, until=None, hz: float = 10.0) -> bool:
        end, nxt = time.monotonic() + seconds, 0.0
        while time.monotonic() < end:
            if time.monotonic() >= nxt:
                self.publish(wheel=wheel, depth=depth)
                nxt = time.monotonic() + 1.0 / hz
            self.ex.spin_once(timeout_sec=0.01)
            if until is not None and until():
                return True
        return until is None

    def close(self) -> str:
        os.killpg(self.proc.pid, signal.SIGINT)
        try:
            out, _ = self.proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            out, _ = self.proc.communicate()
        self.ex.shutdown()
        self.node.destroy_node()
        rclpy.shutdown(context=self.ctx)
        return out or ""


@pytest.fixture
def stack(tmp_path: Path, request):
    s = Stack(tmp_path, "auto", request.param)
    yield s
    log = s.close()
    (tmp_path / "launch.log").write_text(log, encoding="utf-8")
    print(f"launch log: {tmp_path / 'launch.log'}")


@pytest.mark.parametrize("stack", ["cloud", "image"], indirect=True)
def test_x1_full_stack_endpoints_auto(stack: Stack) -> None:
    seen = stack.seen
    # Dev 5 best-effort camera + Dev 1 depth → rgbd_sync exact-stamp pairs
    assert stack.run(40.0, until=lambda: len(seen["rgbd"]) >= 3), "rgbd_sync never paired RGB + depth + CameraInfo"
    if stack.depth_input == "cloud":  # the converted depth lands on its own pixels (same K, same size)
        rgbd = seen["rgbd"][-1]
        assert (rgbd.depth.width, rgbd.depth.height, rgbd.depth.encoding) == (_W, _H, "32FC1")
        d = np.frombuffer(bytes(rgbd.depth.data), dtype="<f4").reshape(_H, _W)
        valid = np.isfinite(d) & (d > 0)
        assert valid[_H // 6 + 2 :, :].mean() > 0.95  # below the sky band: depth everywhere
        assert not valid[: _H // 6 - 2, :].any()  # sky stays empty (no invented depth)
        assert np.allclose(d[valid], 3.0, atol=1e-3)
    # everything downstream of the pair
    ok = stack.run(
        60.0,
        until=lambda: seen["info"] and seen["odom"] and ("map", "odom") in seen["tf"] and seen["map"]
        and seen["valid"] and seen["valid"][-1],
    )
    assert seen["info"], "no /rtabmap/info"
    assert seen["odom"] and seen["source"] and seen["source"][-1] == "wheel", seen["source"]
    assert ("odom", "base_link") in seen["tf"] and ("map", "odom") in seen["tf"], set(seen["tf"])
    assert seen["map"], "no /map occupancy grid from depth"
    assert ok, f"pose never valid; statuses={seen['status']}"
    # rgbd_odometry runs in auto (fallback ready) on the same synced input
    assert stack.run(20.0, until=lambda: len(seen["vo"]) >= 1), "rgbd_odometry published nothing"

    # Dev 5 wheel odom stops → auto falls back to visual odometry, pose stays usable
    assert stack.run(20.0, wheel=False, until=lambda: seen["source"][-1] == "visual"), seen["source"]
    # Dev 1 depth stops → fail closed
    assert stack.run(10.0, wheel=False, depth=False, until=lambda: not seen["valid"][-1] and any(
        "depth_stale" in s for s in seen["status"][-3:])), seen["status"][-5:]
