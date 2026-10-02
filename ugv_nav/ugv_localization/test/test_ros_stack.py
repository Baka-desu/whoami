"""End-to-end endpoint test: the real launch file + real RTAB-Map nodes, fed synthetic sensors.

Proves the wiring between every endpoint in localization.launch.py (topic names, remaps, QoS,
exact-stamp sync, TF ownership) — not RTAB-Map accuracy. Inputs mimic the team contracts:
  Dev 5  /camera/image_raw rgb8 + /camera/camera_info per frame (BEST-EFFORT, like many drivers),
         /wheel/odom (no TF), /tf_static base_link -> camera_optical
  Dev 1  /perception/depth_cloud PointCloud2 (x/y/z float32 m, unorganized, sky dropped, back-projected
         with the raw K; source image stamp + optical frame) — exactly turing node/cloud.py's format.
         depth_input:=image (default): Dev 1 af7ebbf /perception/depth/image, 32FC1 m, NaN holes, RGB stamp.
x3 (Task 23) counts the frames that reach odometry (320x240 in the default case, 640x480 in the laptop case) and the QoS
   of the rgbd_image subscriptions.
x4 (Task 8) moves the robot (wheel odometry + the scene sliding past the camera) so RTAB-Map adds graph nodes,
   and checks the 3D map outputs: /rtabmap/cloud_map (from map_assembler), /rtabmap/mapPath, /rtabmap/mapData.
m1 (Task 8 review I1) is a measurement, skipped unless UGV_MEASURE_LATE_ATTACH is set: a viewer attaching cloud_map
   mid-mission vs the SLAM step rate and /ugv/pose_valid.
x6 (final review I2) launches with the default map_assembler:=false (D24): SLAM runs, and nothing publishes /rtabmap/cloud_map.
x5 (Task 9) checks /ugv/map/stats from the same moving stack: map_stats is launched, reads rtabmap's graph and info,
   and reports the calibration file's placeholder flag.
Skipped unless ROS 2 + rtabmap_ros + an installed ugv_localization are available (colcon test).
The scene is a static random texture at 3 m: synthetic test input, not a product calibration.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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
import yaml  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from rtabmap_msgs.msg import Info, MapData, MapGraph, OdomInfo, RGBDImage  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Bool, Float64, String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from ugv_localization.camera import calibration_from_camera_info, calibration_to_yaml_dict  # noqa: E402

_W, _H, _FRAME = 320, 240, "camera_optical"
_BIG = (640, 480)  # the live camera size: one rgbd_image (rgb8 + 32FC1 depth) is ~2.1 MB
_COV = [0.001 if i in (0, 7, 14, 21, 28, 35) else 0.0 for i in range(36)]
_LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
_BEST_EFFORT = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
_RELIABLE_CAMERA = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)  # like Dev 5's camera_driver
_RELIABLE_COUNT = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)  # counting must not lose frames itself
_RGBD_TOPIC, _ODOM_INFO_TOPIC = "/rtabmap/rgbd_image", "/rtabmap/odom_info"


def _backproject(depth: np.ndarray, k: list[float]) -> np.ndarray:
    """Same math as Dev 1 depth/geometry.backproject: optical frame, holes dropped."""
    v, u = np.nonzero(np.isfinite(depth))
    z = depth[v, u].astype(np.float64)
    x = (u - k[2]) * z / k[0]
    y = (v - k[5]) * z / k[4]
    return np.stack([x, y, z], axis=1).astype("<f4")


def _scene(w: int, h: int):
    """The static synthetic scene at w x h: (K, rgb8 texture, 32FC1 depth, back-projected points).
    Same field of view at every size (K scales with the image); 320x240 is the default scene."""
    k = [260.0 * w / _W, 0.0, w / 2, 0.0, 260.0 * h / _H, h / 2, 0.0, 0.0, 1.0]
    texture = np.random.default_rng(7).integers(0, 255, size=(h // 4, w // 4, 3), dtype=np.uint8).repeat(4, 0).repeat(4, 1)
    depth = np.full((h, w), 3.0, dtype="<f4")
    depth[: h // 6, :] = np.nan  # "sky" band
    return k, texture, depth, _backproject(depth, k)


_SCENE_DEPTH_M = 3.0  # the scene is a fronto-parallel plane this far from the camera (see _scene)


def _wide_texture(w: int, h: int, extra: int) -> np.ndarray:
    """A random texture w + extra pixels wide, same 4x4-pixel blocks as _scene: a moving camera sees a w-wide
    window of it, so every frame shows new texture and the scene never repeats (no accidental loop closures)."""
    return np.random.default_rng(7).integers(0, 255, size=(h // 4, (w + extra) // 4, 3), dtype=np.uint8).repeat(4, 0).repeat(4, 1)


def _spawn(argv: list[str], env: dict[str, str], log: Path):
    """Start argv in its own process group with stdout and stderr in the file `log`; returns (process, log file object).
    Never a pipe: nothing reads it while a test runs, a pipe holds 64 KB, and rtabmap logs a line per SLAM step. A long
    run then fills it and every node blocks in write() (measured: rtabmap and rgbd_odometry stopped after about 190 s, their
    main threads in pipe_write; docs/mapping/baseline.md "Review fixes for 0c5f1d9")."""
    out = open(log, "wb")
    return subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True), out


class Stack:
    def __init__(
        self, tmp: Path, odom_source: str, depth_input: str, *, timing: str = "default",
        size: tuple[int, int] = (_W, _H), camera_reliable: bool = False, speed: float = 0.0,
        placeholder_calibration: bool = False, grid_probe: bool = True, launch_args: tuple[str, ...] = (),
    ) -> None:
        self.depth_input = depth_input
        self.w, self.h = size
        self.k, self.texture, self.depth, self.points = _scene(self.w, self.h)
        # speed > 0 (m/s): the robot slides sideways at that speed. /wheel/odom reports the pose and the scene moves
        # across the image by the matching number of pixels (a plane at _SCENE_DEPTH_M: shift = fx * y / depth), so wheel
        # odometry, visual odometry and the pixels agree. Sideways because a pure image shift is exact; a real
        # differential drive cannot do it, the test only needs pose and pixels to move together. Texture for 5 minutes.
        self.speed = speed
        self.y = 0.0  # metres travelled sideways so far
        self._last_t: float | None = None
        self._max_shift = int(self.k[0] * speed * 300.0 / _SCENE_DEPTH_M) // 4 * 4  # pixels, a whole number of blocks
        self._wide = _wide_texture(self.w, self.h, self._max_shift) if speed else None
        self.published = 0  # camera frames published so far (each one complete: image + info + depth)
        self.domain = 30 + (os.getpid() % 30)
        self.ctx = rclpy.Context()
        rclpy.init(context=self.ctx, domain_id=self.domain)
        self.node = rclpy.create_node("stack_probe", context=self.ctx)
        self.ex = rclpy.executors.SingleThreadedExecutor(context=self.ctx)
        self.ex.add_node(self.node)
        env = {**os.environ, "ROS_DOMAIN_ID": str(self.domain)}
        calibration_args = []
        if placeholder_calibration:  # a calibration file that says "placeholder": map_stats reports it
            cal = calibration_from_camera_info(
                camera_name="stack_probe_cam", width=self.w, height=self.h, distortion_model="plumb_bob", d=[0.0] * 5,
                k=self.k, r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                p=[self.k[0], 0.0, self.k[2], 0.0, 0.0, self.k[4], self.k[5], 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            path = tmp / "placeholder_calibration.yaml"
            path.write_text(yaml.safe_dump(calibration_to_yaml_dict(dataclasses.replace(cal, placeholder=True))), encoding="utf-8")
            calibration_args = [f"calibration_file:={path}"]
        self.log_path = tmp / "launch.log"
        self.proc, self._log = _spawn(
            [
                "ros2", "launch", "ugv_localization", "localization.launch.py",
                "mode:=mapping", "fresh_db:=true", f"odom_source:={odom_source}", "profile:=live_cam",
                f"database_path:={tmp / 'rtabmap.db'}", f"depth_input:={depth_input}", f"timing:={timing}",
                *calibration_args, *launch_args,
            ],
            env, self.log_path,
        )
        n = self.node
        camera_qos = _RELIABLE_CAMERA if camera_reliable else _BEST_EFFORT
        self.pub_img = n.create_publisher(Image, "/camera/image_raw", camera_qos)
        self.pub_info = n.create_publisher(CameraInfo, "/camera/camera_info", camera_qos)
        self.pub_depth = n.create_publisher(Image, "/perception/depth/image", 10)  # Dev 1 af7ebbf: reliable, 10
        self.pub_cloud = n.create_publisher(PointCloud2, "/perception/depth_cloud", 10)  # Dev 1: reliable, depth 10
        self.pub_wheel = n.create_publisher(Odometry, "/wheel/odom", 20)
        static = n.create_publisher(TFMessage, "/tf_static", _LATCHED)
        t = TransformStamped()
        t.header.stamp = n.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "base_link", _FRAME
        t.transform.translation.z = 0.5
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = -0.5, 0.5, -0.5, 0.5
        static.publish(TFMessage(transforms=[t]))
        self.seen: dict[str, list] = {k: [] for k in ("rgbd", "info", "odom", "vo", "map", "valid", "status", "source", "tf", "dist", "basis")}
        n.create_subscription(RGBDImage, _RGBD_TOPIC, self.seen["rgbd"].append, qos_profile_sensor_data)
        n.create_subscription(Info, "/rtabmap/info", self.seen["info"].append, 10)
        n.create_subscription(Odometry, "/odom", self.seen["odom"].append, 20)
        n.create_subscription(Odometry, "/rtabmap/odom_visual", self.seen["vo"].append, 20)
        if grid_probe:  # off: nothing subscribes /map (as on the robot), so rtabmap keeps no map cache up to date
            n.create_subscription(OccupancyGrid, "/map", self.seen["map"].append, _LATCHED)
        n.create_subscription(Bool, "/ugv/pose_valid", lambda m: self.seen["valid"].append(m.data), 50)
        n.create_subscription(String, "/ugv/localization_status", lambda m: self.seen["status"].append(m.data), 10)
        n.create_subscription(String, "/ugv/localization/odom_source", lambda m: self.seen["source"].append(m.data), _LATCHED)
        n.create_subscription(Float64, "/ugv/localization/distance_travelled", lambda m: self.seen["dist"].append(m.data), 20)
        n.create_subscription(String, "/ugv/localization/distance_basis", lambda m: self.seen["basis"].append(m.data), _LATCHED)
        n.create_subscription(
            TFMessage, "/tf", lambda m: self.seen["tf"].extend((x.header.frame_id, x.child_frame_id) for x in m.transforms), 100
        )

    def _advance(self, now_s: float) -> np.ndarray:
        """Move the robot to now and return the rgb8 texture it sees there (the static scene when speed is 0)."""
        if self._wide is None:
            return self.texture
        if self._last_t is not None:
            self.y += self.speed * min(now_s - self._last_t, 0.5)  # a pause between run() calls is not travel
        self._last_t = now_s
        shift = min(int(round(self.k[0] * self.y / _SCENE_DEPTH_M)), self._max_shift)
        start = self._max_shift - shift  # moving left (+y) the scene slides right: the window slides left
        return self._wide[:, start : start + self.w]

    def publish(self, *, wheel: bool, depth: bool) -> None:
        now = self.node.get_clock().now()
        stamp = now.to_msg()
        img = Image()
        img.header.stamp, img.header.frame_id = stamp, _FRAME
        img.height, img.width, img.encoding, img.step = self.h, self.w, "rgb8", 3 * self.w
        img.data = self._advance(now.nanoseconds * 1e-9).tobytes()
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, _FRAME
        info.width, info.height, info.distortion_model = self.w, self.h, "plumb_bob"
        info.d, info.k = [0.0] * 5, list(self.k)
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [self.k[0], 0.0, self.k[2], 0.0, 0.0, self.k[4], self.k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_img.publish(img)
        self.pub_info.publish(info)
        self.published += depth  # complete frames only: image + info + depth (rgbd_sync needs all three)
        if depth and self.depth_input == "cloud":
            c = PointCloud2()
            c.header.stamp, c.header.frame_id = stamp, _FRAME  # source image stamp (Dev 1 contract)
            c.height, c.width = 1, len(self.points)
            c.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
            c.is_bigendian, c.point_step, c.is_dense = False, 12, False
            c.row_step = 12 * len(self.points)
            c.data = self.points.tobytes()
            self.pub_cloud.publish(c)
        elif depth:
            d = Image()
            d.header.stamp, d.header.frame_id = stamp, _FRAME
            d.height, d.width, d.encoding, d.step = self.h, self.w, "32FC1", 4 * self.w
            d.data = self.depth.tobytes()
            self.pub_depth.publish(d)
        if wheel:
            o = Odometry()
            o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, "odom", "base_link"
            o.pose.pose.position.y = self.y
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

    def count_frames(self) -> dict[str, int]:
        """Live counters for /rtabmap/rgbd_image and /rtabmap/odom_info. Both subscribers are RELIABLE with a
        deep queue, so this process never loses a frame itself: the counts are what the producers sent."""
        n = {"rgbd": 0, "odom_info": 0}

        def bump(key: str):
            def cb(_msg) -> None:
                n[key] += 1

            return cb

        # raw: count the serialized bytes, no 2.1 MB Python deserialization slowing this single-threaded probe down
        self.node.create_subscription(RGBDImage, _RGBD_TOPIC, bump("rgbd"), _RELIABLE_COUNT, raw=True)
        self.node.create_subscription(OdomInfo, _ODOM_INFO_TOPIC, bump("odom_info"), _RELIABLE_COUNT)
        return n

    def reliability(self, topic: str) -> dict[str, dict[str, str]]:
        """{'pub'|'sub': {node name: reliability}} for the stack's own endpoints on topic (this probe excluded)."""
        def names(infos) -> dict[str, str]:
            return {i.node_name: i.qos_profile.reliability.name for i in infos if i.node_name != "stack_probe"}

        return {
            "pub": names(self.node.get_publishers_info_by_topic(topic)),
            "sub": names(self.node.get_subscriptions_info_by_topic(topic)),
        }

    def watch_map_outputs(self) -> dict[str, list]:
        """Record /rtabmap/cloud_map, /rtabmap/mapPath and /rtabmap/mapData. rtabmap assembles them only while someone
        subscribes. QoS as the publishers have it (`ros2 topic info -v`, rtabmap_ros 0.23.7): cloud_map RELIABLE +
        TRANSIENT_LOCAL (node param `latch`), mapPath and mapData RELIABLE + VOLATILE. Deep queues: this probe is
        single-threaded and must not lose a message itself."""
        out: dict[str, list] = {"cloud": [], "path": [], "mapdata": []}
        volatile = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(PointCloud2, "/rtabmap/cloud_map", out["cloud"].append, latched)
        self.node.create_subscription(NavPath, "/rtabmap/mapPath", out["path"].append, volatile)
        self.node.create_subscription(MapData, "/rtabmap/mapData", out["mapdata"].append, volatile)
        return out

    def close(self) -> str:
        os.killpg(self.proc.pid, signal.SIGINT)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        self._log.close()
        self.ex.shutdown()
        self.node.destroy_node()
        rclpy.shutdown(context=self.ctx)
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def test_x0_a_chatty_launch_never_blocks(tmp_path: Path) -> None:
    """The launch output goes to a file. Nothing reads it while a test runs, and a 64 KB pipe filled in about 190 s of
    rtabmap's one-line-per-step log: every node then blocked in write() (rtabmap and rgbd_odometry stopped)."""
    proc, log = _spawn([sys.executable, "-c", "for i in range(3000): print('x' * 100, flush=True)"], dict(os.environ), tmp_path / "out.log")
    try:
        try:
            code = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail("the child did not finish: its output pipe is full and nobody reads it")
        assert code == 0
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()
    assert (tmp_path / "out.log").stat().st_size >= 300_000


@pytest.fixture
def stack(tmp_path: Path, request):
    if isinstance(request.param, dict):
        s = Stack(tmp_path, **{"odom_source": "auto", "depth_input": "image", **request.param})
    else:
        odom_source, depth_input = request.param if isinstance(request.param, tuple) else ("auto", request.param)
        s = Stack(tmp_path, odom_source, depth_input)
    yield s
    s.close()
    print(f"launch log: {s.log_path}")


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


@pytest.mark.parametrize("stack", [("visual", "image")], indirect=True)
def test_x2_visual_only_no_wheel_sensor(stack: Stack) -> None:
    """No /wheel/odom ever: camera + DA3 depth alone must give odom, the full TF chain and a valid pose."""
    seen = stack.seen
    ok = stack.run(
        60.0,
        wheel=False,
        until=lambda: seen["vo"] and seen["odom"] and ("odom", "base_link") in seen["tf"]
        and ("map", "odom") in seen["tf"] and seen["map"] and seen["valid"] and seen["valid"][-1],
    )
    # The pose can turn valid before this probe has discovered the latched source topic or the distance heartbeat
    # (seen in 3 of ~100 runs, which checked them the instant the pose was valid). Give them a moment; still asserted below.
    stack.run(5.0, wheel=False, until=lambda: seen["source"] and seen["dist"])
    assert seen["source"] and set(seen["source"]) == {"visual"}, seen["source"]
    assert seen["odom"] and seen["odom"][-1].header.frame_id == "odom", "no /odom from visual odometry"
    assert ("odom", "base_link") in seen["tf"] and ("map", "odom") in seen["tf"], set(seen["tf"])
    assert ok, f"pose never valid without wheel odom; statuses={seen['status']}"
    # distance_tracker is wired to the real /odom: heartbeat up; parked robot → nothing counted, label says so
    assert seen["dist"] and max(seen["dist"]) == 0.0, seen["dist"][-5:]
    assert seen["basis"] == ["none"], seen["basis"]
    # visual odometry is the only source: losing depth stops odometry → fail closed
    assert stack.run(10.0, wheel=False, depth=False, until=lambda: not seen["valid"][-1]), seen["status"][-5:]


_X3_HZ = 4.0  # synced frames per second: about the live rate (3.3 Hz); each message is ~2.1 MB at 640x480
_X3_WINDOW_S = 30.0


@pytest.mark.parametrize(
    "stack",
    [
        # default timing keeps best-effort inputs on purpose (they connect to any camera driver, interfaces.md), and
        # best effort cannot carry raw 640x480 frames: at that size only ~1 in 15 frames pairs, so no sample is left
        # to count. Its case therefore runs the documented best-effort camera at the small test size.
        {"timing": "default", "size": (_W, _H), "camera_reliable": False},
        # laptop timing is the live profile: reliable 640x480 camera like Dev 5's driver (~2.1 MB per rgbd_image).
        {"timing": "laptop", "size": _BIG, "camera_reliable": True},
    ],
    ids=["default-320x240-best-effort-camera", "laptop-640x480-reliable-camera"],
    indirect=True,
)
def test_x3_every_synced_frame_reaches_odometry_and_slam(stack: Stack) -> None:
    """Task 23. rgbd_sync sends 2.1 MB messages; a best-effort consumer silently drops some of them, so visual
    odometry saw ~1.2 of 3.3 frames/s. Both consumers must hold RELIABLE subscriptions (and rgbd_sync a RELIABLE
    publisher, or they never connect), and odometry must publish an odom_info for (nearly) every frame sent."""
    n = stack.count_frames()
    # Both consumers are up and wired to rgbd_sync (their subscriptions exist before any frame is needed).
    stack.run(
        60.0,
        until=lambda: {"rgbd_odometry", "rtabmap"} <= set(stack.reliability(_RGBD_TOPIC)["sub"]),
        hz=_X3_HZ,
    )
    # Warm-up: pipeline paired and odometry producing. Not asserted: a broken stack is reported with the counts below.
    stack.run(60.0, until=lambda: n["rgbd"] >= 5 and n["odom_info"] >= 2, hz=_X3_HZ)

    # Measured window: frames sent vs frames the two consumers' common source delivered vs odometry results.
    sent0, rgbd0, info0 = stack.published, n["rgbd"], n["odom_info"]
    stack.run(_X3_WINDOW_S, hz=_X3_HZ)
    stack.run(2.0, depth=False, hz=_X3_HZ)  # drain: images keep flowing but nothing new pairs, so nothing new is sent
    sent, rgbd, info = stack.published - sent0, n["rgbd"] - rgbd0, n["odom_info"] - info0
    wired = stack.reliability(_RGBD_TOPIC)
    counts = f"sent={sent} rgbd_image={rgbd} odom_info={info} ({info / max(rgbd, 1):.0%} of rgbd_image); endpoints={wired}"

    problems = []
    if wired["pub"].get("rgbd_sync") != "RELIABLE":
        problems.append("rgbd_sync does not publish /rtabmap/rgbd_image RELIABLE")
    for consumer in ("rgbd_odometry", "rtabmap"):
        if wired["sub"].get(consumer) != "RELIABLE":
            problems.append(f"{consumer} subscribes /rtabmap/rgbd_image {wired['sub'].get(consumer)}, not RELIABLE")
    if rgbd < 0.5 * sent:  # vacuity floor only: camera -> rgbd_sync is best effort in `default` (lossy by design), not under test
        problems.append("rgbd_sync delivered fewer than half of the frames sent: too few to count odometry against")
    if info < 0.9 * rgbd:
        problems.append("odometry produced fewer than 90% as many odom_info as rgbd_image frames")
    assert not problems, "; ".join(problems) + " | " + counts
    print(counts)


_X4_SPEED = 0.3  # m/s: RTAB-Map processes 2 frames/s (Rtabmap/DetectionRate), so 0.15 m per frame beats RGBD/LinearUpdate 0.1


def _cloud_field(c: PointCloud2, name: str, dtype: str) -> np.ndarray:
    """One 4-byte field of every point of c (dtype '<f4' for x/y/z, '<u4' for the packed rgb)."""
    f = next(f for f in c.fields if f.name == name)
    rows = np.frombuffer(bytes(c.data), dtype=np.uint8).reshape(-1, c.point_step)
    return rows[:, f.offset : f.offset + 4].copy().view(dtype)[:, 0]


def _z_span_problem(z: np.ndarray) -> str | None:
    """Why a cloud's heights say it is not 3D (None when they span at least 0.2 m). NaN-safe: a NaN span must not pass."""
    z = z[np.isfinite(z)]
    if not z.size:
        return "cloud_map has no finite z"
    span = float(z.max() - z.min())
    return None if span >= 0.2 else f"cloud_map is flat (z spans {span:.3f} m): Grid/3D is off"


def _depth_png_problem(depth: bytes, width: int, height: int, expect_m: float) -> str | None:
    """Why a node's right_compressed is not what the elevation mapper will decode: a PNG whose 4 bytes per pixel are a
    float32 depth in metres (rtabmap wraps 32-bit depth that way when Mem/SaveDepth16Format is false, it is not RVL)."""
    import cv2  # noqa: PLC0415 - only this check needs it

    im = cv2.imdecode(np.frombuffer(depth, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        return "right_compressed is not a decodable image"
    if im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 4:
        return f"right_compressed decodes to {im.shape} {im.dtype}, not 4 x uint8 (a float32 per pixel)"
    d = np.ascontiguousarray(im).view("<f4")[..., 0]
    if d.shape != (height, width):
        return f"right_compressed is {d.shape[1]}x{d.shape[0]}, the camera is {width}x{height}"
    valid = d[np.isfinite(d) & (d > 0)]
    if not valid.size or abs(float(np.median(valid)) - expect_m) > 0.05:
        return f"right_compressed depth median {float(np.median(valid)) if valid.size else None} m, the scene is at {expect_m} m"
    return None


def _map_data_problems(msgs: list, width: int, height: int, expect_m: float) -> tuple[list[str], int]:
    """(problems, graph nodes checked) for a list of /rtabmap/mapData messages. The message that brings a graph node carries
    its depth, camera info and camera-to-base transform. Later entries for the same id do not: for each processed frame that
    adds no node (the robot has not moved 0.1 m since the last one) RTAB-Map sends the newest graph node again, same id and
    stamp, with its images empty. So check the first entry of each id that is in graph.poses_id (an entry whose id is not
    in the graph is not a node delivery), and that every node of the final graph was delivered at least once."""
    first: dict[int, object] = {}
    for m in msgs:
        in_graph = set(m.graph.poses_id)
        for node in m.nodes:
            if node.id in in_graph:
                first.setdefault(node.id, node.data)
    problems = []
    if len(first) < 3:
        problems.append(f"mapData delivered only {len(first)} graph node(s) in {len(msgs)} messages")
    bad = [(i, f) for i, d in first.items() for f in ("right_compressed", "left_camera_info", "local_transform") if not len(getattr(d, f))]
    if bad:
        problems.append(f"mapData graph nodes delivered with empty data: {bad[:6]}")
    missing = sorted(i for i in (msgs[-1].graph.poses_id if msgs else ()) if i > 0 and i not in first)
    if missing:
        problems.append(f"graph nodes never delivered through mapData: {missing[:6]}")
    png = next((d.right_compressed for _, d in sorted(first.items()) if len(d.right_compressed)), None)
    if png is not None and (why := _depth_png_problem(bytes(png), width, height, expect_m)):
        problems.append(why)
    return problems, len(first)


@pytest.mark.parametrize(
    "stack", [{"speed": _X4_SPEED, "launch_args": ("map_assembler:=true",)}], ids=["moving-320x240"], indirect=True
)
def test_x4_rtabmap_3d_map_outputs(stack: Stack) -> None:
    """Task 8. RTAB-Map must publish its 3D products, not only the 2D /map: a coloured 3D /rtabmap/cloud_map, the
    growing /rtabmap/mapPath, and /rtabmap/mapData whose graph nodes carry what the elevation mapper needs (depth,
    camera info, camera-to-base transform). The robot has to move, or RTAB-Map adds no graph nodes after the first."""
    out = stack.watch_map_outputs()
    seen = stack.seen
    stack.run(
        90.0,
        until=lambda: out["path"] and len(out["path"][-1].poses) >= 2 and any(c.width * c.height for c in out["cloud"]),
    )
    poses_early = len(out["path"][-1].poses) if out["path"] else 0
    stack.run(10.0)  # keep moving: the path must keep growing
    problems = []

    # /map: rtabmap's 2D occupancy grid must survive Grid/3D (nothing in the repo consumes it yet; the gateway reads /global_costmap)
    grid = seen["map"][-1] if seen["map"] else None
    if grid is None or not any(v != -1 for v in grid.data):
        problems.append("/map is missing or has no known cell")

    # /rtabmap/cloud_map: non-empty, coloured, genuinely 3D (Grid/3D false gives a flat z = 0 cloud)
    clouds = [c for c in out["cloud"] if c.width * c.height]
    names = [f.name for f in clouds[-1].fields] if clouds else []
    z = _cloud_field(clouds[-1], "z", "<f4") if clouds else np.zeros(0, dtype="<f4")
    z_span = float(np.nanmax(z) - np.nanmin(z)) if np.isfinite(z).any() else float("nan")
    if not clouds:
        problems.append("/rtabmap/cloud_map never had a point")
    else:
        colour_field = next((n for n in ("rgb", "rgba") if n in names), None)
        if colour_field is None:
            problems.append(f"cloud_map has no rgb/rgba field: {names}")
        elif not _cloud_field(clouds[-1], colour_field, "<u4").any():
            problems.append("cloud_map colour is all zero")
        if why := _z_span_problem(z):
            problems.append(why)
        if clouds[-1].header.frame_id != "map":
            problems.append(f"cloud_map frame_id {clouds[-1].header.frame_id!r}, not 'map'")
    # ... and it comes from map_assembler, not from rtabmap: rtabmap assembles its own copy inside the SLAM step, and a
    # viewer opened mid-mission made that step assemble the whole map at once (review I1, docs/mapping/baseline.md)
    cloud_pubs = sorted({i.node_name for i in stack.node.get_publishers_info_by_topic("/rtabmap/cloud_map")})
    if cloud_pubs != ["map_assembler"]:
        problems.append(f"/rtabmap/cloud_map is published by {cloud_pubs}, not by map_assembler alone")

    # /rtabmap/mapPath: at least 2 poses, and more of them later in the run
    lens = [len(p.poses) for p in out["path"]]
    if not lens or max(lens) < 2:
        problems.append(f"mapPath never had 2 poses: {lens[-5:]}")
    elif lens[-1] < poses_early + 3:
        problems.append(f"mapPath did not grow: {poses_early} poses early, {lens[-1]} at the end")

    # /rtabmap/mapData: what the elevation mapper will consume (see _map_data_problems)
    md_problems, checked = _map_data_problems(out["mapdata"], stack.w, stack.h, _SCENE_DEPTH_M)
    problems += md_problems

    summary = (
        f"clouds={len(out['cloud'])} (points {[c.width * c.height for c in out['cloud'][-3:]]}, fields {names}, z span {z_span:.2f} m) "
        f"mapPath lens={lens[:3]}..{lens[-3:]} (early {poses_early}) mapData msgs={len(out['mapdata'])} graph nodes checked={checked}"
    )
    assert not problems, "; ".join(problems) + " | " + summary
    print(summary)


def _float_depth_png(w: int, h: int, metres: float = 3.0) -> bytes:
    """What rtabmap puts in right_compressed: a w x h float32 depth image reinterpreted as 4 x uint8 and PNG-compressed."""
    import cv2  # noqa: PLC0415

    d = np.full((h, w), metres, dtype="<f4")
    d[: h // 6] = np.nan  # the sky band
    return bytes(cv2.imencode(".png", d.view(np.uint8).reshape(h, w, 4))[1])


def _md(graph_ids: list[int], *entries: tuple[int, bytes]) -> SimpleNamespace:
    """A fake /rtabmap/mapData message: the graph and the node entries (id, right_compressed)."""
    nodes = [SimpleNamespace(id=i, data=SimpleNamespace(right_compressed=png, left_camera_info=[0], local_transform=[0])) for i, png in entries]
    return SimpleNamespace(graph=SimpleNamespace(poses_id=graph_ids), nodes=nodes)


def test_x4a_map_data_checks_node_deliveries_not_image_less_repeats() -> None:
    png = _float_depth_png(64, 48)
    msgs = [_md([1], (1, png)), _md([1], (1, b"")), _md([1, 2], (2, png)), _md([1, 2, 3], (3, png)), _md([1, 2, 3], (3, b""))]
    assert _map_data_problems(msgs, 64, 48, 3.0) == ([], 3)  # the repeats of nodes 1 and 3 carry no images and are fine


def test_x4a_map_data_flags_a_node_that_never_arrived_or_arrived_empty() -> None:
    png = _float_depth_png(64, 48)
    problems, _ = _map_data_problems([_md([1, 2, 3], (1, png), (2, png), (3, png)), _md([1, 2, 3, 4])], 64, 48, 3.0)
    assert problems == ["graph nodes never delivered through mapData: [4]"]
    problems, _ = _map_data_problems([_md([1, 2, 3], (1, png), (2, b""), (3, png))], 64, 48, 3.0)
    assert problems == ["mapData graph nodes delivered with empty data: [(2, 'right_compressed')]"]
    problems, _ = _map_data_problems([_md([1, 2], (1, png), (2, png))], 64, 48, 3.0)
    assert problems and problems[0].startswith("mapData delivered only 2 graph node(s)")


def test_x4a_depth_must_be_a_png_of_float32_metres() -> None:
    import cv2

    assert _depth_png_problem(_float_depth_png(64, 48), 64, 48, 3.0) is None
    assert "not a decodable image" in _depth_png_problem(b"\x00\x01 an rvl blob", 64, 48, 3.0)
    gray16 = bytes(cv2.imencode(".png", np.full((48, 64), 3000, dtype=np.uint16))[1])  # the 16-bit millimetre format
    assert "not 4 x uint8" in _depth_png_problem(gray16, 64, 48, 3.0)
    assert "camera is 128x96" in _depth_png_problem(_float_depth_png(64, 48), 128, 96, 3.0)
    assert "median" in _depth_png_problem(_float_depth_png(64, 48, metres=7.0), 64, 48, 3.0)


def test_x4a_flat_or_nan_heights_are_not_3d() -> None:
    nan = float("nan")
    assert _z_span_problem(np.array([0.0, 0.0, 0.0], dtype="<f4")) is not None  # Grid/3D false: z = 0
    assert _z_span_problem(np.array([nan, nan], dtype="<f4")) is not None  # a NaN span must not pass as "not flat"
    assert _z_span_problem(np.zeros(0, dtype="<f4")) is not None
    assert _z_span_problem(np.array([nan, -0.8, 0.9], dtype="<f4")) is None


_STEP_LOG = re.compile(
    r"\[(?P<t>\d+\.\d+)\] \[rtabmap\.rtabmap\]: rtabmap \((?P<node>\d+)\): .*?RTAB-Map=(?P<core>[\d.]+)s, "
    r"Maps update=(?P<maps>[\d.]+)s pub=(?P<pub>[\d.]+)s"
)
_ASSEMBLER_LOG = re.compile(  # map_assembler, once per /rtabmap/mapData message it processes
    r"\[(?P<t>\d+\.\d+)\] \[[\w.]*map_assembler\]: map_assembler: Updating = (?P<upd>[\d.]+)s, Publishing data = (?P<pubd>[\d.]+)s"
)


_RSS_PROCS = ("map_assembler", "rtabmap")


def _rss_mb(root_pid: int) -> dict[str, float]:
    """Resident memory (MB, VmRSS) of the processes named in _RSS_PROCS that descend from root_pid (the launch), from /proc."""
    procs: dict[int, tuple[str, int]] = {}  # pid -> (comm, parent pid)
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
            comm = stat[stat.index("(") + 1 : stat.rindex(")")]
            procs[int(entry)] = (comm, int(stat[stat.rindex(")") + 2 :].split()[1]))
        except (OSError, ValueError, IndexError):
            continue
    out: dict[str, float] = {}
    for pid, (comm, _ppid) in procs.items():
        if comm not in _RSS_PROCS:
            continue
        up, hops = pid, 0
        while up not in (root_pid, 0, 1) and up in procs and hops < 20:
            up, hops = procs[up][1], hops + 1
        if up != root_pid:
            continue
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    out[comm] = round(out.get(comm, 0.0) + int(line.split()[1]) / 1024.0, 1)
        except (OSError, ValueError):
            continue
    return out


class _RssSampler(threading.Thread):
    """Samples _rss_mb every period_s on its own thread (no ROS calls), so the probe's publish loop is not delayed."""

    def __init__(self, root_pid: int, period_s: float) -> None:
        super().__init__(daemon=True)
        self.root_pid, self.period_s = root_pid, period_s
        self.samples: list[tuple[int, dict[str, float]]] = []  # (wall time ns, {comm: MB})
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.samples.append((time.time_ns(), _rss_mb(self.root_pid)))
            self._halt.wait(self.period_s)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5.0)


def _late_attach_window(infos: list, statuses: list, steps: list, assembled: list, t0: int, t1: int) -> dict:
    """Numbers for the arrival-time window [t0, t1) ns: /rtabmap/info steps, the largest gap between consecutive infos whose
    later one arrived in the window (the pair spanning t0 counts), the largest stamp age an info reached before the next one
    replaced it (what pose_validity's slam_max_age_s is compared with), the pose status reasons seen, rtabmap's own
    per-step `Maps update` + `pub` time from its log, and map_assembler's messages and largest update + publish time."""
    gap = age = 0.0
    n = 0
    for (a0, s0), (a1, _s1) in zip(infos, infos[1:]):
        if t0 <= a1 < t1:
            n += 1
            gap, age = max(gap, (a1 - a0) / 1e9), max(age, (a1 - s0) / 1e9)
    reasons = sorted({r for a, s in statuses if t0 <= a < t1 for r in s.split(",") if r != "valid"})
    maps = [st for st in steps if t0 <= st["t"] < t1]
    asm = [a for a in assembled if t0 <= a["t"] < t1]
    return {
        "steps": n, "max_gap_s": round(gap, 3), "max_stamp_age_s": round(age, 3), "reasons": reasons,
        "max_maps_pub_s": round(max((st["maps"] + st["pub"] for st in maps), default=0.0), 4),
        "max_step_s": round(max((st["core"] + st["maps"] + st["pub"] for st in maps), default=0.0), 4),
        "log_steps": len(maps), "assembler_msgs": len(asm),
        "assembler_max_s": round(max((a["upd"] + a["pubd"] for a in asm), default=0.0), 4),
    }


@pytest.mark.skipif(
    not os.environ.get("UGV_MEASURE_LATE_ATTACH"),
    reason="measurement, not a check (docs/mapping/baseline.md 'Task 8 fix round 1'): set UGV_MEASURE_LATE_ATTACH=1",
)
@pytest.mark.parametrize(
    "stack",
    [
        {"speed": _X4_SPEED, "grid_probe": False, "launch_args": ("map_assembler:=true",)},
        {"speed": _X4_SPEED, "grid_probe": False, "timing": "laptop", "size": _BIG, "camera_reliable": True,
         "launch_args": ("map_assembler:=true",)},
    ],
    ids=["default-320x240", "laptop-640x480-reliable-camera"],
    indirect=True,
)
def test_m1_measure_late_cloud_map_attach(stack: Stack) -> None:
    """Task 8 review I1. The gateway subscribes /rtabmap/cloud_map + /rtabmap/mapPath only while a viewer is open (destroyed
    idle_timeout_s = 10 s after the last request), so rtabmap first assembles the whole cloud in its SLAM callback when
    someone opens the viewer mid-mission. Map UGV_MEASURE_MAP_S seconds (default 150) with no map subscriber (not even
    /map: nothing subscribes it on the robot), then attach like the gateway, detach and re-attach (UGV_MEASURE_CYCLES,
    default 3). Per attach: the /rtabmap/info gap and stamp age in the 10 s after it vs the 20 s before, pose status
    reasons, cloud size, rtabmap's own map-assembly time, and (since the fix) map_assembler's time and message count,
    which must equal rtabmap's steps (its mapData subscription keeps 1). Also samples the RSS of map_assembler and rtabmap
    every UGV_MEASURE_RSS_PERIOD_S seconds (default 5) with the graph node reached at that time (final review I2).
    Appends one JSON line to UGV_MEASURE_OUT if set."""
    map_s = float(os.environ.get("UGV_MEASURE_MAP_S", "150"))
    cycles = int(os.environ.get("UGV_MEASURE_CYCLES", "3"))
    rss = _RssSampler(stack.proc.pid, float(os.environ.get("UGV_MEASURE_RSS_PERIOD_S", "5")))
    rss.start()
    hz = _X3_HZ if stack.w == _BIG[0] else 10.0  # 640x480: about the live rate (each rgbd_image is 2.1 MB)
    clock, node = stack.node.get_clock(), stack.node
    infos: list[tuple[int, int]] = []  # (arrival ns, header stamp ns)
    statuses: list[tuple[int, str]] = []
    node.create_subscription(
        Info, "/rtabmap/info", lambda m: infos.append((clock.now().nanoseconds, m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)), 50
    )
    node.create_subscription(String, "/ugv/localization_status", lambda m: statuses.append((clock.now().nanoseconds, m.data)), 50)
    start = clock.now().nanoseconds
    stack.run(map_s, hz=hz)
    attaches = []
    for _ in range(cycles):
        clouds: list[tuple[int, int, int]] = []  # (arrival ns, points, stamp ns)
        t_attach = clock.now().nanoseconds
        subs = [  # the gateway's QoS: depth 1, reliable, the offered durability (cloud_map latched, mapPath volatile)
            node.create_subscription(
                PointCloud2, "/rtabmap/cloud_map",
                lambda m: clouds.append((clock.now().nanoseconds, m.width * m.height, m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            ),
            node.create_subscription(NavPath, "/rtabmap/mapPath", lambda _m: None, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)),
        ]
        stack.run(15.0, hz=hz)
        for sub in subs:
            node.destroy_subscription(sub)
        fresh = [c for c in clouds if c[2] >= t_attach - 2 * 10**9]  # not the latched cloud of the previous attach
        attaches.append({
            "t_attach_s": round((t_attach - start) / 1e9, 1),
            "first_fresh_cloud_after_s": round((fresh[0][0] - t_attach) / 1e9, 2) if fresh else None,
            "cloud_points": [fresh[0][1], fresh[-1][1]] if fresh else None, "clouds": len(clouds),
            "t0": t_attach,
        })
        stack.run(20.0, hz=hz)  # viewer closed: the gateway drops the subscriptions; mapping goes on
    rss.stop()
    log = stack.log_path.read_text(encoding="utf-8", errors="replace")
    steps = [
        {"t": int(float(m["t"]) * 1e9), "node": int(m["node"]), "core": float(m["core"]), "maps": float(m["maps"]), "pub": float(m["pub"])}
        for m in _STEP_LOG.finditer(log)
    ]
    assembled = [{"t": int(float(m["t"]) * 1e9), "upd": float(m["upd"]), "pubd": float(m["pubd"])} for m in _ASSEMBLER_LOG.finditer(log)]
    s = 10**9
    for a in attaches:
        t0 = a.pop("t0")
        a["graph_node_at_attach"] = max((st["node"] for st in steps if st["t"] < t0), default=None)
        a["before_20s"] = _late_attach_window(infos, statuses, steps, assembled, t0 - 20 * s, t0)  # no viewer
        a["after_10s"] = _late_attach_window(infos, statuses, steps, assembled, t0, t0 + 10 * s)
    result = {
        "profile": "laptop-640x480" if stack.w == _BIG[0] else "default-320x240", "map_s": map_s, "hz": hz,
        "infos": len(infos), "log_steps": len(steps), "assembler_msgs": len(assembled), "attaches": attaches,
        "cloud_map_publishers": sorted({i.node_name for i in node.get_publishers_info_by_topic("/rtabmap/cloud_map")}),
        "no_viewer_60s_to_first_attach": _late_attach_window(
            infos, statuses, steps, assembled, start + 60 * s, start + int(map_s * 1e9)
        ),
        "whole_run": _late_attach_window(infos, statuses, steps, assembled, start, clock.now().nanoseconds),
        "rss_mb": [
            {"t_s": round((t - start) / 1e9, 1), "node": max((st["node"] for st in steps if st["t"] < t), default=None), **mb}
            for t, mb in rss.samples
        ],
    }
    print(json.dumps(result))
    if out := os.environ.get("UGV_MEASURE_OUT"):
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    assert infos and all(a["cloud_points"] for a in attaches), "the measurement did not run: " + json.dumps(result)


_STATS_KEYS = ("keyframes", "loop_closures", "path_length_m", "db_bytes", "last_update_age_s", "mode", "calibration_placeholder")


@pytest.mark.parametrize(
    "stack", [{"speed": _X4_SPEED, "placeholder_calibration": True}], ids=["moving-320x240-placeholder-calibration"], indirect=True
)
def test_x5_map_stats_on_the_real_stack(stack: Stack) -> None:
    """Task 9. localization.launch.py starts map_stats and it works against the real rtabmap publishers: the graph
    (reliable + transient local) and info (reliable) connect, the keyframe count and path length follow the moving
    robot, the calibration file passed to the launch reaches the node, and the node never subscribes to cloud_map or
    mapData (either makes rtabmap assemble and send the whole map). Then the robot stops: rtabmap publishes no new graph
    (so the age must grow) but keeps reporting a loop closure with the same node in every step (so the count must not)."""
    stats: list[dict] = []
    stack.node.create_subscription(String, "/ugv/map/stats", lambda m: stats.append(json.loads(m.data)), _LATCHED)
    stack.run(90.0, until=lambda: bool(stats) and stats[-1]["keyframes"] >= 3 and stats[-1]["last_update_age_s"] is not None)
    assert stats, "/ugv/map/stats never published"
    early = stats[-1]["keyframes"]
    stack.run(10.0)  # keep moving: the numbers must keep following the graph
    moving = stats[-1]
    stack.speed = 0.0  # the robot stops
    stack.run(3.0)  # frames already in flight may still add a keyframe or a match
    parked = stats[-1]
    stack.run(8.0)
    last = stats[-1]

    problems = []
    if tuple(last) != _STATS_KEYS:
        problems.append(f"keys {tuple(last)}")
    if not all(v is None or type(v) in (bool, int, float, str) for v in last.values()):
        problems.append("a value is not a JSON scalar")
    if moving["keyframes"] < early + 3:
        problems.append(f"keyframes did not follow the graph: {early} early, {moving['keyframes']} after 10 s more")
    if not last["path_length_m"] or last["path_length_m"] <= 0.0:
        problems.append(f"path_length_m {last['path_length_m']}")
    if moving["last_update_age_s"] is None or not 0.0 <= moving["last_update_age_s"] < 5.0:
        problems.append(f"last_update_age_s {moving['last_update_age_s']} while the graph keeps growing")
    if last["last_update_age_s"] is None or last["last_update_age_s"] < 7.0:
        problems.append(f"last_update_age_s {last['last_update_age_s']} after 11 s parked: it did not grow")
    if last["keyframes"] != parked["keyframes"]:
        problems.append(f"keyframes {parked['keyframes']} -> {last['keyframes']} with the robot parked")
    if type(last["loop_closures"]) is not int or last["loop_closures"] > parked["loop_closures"] + 1:
        problems.append(f"loop_closures {parked['loop_closures']} -> {last['loop_closures']!r} with the robot parked")
    if last["db_bytes"] is not None and type(last["db_bytes"]) is not int:
        problems.append(f"db_bytes {last['db_bytes']!r}")
    if last["mode"] != "mapping":
        problems.append(f"mode {last['mode']!r}")
    if last["calibration_placeholder"] is not True:
        problems.append("calibration_placeholder is not true: calibration_file did not reach map_stats")
    for topic in ("/rtabmap/cloud_map", "/rtabmap/mapData"):
        subscribers = {i.node_name for i in stack.node.get_subscriptions_info_by_topic(topic)}
        if "map_stats" in subscribers:
            problems.append(f"map_stats subscribes {topic}")
    assert not problems, "; ".join(problems) + f" | stats={last} messages={len(stats)}"
    print(f"map stats moving: {json.dumps(moving)}")
    print(f"map stats parked: {json.dumps(last)}")


@pytest.mark.parametrize(
    "stack", [{"speed": _X4_SPEED}], ids=["moving-320x240-default-no-assembler"], indirect=True
)
def test_x6_map_assembler_off_leaves_slam_running_and_no_cloud_map(stack: Stack) -> None:
    """Final review I2 + PR #40 review (mindmap D24): the default (map_assembler:=false, opt-in) starts no map_assembler, so nothing
    publishes /rtabmap/cloud_map (rtabmap's own copy stays on /rtabmap/slam/cloud_map); SLAM, mapGraph and the graph
    still grow."""
    seen = stack.seen
    graphs: list = []
    stack.node.create_subscription(MapGraph, "/rtabmap/mapGraph", graphs.append, _LATCHED)
    assert stack.run(60.0, until=lambda: len(seen["info"]) >= 8 and graphs and len(graphs[-1].poses) >= 5), (
        f"SLAM did not run without the assembler: {len(seen['info'])} info, {len(graphs)} graphs"
    )
    nodes = set(stack.node.get_node_names())
    assert "map_assembler" not in nodes and "rtabmap" in nodes, sorted(nodes)
    assert stack.node.get_publishers_info_by_topic("/rtabmap/cloud_map") == []
    assert "map_assembler=False" in stack.log_path.read_text(encoding="utf-8", errors="replace")
