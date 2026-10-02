"""End-to-end endpoint test: the real launch file + real RTAB-Map nodes, fed synthetic sensors.

Proves the wiring between every endpoint in localization.launch.py (topic names, remaps, QoS,
exact-stamp sync, TF ownership) — not RTAB-Map accuracy. Inputs mimic the team contracts:
  Dev 5  /camera/image_raw rgb8 + /camera/camera_info per frame (BEST-EFFORT, like many drivers),
         /wheel/odom (no TF), /tf_static base_link -> camera_optical
  Dev 1  /perception/depth_cloud PointCloud2 (x/y/z float32 m, unorganized, sky dropped, back-projected
         with the raw K; source image stamp + optical frame) — exactly turing node/cloud.py's format.
         depth_input:=image (default): Dev 1 af7ebbf /perception/depth/image, 32FC1 m, NaN holes, RGB stamp.
x3 (Task 23) counts real 640x480 frames and the QoS of the rgbd_image subscriptions.
x4 (Task 8) moves the robot (wheel odometry + the scene sliding past the camera) so RTAB-Map adds graph nodes,
   and checks the 3D map outputs: /rtabmap/cloud_map, /rtabmap/mapPath, /rtabmap/mapData.
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
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from rtabmap_msgs.msg import Info, MapData, OdomInfo, RGBDImage  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Bool, Float64, String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

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


_K, _TEXTURE, _DEPTH, _POINTS = _scene(_W, _H)
_SCENE_DEPTH_M = 3.0  # the scene is a fronto-parallel plane this far from the camera (see _scene)


def _wide_texture(w: int, h: int, extra: int) -> np.ndarray:
    """A random texture w + extra pixels wide, same 4x4-pixel blocks as _scene: a moving camera sees a w-wide
    window of it, so every frame shows new texture and the scene never repeats (no accidental loop closures)."""
    return np.random.default_rng(7).integers(0, 255, size=(h // 4, (w + extra) // 4, 3), dtype=np.uint8).repeat(4, 0).repeat(4, 1)


class Stack:
    def __init__(
        self, tmp: Path, odom_source: str, depth_input: str, *, timing: str = "default",
        size: tuple[int, int] = (_W, _H), camera_reliable: bool = False, speed: float = 0.0,
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
        self.proc = subprocess.Popen(
            [
                "ros2", "launch", "ugv_localization", "localization.launch.py",
                "mode:=mapping", "fresh_db:=true", f"odom_source:={odom_source}", "profile:=live_cam",
                f"database_path:={tmp / 'rtabmap.db'}", f"depth_input:={depth_input}", f"timing:={timing}",
            ],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
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
    if isinstance(request.param, dict):
        s = Stack(tmp_path, **{"odom_source": "auto", "depth_input": "image", **request.param})
    else:
        odom_source, depth_input = request.param if isinstance(request.param, tuple) else ("auto", request.param)
        s = Stack(tmp_path, odom_source, depth_input)
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
    if rgbd < 0.9 * sent:  # not vacuous: the sync really delivered the frames we sent
        problems.append("rgbd_sync delivered fewer than 90% of the frames sent")
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


@pytest.mark.parametrize("stack", [{"speed": _X4_SPEED}], ids=["moving-320x240"], indirect=True)
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

    # /map: the 2D occupancy grid stays (Nav2 and the pose checks use it)
    grid = seen["map"][-1] if seen["map"] else None
    if grid is None or not any(v != -1 for v in grid.data):
        problems.append("/map is missing or has no known cell")

    # /rtabmap/cloud_map: non-empty, coloured, genuinely 3D (Grid/3D false gives a flat z = 0 cloud)
    clouds = [c for c in out["cloud"] if c.width * c.height]
    names = [f.name for f in clouds[-1].fields] if clouds else []
    z_span = 0.0
    if clouds:
        z = _cloud_field(clouds[-1], "z", "<f4")
        z_span = float(z.max() - z.min())
    if not clouds:
        problems.append("/rtabmap/cloud_map never had a point")
    else:
        colour_field = next((n for n in ("rgb", "rgba") if n in names), None)
        if colour_field is None:
            problems.append(f"cloud_map has no rgb/rgba field: {names}")
        elif not _cloud_field(clouds[-1], colour_field, "<u4").any():
            problems.append("cloud_map colour is all zero")
        if z_span < 0.2:
            problems.append(f"cloud_map is flat (z spans {z_span:.3f} m): Grid/3D is off")
        if clouds[-1].header.frame_id != "map":
            problems.append(f"cloud_map frame_id {clouds[-1].header.frame_id!r}, not 'map'")

    # /rtabmap/mapPath: at least 2 poses, and more of them later in the run
    lens = [len(p.poses) for p in out["path"]]
    if not lens or max(lens) < 2:
        problems.append(f"mapPath never had 2 poses: {lens[-5:]}")
    elif lens[-1] < poses_early + 3:
        problems.append(f"mapPath did not grow: {poses_early} poses early, {lens[-1]} at the end")

    # /rtabmap/mapData: the message that brings a graph node carries its depth, camera info and camera-to-base transform.
    # Later entries for the same id do not: for each processed frame that adds no node (the robot has not moved 0.1 m since
    # the last one) RTAB-Map sends the newest graph node again, same id and stamp, with its images empty. So check the
    # first entry of each id that is in graph.poses_id (an entry whose id is not in the graph is not a node delivery).
    first: dict[int, object] = {}
    for m in out["mapdata"]:
        in_graph = set(m.graph.poses_id)
        for node in m.nodes:
            if node.id in in_graph:
                first.setdefault(node.id, node.data)
    bad = [(i, f) for i, d in first.items() for f in ("right_compressed", "left_camera_info", "local_transform") if not len(getattr(d, f))]
    checked = len(first)
    if checked < 3:
        problems.append(f"mapData delivered only {checked} graph node(s) in {len(out['mapdata'])} messages")
    if bad:
        problems.append(f"mapData graph nodes delivered with empty data: {bad[:6]}")

    summary = (
        f"clouds={len(out['cloud'])} (points {[c.width * c.height for c in out['cloud'][-3:]]}, fields {names}, z span {z_span:.2f} m) "
        f"mapPath lens={lens[:3]}..{lens[-3:]} (early {poses_early}) mapData msgs={len(out['mapdata'])} graph nodes checked={checked}"
    )
    assert not problems, "; ".join(problems) + " | " + summary
    print(summary)
