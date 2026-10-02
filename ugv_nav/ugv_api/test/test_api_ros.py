"""Gateway on a real ROS graph over real HTTP (uvicorn on a free port, httpx client).

Real rclpy nodes publish the contract topics; no Nav2 or RTAB-Map is running, so goals and mode
switches must answer 503 problems, and a tripped §12 watch must answer 409. Runs under colcon test
(or any shell with ROS sourced); skipped in plain Python.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import socket
import threading
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Path  # noqa: E402
from rclpy.context import Context  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402

from ugv_api import mapcodec as codec  # noqa: E402
from ugv_api.app import create_app  # noqa: E402
from ugv_api.goals import GoalRegistry  # noqa: E402
from ugv_api.mapstore import MapStore  # noqa: E402
from ugv_api.ros_node import GatewayNode  # noqa: E402
from ugv_api.state import StateStore  # noqa: E402

PROBLEM = "application/problem+json"

# Short on purpose: the map tests wait for the demand timer (1 Hz) and for these limits to pass.
IDLE_S = 1.5
STATS_STALE_S = 1.5

CLOUD, PATH = "/rtabmap/cloud_map", "/rtabmap/mapPath"
ELEV_CLOUD, ELEV_GRID = "/ugv/elevation/cloud", "/ugv/elevation/obstacles"
GRID, DEPTH, CAMERA = "/global_costmap/costmap", "/perception/depth/image", "/image_raw/compressed"
MAP_STATS, PERCEPTION_STATS = "/ugv/map/stats", "/ugv/perception/stats"
HEAVY_TOPICS = (CLOUD, PATH, ELEV_CLOUD, ELEV_GRID, GRID, DEPTH, CAMERA)

# What Inputs broadcasts: map -> base_link at POSE, base_link -> camera_optical_frame at CAMERA_MOUNT, and a
# CameraInfo for an 8 x 6 camera.
POSE = {"x": 1.5, "y": -0.5, "z": 0.25, "yaw": 0.4}
CAMERA_MOUNT = (0.1, 0.0, 0.3)  # metres in base_link
OPTICAL_TO_BODY_Q = (-0.5, 0.5, -0.5, 0.5)  # optical (x right, y down, z forward) -> body (x forward, y left, z up)
K = (10.0, 0.0, 4.0, 0.0, 10.0, 2.0, 0.0, 0.0, 1.0)
CAM_W, CAM_H = 8, 6


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Inputs:
    """Real publishers for the §12 inputs, driven at 20 Hz by a timer on `node`."""

    def __init__(self, node) -> None:
        self.node = node
        self.healthy = True
        self.cam = node.create_publisher(CameraInfo, "/camera/camera_info", 10)
        self.mask = node.create_publisher(Image, "/segmentation/mask", 10)
        self.degraded = node.create_publisher(Bool, "/ugv/perception_degraded", 10)
        self.pose = node.create_publisher(Bool, "/ugv/pose_valid", 10)
        self.loc_status = node.create_publisher(String, "/ugv/localization_status", 10)
        self.hb = node.create_publisher(Bool, "/ugv/nav2_heartbeat", 10)
        self.cmd = node.create_publisher(Twist, "/cmd_vel", 10)
        self.tf = TransformBroadcaster(node)
        node.create_timer(0.05, self.tick)

    def tick(self) -> None:
        if not self.healthy:
            return
        now = self.node.get_clock().now().to_msg()
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = now, "camera_optical_frame"
        info.width, info.height, info.k = CAM_W, CAM_H, list(K)
        self.cam.publish(info)
        mask = Image()
        mask.header.stamp, mask.header.frame_id, mask.encoding = now, "camera_optical_frame", "mono8"
        self.mask.publish(mask)
        self.degraded.publish(Bool(data=False))
        self.pose.publish(Bool(data=True))
        self.hb.publish(Bool(data=True))
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "map", "base_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            POSE["x"], POSE["y"], POSE["z"])
        t.transform.rotation.z, t.transform.rotation.w = math.sin(POSE["yaw"] / 2), math.cos(POSE["yaw"] / 2)
        mount = TransformStamped()
        mount.header.stamp, mount.header.frame_id, mount.child_frame_id = now, "base_link", "camera_optical_frame"
        mount.transform.translation.x, mount.transform.translation.y, mount.transform.translation.z = CAMERA_MOUNT
        q = mount.transform.rotation
        q.x, q.y, q.z, q.w = OPTICAL_TO_BODY_Q
        self.tf.sendTransform([t, mount])


class MapPubs:
    """Publishers for the map inputs, with the durability each real publisher uses: RTAB-Map's cloud, the
    elevation pair and the map stats are latched (transient local); the path, the depth image, the camera
    stream and the perception stats are volatile."""

    def __init__(self, node) -> None:
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        volatile = QoSProfile(depth=1)
        self.node = node
        self.cloud = node.create_publisher(PointCloud2, CLOUD, latched)
        self.path = node.create_publisher(Path, PATH, volatile)
        self.elev_cloud = node.create_publisher(PointCloud2, ELEV_CLOUD, latched)
        self.elev_grid = node.create_publisher(OccupancyGrid, ELEV_GRID, latched)
        self.depth = node.create_publisher(Image, DEPTH, volatile)
        self.camera = node.create_publisher(CompressedImage, CAMERA, volatile)
        self.map_stats = node.create_publisher(String, MAP_STATS, latched)
        self.perception_stats = node.create_publisher(String, PERCEPTION_STATS, QoSProfile(depth=10))

    def now(self):
        return self.node.get_clock().now().to_msg()


@pytest.fixture(scope="module")
def graph():
    os.environ.setdefault("ROS_DOMAIN_ID", "57")
    rclpy.init()
    store, goals, maps = StateStore(), GoalRegistry(), MapStore()
    gw = GatewayNode(store, goals, maps, parameter_overrides=[
        Parameter("map.idle_timeout_s", Parameter.Type.DOUBLE, IDLE_S),
        Parameter("map.stats_stale_s", Parameter.Type.DOUBLE, STATS_STALE_S),
    ])
    pub_node = rclpy.create_node("ugv_api_test_inputs")
    inputs = Inputs(pub_node)
    pubs = MapPubs(pub_node)
    seen: list[bool] = []
    pub_node.create_subscription(Bool, "/ugv/e_stop", lambda m: seen.append(bool(m.data)), 10)
    ex = MultiThreadedExecutor()
    ex.add_node(gw)
    ex.add_node(pub_node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    port = _free_port()
    app = create_app(gw, store, goals, telemetry_hz=10.0, maps=maps, **gw.map_cfg.app_kwargs())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serve = threading.Thread(target=server.run, daemon=True)
    serve.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}/api/v1", timeout=10.0)
    yield {"client": client, "gw": gw, "inputs": inputs, "pub_node": pub_node, "seen": seen, "pubs": pubs,
           "maps": maps}
    client.close()
    server.should_exit = True
    serve.join(timeout=5)
    ex.shutdown()
    gw.destroy_node()
    pub_node.destroy_node()
    rclpy.shutdown()


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    return pred()


def _safety(c):
    r = c.get("/safety/status")
    assert r.status_code == 200
    return r.json()


def test_health_and_openapi(graph):
    c = graph["client"]
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["rosNode"] == "ugv_api"
    spec = c.get("/openapi.json").json()
    assert "/api/v1/navigation/goals" in spec["paths"] and "/api/v1/safety/e-stop" in spec["paths"]


def test_all_watches_ok_with_live_inputs_and_arbiter_absent(graph):
    c = graph["client"]
    body = _wait(lambda: (lambda b: b if b["ok"] else None)(_safety(c)))
    assert body, _safety(c)
    assert [w["name"] for w in body["watches"]] == ["camera", "perception", "localization", "tf", "nav2", "e_stop"]
    assert body["arbiter"] == {"status": None, "ageS": None, "present": False}  # no Dev 5 arbiter yet


def test_heartbeat_that_stops_trips_watches(graph):
    c, inputs = graph["client"], graph["inputs"]
    inputs.healthy = False
    expected = {"camera", "perception", "localization", "tf", "nav2"}

    def tripped_names(body):
        return {w["name"] for w in body["watches"] if not w["ok"]}

    try:
        # Wait for all five, not for the first: the stamp-based watches (camera, perception, tf) trip a few
        # milliseconds before the receipt-based ones (localization, nav2), and a poll can land in between.
        tripped = _wait(lambda: (lambda b: b if expected <= tripped_names(b) else None)(_safety(c)), timeout=3.0)
        assert tripped, f"watches stayed ok after inputs stopped: tripped {tripped_names(_safety(c))}"
    finally:
        inputs.healthy = True
    assert _wait(lambda: _safety(c)["ok"])


def test_base_command_reads_final_cmd_vel(graph):
    c, inputs = graph["client"], graph["inputs"]
    assert c.get("/base/command").json()["available"] is False
    tw = Twist()
    tw.linear.x, tw.angular.z = 0.25, -0.5
    body = _wait(lambda: (inputs.cmd.publish(tw), (lambda b: b if b["available"] else None)(
        c.get("/base/command").json()))[1])
    assert body["linear"]["x"] == pytest.approx(0.25) and body["angular"]["z"] == pytest.approx(-0.5)


def test_estop_assert_republish_latch_and_release(graph):
    c, seen, pub_node = graph["client"], graph["seen"], graph["pub_node"]
    seen.clear()
    r = c.put("/safety/e-stop", json={"asserted": True})
    assert r.status_code == 200 and r.json()["asserted"] is True and r.json()["assertedByGateway"] is True
    assert _wait(lambda: seen.count(True) >= 3, timeout=3.0), f"e-stop not re-published: {seen}"

    late: list[bool] = []
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = pub_node.create_subscription(Bool, "/ugv/e_stop", lambda m: late.append(bool(m.data)), latched)
    assert _wait(lambda: late and late[0] is True), "late subscriber did not get the latched e-stop"
    pub_node.destroy_subscription(sub)

    e = next(w for w in _safety(c)["watches"] if w["name"] == "e_stop")
    assert e["ok"] is False

    goal = c.post("/navigation/goals", json={"x": 1.0, "y": 0.0})
    assert goal.status_code == 409 and goal.headers["content-type"].startswith(PROBLEM)
    assert any(r.startswith("e_stop") for r in goal.json()["reasons"])

    seen.clear()
    r = c.put("/safety/e-stop", json={"asserted": False})
    assert r.status_code == 200 and r.json()["assertedByGateway"] is False
    assert _wait(lambda: False in seen)
    time.sleep(0.5)
    assert True not in seen[seen.index(False):], "e-stop re-published after release"
    assert _wait(lambda: _safety(c)["ok"])


def test_goal_without_nav2_is_503_problem(graph):
    c = graph["client"]
    assert _wait(lambda: _safety(c)["ok"])
    r = c.post("/navigation/goals", json={"x": 3.0, "y": 0.0, "yaw": 0.0, "frameId": "map"})
    assert r.status_code == 503 and r.headers["content-type"].startswith(PROBLEM)
    assert "navigate_to_pose" in r.json()["detail"]
    nav = c.get("/navigation").json()
    assert nav["actionServerReady"] is False and nav["heartbeat"] is True and nav["activeGoal"] is None


def test_mode_without_rtabmap_is_503_problem(graph):
    r = graph["client"].put("/localization/mode", json={"mode": "localize"})
    assert r.status_code == 503 and r.headers["content-type"].startswith(PROBLEM)
    assert graph["client"].get("/localization").json()["requestedMode"] is None


@pytest.mark.parametrize(
    "path,method,body",
    [
        ("/navigation/goals", "post", {"x": "north", "y": 0}),
        ("/navigation/goals", "post", {"x": 1, "y": 0, "frameId": "odom"}),  # map-frame goals only (§11)
        ("/navigation/goals", "post", {"x": 1, "y": 0, "speed": 9}),  # unknown fields rejected
        ("/localization/mode", "put", {"mode": "slam"}),
        ("/safety/e-stop", "put", {}),
    ],
)
def test_invalid_bodies_are_422_problems(graph, path, method, body):
    r = getattr(graph["client"], method)(path, json=body)
    assert r.status_code == 422 and r.headers["content-type"].startswith(PROBLEM)
    assert r.json()["errors"]


def test_unknown_goal_is_404_problem(graph):
    c = graph["client"]
    for r in (c.get("/navigation/goals/nope"), c.delete("/navigation/goals/nope")):
        assert r.status_code == 404 and r.headers["content-type"].startswith(PROBLEM)


def test_sse_stream_emits_each_resource(graph):
    c = graph["client"]
    got: dict[str, dict] = {}
    with c.stream("GET", "/telemetry/stream") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                got[event] = json.loads(line[6:])
            if len(got) == 6:
                break
    assert set(got) == {"safety", "command", "localization", "navigation", "map", "pose"}
    assert "watches" in got["safety"] and "heartbeat" in got["navigation"]


def test_gateway_never_publishes_cmd_vel(graph):
    gw = graph["gw"]
    topics = {name for name, _ in gw.get_publisher_names_and_types_by_node("ugv_api", "/")}
    assert "/ugv/e_stop" in topics
    assert not topics & {"/cmd_vel", "/cmd_vel_nav2"}, topics


# ------------------------------------------------------------------------------------------- map inputs
# The heavy subscriptions exist only while a client keeps calling GET /map; each test that needs them holds the
# heartbeat with `watching`. Volatile topics are published repeatedly until the HTTP side shows them.


@contextlib.contextmanager
def watching(client):
    """Call GET /map from a thread every 0.2 s: the map view's demand heartbeat."""
    stop = threading.Event()

    def beat() -> None:
        with httpx.Client(base_url=str(client.base_url), timeout=5.0) as own:
            while not stop.is_set():
                with contextlib.suppress(httpx.HTTPError):
                    own.get("/map")
                stop.wait(0.2)

    client.get("/map")  # the first touch lands before the thread is scheduled
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(2.0)


def _fetch(c, path, publish=None, timeout=10.0):
    """GET `path` until it answers 200, calling `publish()` before each try."""
    deadline = time.monotonic() + timeout
    r = None
    while time.monotonic() < deadline:
        if publish is not None:
            publish()
        r = c.get(path)
        if r.status_code == 200:
            return r
        time.sleep(0.15)
    pytest.fail(f"GET {path} never answered 200; last answer {r.status_code} {r.text[:200]!r}")


def _map_status(c):
    r = c.get("/map")
    assert r.status_code == 200
    return r.json()


def _subscribed(node, topics, *, at_least=1):
    return all(node.count_subscribers(t) >= at_least for t in topics)


def _pack(columns, offsets, point_step):
    n = len(columns[0])
    buf = np.zeros((n, point_step), dtype=np.uint8)
    for col, offset in zip(columns, offsets):
        buf[:, offset : offset + 4] = np.ascontiguousarray(col, dtype="<f4").view(np.uint8).reshape(n, 4)
    return buf.tobytes()


def _fields(*names):
    return [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(names)]


def make_rgb_cloud(xyz, rgb, stamp, frame="map"):
    """RTAB-Map's layout: x y z, 4 bytes padding, packed rgb at 16, 32-byte records."""
    word = ((rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2]).view(np.float32)
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp, frame
    msg.height, msg.width, msg.point_step, msg.is_dense = 1, len(xyz), 32, True
    msg.row_step = 32 * len(xyz)
    msg.fields = _fields("x", "y", "z") + [PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1)]
    msg.data = _pack([xyz[:, 0], xyz[:, 1], xyz[:, 2], word], [0, 4, 8, 16], 32)
    return msg


def make_path(rows, stamp):
    msg = Path()
    msg.header.stamp, msg.header.frame_id = stamp, "map"
    for x, y, z, qx, qy, qz, qw in ((float(v) for v in row) for row in rows):  # message fields take floats only
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        msg.poses.append(ps)
    return msg


def make_elevation_cloud(columns, stamp):
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp, "map"
    n = len(columns[0])
    msg.height, msg.width, msg.point_step, msg.is_dense, msg.row_step = 1, n, 20, True, 20 * n
    msg.fields = _fields("x", "y", "z", "confidence", "obstacle_h")
    msg.data = _pack(columns, [0, 4, 8, 12, 16], 20)
    return msg


def make_grid(cells, resolution, origin, stamp, yaw=0.0):
    msg = OccupancyGrid()
    msg.header.stamp, msg.header.frame_id = stamp, "map"
    msg.info.resolution, msg.info.height, msg.info.width = resolution, cells.shape[0], cells.shape[1]
    msg.info.origin.position.x, msg.info.origin.position.y = origin
    msg.info.origin.orientation.z, msg.info.origin.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    msg.data = cells.astype(np.int8).reshape(-1).tolist()
    return msg


def make_depth(frame, stamp, depth_m):
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp, frame
    msg.height, msg.width = depth_m.shape
    msg.encoding, msg.is_bigendian, msg.step = "32FC1", False, 4 * depth_m.shape[1]
    msg.data = depth_m.astype("<f4").tobytes()
    return msg


def test_cloud_and_trajectory_published_on_ros_come_out_of_http_decoded(graph):
    c, pubs = graph["client"], graph["pubs"]
    xyz = np.array([[0, 0, 0], [1, 2, 3], [-1.5, 0.5, 2], [4, -4, 0.25], [0.125, 0.25, 0.5]], dtype=np.float32)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 20, 30], [250, 128, 7]], dtype=np.uint8)
    pubs.cloud.publish(make_rgb_cloud(xyz, rgb, pubs.now()))  # latched: out before anyone watches
    rows = [(0, 0, 0, 0, 0, 0, 1), (3, 4, 0, 0, 0, 1, 0), (3, 4, 12, 0.5, 0.5, 0.5, 0.5)]

    with watching(c):
        cloud = codec.decode_cloud(_fetch(c, "/map/cloud").content)  # only a transient-local subscription gets it
        traj = codec.decode_trajectory(
            _fetch(c, "/map/trajectory", publish=lambda: pubs.path.publish(make_path(rows, pubs.now()))).content)
        status = _map_status(c)

    assert cloud["count"] == 5 and cloud["source_count"] == 5 and cloud["has_rgb"]
    assert np.array_equal(cloud["xyz"], xyz) and np.array_equal(cloud["rgb"], rgb)
    assert traj["count"] == 3 and traj["length_m"] == pytest.approx(17.0)
    assert traj["poses"].tolist() == [[float(v) for v in r] for r in rows]
    assert status["seq"]["cloud"] >= 1 and status["seq"]["trajectory"] >= 1
    assert status["stats"]["cloud_source_points"] == 5  # gateway statistic: points in the last cloud_map


def test_a_newer_cloud_replaces_the_older_one_and_never_changes_what_was_served(graph):
    c, pubs, node = graph["client"], graph["pubs"], graph["pub_node"]
    first = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    second = np.array([[9, 9, 9], [8, 8, 8], [7, 7, 7]], dtype=np.float32)
    grey = np.full((3, 3), 128, dtype=np.uint8)
    def with_points(n):
        r = c.get("/map/cloud")
        return r if r.status_code == 200 and codec.decode_cloud(r.content)["count"] == n else None

    with watching(c):
        assert _wait(lambda: _subscribed(node, [CLOUD]), timeout=8.0)  # a latched sample only reaches a live sub
        pubs.cloud.publish(make_rgb_cloud(first, grey[:2], pubs.now()))
        a = _wait(lambda: with_points(2))
        assert a is not None
        kept = a.content
        pubs.cloud.publish(make_rgb_cloud(second, grey, pubs.now()))
        b = _wait(lambda: with_points(3))
    assert b is not None and np.array_equal(codec.decode_cloud(b.content)["xyz"], second)
    assert np.array_equal(codec.decode_cloud(kept)["xyz"], first)  # the earlier body is untouched


def test_heavy_subscriptions_exist_only_while_a_client_is_watching(graph):
    c, node = graph["client"], graph["pub_node"]
    always_on = (MAP_STATS, PERCEPTION_STATS)
    assert _wait(lambda: _subscribed(node, always_on), timeout=3.0), "the two stats subscriptions are always on"

    with watching(c):
        assert _wait(lambda: _subscribed(node, HEAVY_TOPICS), timeout=8.0), [
            (t, node.count_subscribers(t)) for t in HEAVY_TOPICS]
    # no heartbeat for longer than IDLE_S: the demand timer destroys them (it ticks once a second)
    assert _wait(lambda: all(node.count_subscribers(t) == 0 for t in HEAVY_TOPICS), timeout=IDLE_S + 6.0), [
        (t, node.count_subscribers(t)) for t in HEAVY_TOPICS]
    assert _subscribed(node, always_on), "the stats subscriptions stay"

    with watching(c):  # and they come back for the next viewer
        assert _wait(lambda: _subscribed(node, HEAVY_TOPICS), timeout=8.0)


def test_pose_resource_and_event_report_the_published_transform(graph):
    c = graph["client"]

    def available():
        body = c.get("/map/pose").json()
        return body if body["available"] else None

    body = _wait(available)
    assert body, "no pose"
    expected = {"x": POSE["x"], "y": POSE["y"], "z": POSE["z"], "qx": 0.0, "qy": 0.0,
                "qz": math.sin(POSE["yaw"] / 2), "qw": math.cos(POSE["yaw"] / 2)}
    assert body["available"] is True
    for key, want in expected.items():
        assert body[key] == pytest.approx(want, abs=1e-6), key
    assert -0.5 < body["ageS"] < 2.0

    got = None
    with c.stream("GET", "/telemetry/stream") as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "pose":
                got = json.loads(line[6:])
                if got["available"]:
                    break
    assert got and got["available"] and got["x"] == pytest.approx(POSE["x"])
    assert got["qz"] == pytest.approx(math.sin(POSE["yaw"] / 2))


def test_elevation_is_built_only_from_a_cloud_and_a_grid_with_equal_stamps(graph):
    c, pubs, node = graph["client"], graph["pubs"], graph["pub_node"]
    cols = [np.array(v, dtype=np.float32) for v in (
        [-1.75, -1.25, -0.25],  # x: cell columns 0 1 3 of the grid below
        [1.75, 2.25, 3.25],  # y: rows 0 1 3
        [0.5, 1.0, 1.5],  # z
        [1.0, 0.5, 0.25],  # confidence
        [0.0, 0.15, 0.0],  # obstacle_h
    )]
    s1 = pubs.now()
    time.sleep(0.01)
    s2 = pubs.now()  # a later stamp: same publishers, different sample
    assert (s1.sec, s1.nanosec) != (s2.sec, s2.nanosec)
    cells = np.zeros((6, 8), dtype=np.int8)
    with watching(c):
        assert _wait(lambda: _subscribed(node, HEAVY_TOPICS), timeout=8.0)
        before = _map_status(c)["seq"]["elevation"]
        pubs.elev_cloud.publish(make_elevation_cloud(cols, s1))
        pubs.elev_grid.publish(make_grid(cells, 0.5, (-2.0, 1.5), s2))  # stamps differ: no pair
        time.sleep(0.8)
        assert _map_status(c)["seq"]["elevation"] == before, "an unequal pair was used"
        pubs.elev_grid.publish(make_grid(cells, 0.5, (-2.0, 1.5), s1))  # now they match
        assert _wait(lambda: _map_status(c)["seq"]["elevation"] == before + 1)
        out = codec.decode_elevation(_fetch(c, "/map/elevation").content)
        stats = _map_status(c)["stats"]
    assert out["known_cells"] == 3 and out["resolution_m"] == pytest.approx(0.5)
    assert (out["origin_x"], out["origin_y"]) == pytest.approx((-2.0, 1.5))
    assert out["heights"][0, 0] == pytest.approx(0.5) and out["heights"][1, 1] == pytest.approx(1.0)
    assert out["heights"][3, 3] == pytest.approx(1.5)
    assert out["obstacle"][1, 1] == 3  # 0.15 m in 5 cm units
    assert stats["elevation_known_cells"] == 3


def test_a_latched_costmap_is_received_when_its_publisher_appears_after_the_subscription(graph):
    c, node = graph["client"], graph["pub_node"]
    cells = np.array([[-1, 0, 10], [20, 100, 50]], dtype=np.int8)
    with watching(c):
        # no publisher yet: the gateway subscribes volatile and has to notice the latched publisher later
        assert _wait(lambda: node.count_subscribers(GRID) >= 1, timeout=8.0)
        assert not node.get_publishers_info_by_topic(GRID)
        # The publisher gets a DDS participant of its own (a new rclpy context): a writer added to a participant
        # the gateway already knows is matched at once, before its first sample, and then the reader's durability
        # would not matter. A new participant is discovered only after the single latched sample was written, so
        # only a transient-local subscription receives it.
        context = Context()
        rclpy.init(context=context)
        other = rclpy.create_node("ugv_api_test_costmap", context=context)
        try:
            pub = other.create_publisher(OccupancyGrid, GRID,
                                         QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            pub.publish(make_grid(cells, 0.25, (-0.5, 1.0), other.get_clock().now().to_msg(), yaw=0.5))  # once
            out = codec.decode_grid(_fetch(c, "/map/grid", timeout=12.0).content)
        finally:
            other.destroy_node()
            context.try_shutdown()
    assert out["cells"].tolist() == cells.tolist()
    assert out["resolution_m"] == pytest.approx(0.25) and (out["origin_x"], out["origin_y"]) == pytest.approx((-0.5, 1.0))
    assert out["origin_yaw"] == pytest.approx(0.5, abs=1e-6)


def _expected_live_points():
    """Pixels (0,0) (4,0) (0,4) (4,4) of an 8 x 6 image at 2 m, through the optical -> base -> map chain, with
    plain numpy and the numbers published above."""
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    z = 2.0
    optical = np.array([[(u - cx) * z / fx, (v - cy) * z / fy, z] for v in (0, 4) for u in (0, 4)])
    r_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # body = (z, -x, -y)
    yaw = POSE["yaw"]
    r_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    in_base = optical @ r_opt.T + np.array(CAMERA_MOUNT)
    return in_base @ r_yaw.T + np.array([POSE["x"], POSE["y"], POSE["z"]])


def test_depth_and_live_layers_come_from_the_same_image(graph):
    c, pubs = graph["client"], graph["pubs"]
    depth = np.full((CAM_H, CAM_W), 2.0, dtype=np.float32)
    depth[0, 6] = np.nan  # a hole on a pixel the depth stride samples (column 6) and the live stride does not

    def publish():
        pubs.depth.publish(make_depth("camera_optical_frame", pubs.now(), depth))

    with watching(c):
        d = codec.decode_depth(_fetch(c, "/map/depth", publish=publish).content)
        live = codec.decode_cloud(_fetch(c, "/map/live", publish=publish).content)
    assert (d["width"], d["height"]) == (4, 3)  # depth_stride 2
    assert d["counts"][0, 0] == 2000 and d["counts"][0, 3] == 0  # 2 m in mm; the hole at (row 0, col 6) is 0
    assert live["count"] == 4 and not live["has_rgb"]  # live_stride 4: columns 0 4, rows 0 4
    assert np.allclose(live["xyz"], _expected_live_points(), atol=1e-4)


def test_live_is_skipped_without_a_transform_but_depth_still_updates(graph):
    c, pubs, gw = graph["client"], graph["pubs"], graph["gw"]
    depth = np.full((CAM_H, CAM_W), 2.0, dtype=np.float32)
    with watching(c):
        before = _map_status(c)["seq"]
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and _map_status(c)["seq"]["depth"] < before["depth"] + 3:
            pubs.depth.publish(make_depth("no_such_frame", pubs.now(), depth))
            time.sleep(0.1)
        after = _map_status(c)["seq"]
    assert after["depth"] >= before["depth"] + 3
    assert after["live"] == before["live"], "camera-frame points were published as map-frame points"
    assert gw.map_rejects.get("live_no_transform", 0) >= 1


def test_camera_jpeg_is_served_as_received_and_a_non_jpeg_is_refused(graph):
    c, pubs, gw = graph["client"], graph["pubs"], graph["gw"]
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF-pretend-picture\xff\xd9"

    def frame(data):
        msg = CompressedImage()
        msg.header.stamp, msg.header.frame_id, msg.format, msg.data = pubs.now(), "camera", "jpeg", data
        return msg

    with watching(c):
        r = _fetch(c, "/map/camera", publish=lambda: pubs.camera.publish(frame(jpeg)))
        time.sleep(0.3)  # let a JPEG still in flight land before the seq is read
        before = _map_status(c)["seq"]["camera"]
        for _ in range(5):
            pubs.camera.publish(frame(b"\x89PNG\r\n\x1a\n-not-a-jpeg"))
            time.sleep(0.1)
        assert _wait(lambda: gw.map_rejects.get("camera", 0) >= 1)
        after = _map_status(c)["seq"]["camera"]
    assert r.headers["content-type"] == "image/jpeg" and r.content == jpeg
    assert after == before  # the PNG never replaced the JPEG


def test_stats_from_both_sources_merge_malformed_json_is_ignored_and_silence_expires(graph):
    c, pubs, gw = graph["client"], graph["pubs"], graph["gw"]

    def seen(*keys):
        stats = _map_status(c)["stats"]
        return stats if all(k in stats for k in keys) else None

    def publish_good():
        pubs.map_stats.publish(String(data=json.dumps({"keyframes": 12, "mode": "mapping",
                                                       "calibration_placeholder": False})))
        pubs.perception_stats.publish(String(data=json.dumps({"depth_hz": 3.2, "stage_ms": {"seg": 40}})))

    deadline = time.monotonic() + 8.0
    stats = None
    while stats is None and time.monotonic() < deadline:
        publish_good()
        stats = _wait(lambda: seen("keyframes", "depth_hz"), timeout=0.3)
    assert stats, _map_status(c)
    assert stats["keyframes"] == 12 and stats["mode"] == "mapping" and stats["calibration_placeholder"] is False
    assert stats["depth_hz"] == 3.2
    assert "stage_ms" not in stats  # nested values are not part of the flat contract

    publish_good()
    before = gw.map_rejects.get("map_stats", 0)
    pubs.map_stats.publish(String(data="{not json"))
    assert _wait(lambda: gw.map_rejects.get("map_stats", 0) == before + 1)
    assert seen("keyframes"), "malformed JSON must not clear what the source reported"

    # silence: nothing more is published; both sources are dropped after STATS_STALE_S
    assert _wait(lambda: "keyframes" not in _map_status(c)["stats"] and "depth_hz" not in _map_status(c)["stats"],
                 timeout=STATS_STALE_S + 6.0)
    # only the gateway's own statistics (they describe retained layers) outlive the silence
    assert set(_map_status(c)["stats"]) <= {"cloud_source_points", "elevation_known_cells"}
