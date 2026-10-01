"""Gateway on a real ROS graph over real HTTP (uvicorn on a free port, httpx client).

Real rclpy nodes publish the contract topics; no Nav2 or RTAB-Map is running, so goals and mode
switches must answer 503 problems, and a tripped §12 watch must answer 409. Runs under colcon test
(or any shell with ROS sourced); skipped in plain Python.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from geometry_msgs.msg import TransformStamped, Twist  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402

from ugv_api.app import create_app  # noqa: E402
from ugv_api.goals import GoalRegistry  # noqa: E402
from ugv_api.ros_node import GatewayNode  # noqa: E402
from ugv_api.state import StateStore  # noqa: E402

PROBLEM = "application/problem+json"


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
        self.cam.publish(info)
        mask = Image()
        mask.header.stamp, mask.header.frame_id, mask.encoding = now, "camera_optical_frame", "mono8"
        self.mask.publish(mask)
        self.degraded.publish(Bool(data=False))
        self.pose.publish(Bool(data=True))
        self.hb.publish(Bool(data=True))
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "map", "base_link"
        t.transform.rotation.w = 1.0
        self.tf.sendTransform(t)


@pytest.fixture(scope="module")
def graph():
    os.environ.setdefault("ROS_DOMAIN_ID", "57")
    rclpy.init()
    store, goals = StateStore(), GoalRegistry()
    gw = GatewayNode(store, goals)
    pub_node = rclpy.create_node("ugv_api_test_inputs")
    inputs = Inputs(pub_node)
    seen: list[bool] = []
    pub_node.create_subscription(Bool, "/ugv/e_stop", lambda m: seen.append(bool(m.data)), 10)
    ex = MultiThreadedExecutor()
    ex.add_node(gw)
    ex.add_node(pub_node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(gw, store, goals, telemetry_hz=10.0), host="127.0.0.1",
                                           port=port, log_level="warning"))
    serve = threading.Thread(target=server.run, daemon=True)
    serve.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}/api/v1", timeout=10.0)
    yield {"client": client, "gw": gw, "inputs": inputs, "pub_node": pub_node, "seen": seen}
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
    try:
        tripped = _wait(lambda: (lambda b: b if not b["ok"] else None)(_safety(c)), timeout=3.0)
        assert tripped, "watches stayed ok after inputs stopped"
        names = {w["name"] for w in tripped["watches"] if not w["ok"]}
        assert {"camera", "perception", "localization", "tf", "nav2"} <= names
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
            if len(got) == 4:
                break
    assert set(got) == {"safety", "command", "localization", "navigation"}
    assert "watches" in got["safety"] and "heartbeat" in got["navigation"]


def test_gateway_never_publishes_cmd_vel(graph):
    gw = graph["gw"]
    topics = {name for name, _ in gw.get_publisher_names_and_types_by_node("ugv_api", "/")}
    assert "/ugv/e_stop" in topics
    assert not topics & {"/cmd_vel", "/cmd_vel_nav2"}, topics
