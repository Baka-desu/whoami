"""The arbiter node over real ROS 2 topics. Skipped without ROS 2 (runs under colcon test in Lyrical).

An in-process probe plays Dev 1 / Dev 2 / Dev 4 / the camera / the operator with synthetic stimuli
(test input only) and watches the one thing that reaches the base: /cmd_vel.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from sensor_msgs.msg import CameraInfo  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402

CONFIG = Path(__file__).resolve().parents[2] / "config" / "safety" / "safety_timeouts.yaml"


class Rig:
    def __init__(self) -> None:
        os.environ["ROS_DOMAIN_ID"] = str(40 + os.getpid() % 50)
        rclpy.init(args=["--ros-args", "-p", f"config_path:={CONFIG}"])
        from ugv_safety.nodes.safety_arbiter import SafetyArbiterNode

        self.arb = SafetyArbiterNode()
        self.probe = rclpy.create_node("safety_probe")
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.arb)
        self.ex.add_node(self.probe)
        self.cmds: list[Twist] = []
        self.status: list[str] = []
        self.probe.create_subscription(Twist, "/cmd_vel", self.cmds.append, 10)
        self.probe.create_subscription(String, "/ugv/safety_status", lambda m: self.status.append(m.data), 10)
        self.pub = {
            "cand": self.probe.create_publisher(Twist, "/cmd_vel_nav2", 10),
            "estop": self.probe.create_publisher(Bool, "/ugv/e_stop", 10),
            "pose": self.probe.create_publisher(Bool, "/ugv/pose_valid", 10),
            "perc": self.probe.create_publisher(Bool, "/ugv/perception_degraded", 10),
            "nav2": self.probe.create_publisher(Bool, "/ugv/nav2_heartbeat", 10),
            "cam": self.probe.create_publisher(CameraInfo, "/camera/camera_info", 10),
        }

    def run(self, seconds: float, *, silent: tuple[str, ...] = (), pose: bool = True, estop: bool | None = None) -> None:
        end = time.monotonic() + seconds
        next_pub = 0.0
        while time.monotonic() < end:
            now = time.monotonic()
            if now >= next_pub:
                next_pub = now + 0.05
                if "cand" not in silent:
                    t = Twist()
                    t.linear.x = 0.3
                    self.pub["cand"].publish(t)
                if "pose" not in silent:
                    self.pub["pose"].publish(Bool(data=pose))
                if "perc" not in silent:
                    self.pub["perc"].publish(Bool(data=False))
                if "nav2" not in silent:
                    self.pub["nav2"].publish(Bool(data=True))
                if "cam" not in silent:
                    ci = CameraInfo()
                    ci.header.stamp = self.probe.get_clock().now().to_msg()
                    self.pub["cam"].publish(ci)
                if estop is not None:
                    self.pub["estop"].publish(Bool(data=estop))
            self.ex.spin_once(timeout_sec=0.01)

    def last(self) -> Twist:
        return self.cmds[-1]

    def close(self) -> None:
        self.ex.shutdown()
        self.arb.destroy_node()
        self.probe.destroy_node()
        rclpy.shutdown()


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.close()


def test_safe_candidate_reaches_the_base(rig):
    rig.run(1.5)
    assert rig.last().linear.x == pytest.approx(0.3)
    assert rig.status[-1] == "L4 FORWARD"


def test_arbiter_is_the_only_cmd_vel_publisher(rig):
    rig.run(0.5)
    assert rig.probe.count_publishers("/cmd_vel") == 1


def test_estop_zeroes_the_base_and_stays_latched(rig):
    rig.run(1.0)
    rig.run(0.5, estop=True)
    assert rig.last().linear.x == 0.0 and rig.status[-1].startswith("L1 ESTOP")
    rig.run(1.0)  # operator stops publishing: still latched
    assert rig.last().linear.x == 0.0
    rig.run(1.0, estop=False)
    assert rig.last().linear.x == pytest.approx(0.3)


def test_invalid_pose_holds(rig):
    rig.run(1.0)
    rig.run(1.5, pose=False)
    assert rig.last().linear.x == 0.0 and "pose_invalid" in rig.status[-1]


def test_silent_localization_is_a_health_fault(rig):
    rig.run(1.0)
    rig.run(1.5, silent=("pose",))
    assert rig.last().linear.x == 0.0 and "localization_stale" in rig.status[-1]


def test_nothing_published_means_a_zero_base_command(rig):
    rig.run(1.0, silent=("cand", "pose", "perc", "nav2", "cam"))
    assert rig.cmds and all(c.linear.x == 0.0 and c.angular.z == 0.0 for c in rig.cmds)
