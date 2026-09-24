"""rclpy safety arbiter node: sole authoritative publisher to base /cmd_vel.

Wires ugv_safety.arbiter.decide + ugv_safety.watchdog.monitor + ugv_safety.decel.ramp
into ROS topics per architecture.md §3.1 and §12. Contains no arbitration logic
itself -- that lives in the pure kernels so it is testable without rclpy.
"""

from __future__ import annotations

import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from ugv_safety.arbiter.decide import decide
from ugv_safety.arbiter.types import ArbiterInputs, Twist2D
from ugv_safety.decel.ramp import RampLimits, TwistRamp
from ugv_safety.watchdog.monitor import WatchdogMonitor
from ugv_safety.watchdog.table import load_watchdog_profile

_ROOT = Path(__file__).resolve().parents[4]
_NS = 1_000_000_000


def _latest_only() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _latched() -> QoSProfile:
    """Matches ugv_eval.estop_cli's publisher so a late-joining arbiter still
    sees the current /ugv/e_stop state instead of defaulting to False."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class SafetyArbiterNode(Node):
    def __init__(self, *, timeouts_path: Path | None = None, now_ns_fn: object | None = None) -> None:
        super().__init__("ugv_safety_arbiter")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("max_linear_accel", 0.6)
        self.declare_parameter("max_linear_decel", 1.5)
        self.declare_parameter("max_angular_accel", 1.2)
        self.declare_parameter("max_angular_decel", 2.5)

        timeouts_path = timeouts_path or _ROOT / "config" / "safety" / "safety_timeouts.yaml"
        self._watchdog = WatchdogMonitor(load_watchdog_profile(timeouts_path))
        self._now_ns_fn = now_ns_fn or time.time_ns

        limits = RampLimits(
            max_linear_accel=float(self.get_parameter("max_linear_accel").value),
            max_linear_decel=float(self.get_parameter("max_linear_decel").value),
            max_angular_accel=float(self.get_parameter("max_angular_accel").value),
            max_angular_decel=float(self.get_parameter("max_angular_decel").value),
        )
        self._ramp = TwistRamp(limits)

        # Fail closed: hold until every upstream node has reported in at least once.
        self._estop = False
        self._perception_degraded = True
        self._pose_valid = False
        self._candidate = Twist2D(0.0, 0.0)
        self._last_tick_ns: int | None = None

        qos = _latest_only()
        self.create_subscription(Bool, "/ugv/e_stop", self._on_estop, _latched())
        self.create_subscription(Bool, "/ugv/perception_degraded", self._on_degraded, qos)
        self.create_subscription(Bool, "/ugv/pose_valid", self._on_pose_valid, qos)
        self.create_subscription(Twist, "/cmd_vel_nav2", self._on_candidate, qos)
        self.create_subscription(Image, "/camera/image_raw", self._on_camera, qos)

        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self._pub_status = self.create_publisher(String, "/ugv/safety_status", 10)

        rate_hz = float(self.get_parameter("rate_hz").value)
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

    def _on_estop(self, msg: Bool) -> None:
        self._estop = bool(msg.data)

    def _on_degraded(self, msg: Bool) -> None:
        self._perception_degraded = bool(msg.data)
        self._watchdog.touch("perception_mask", self._now_ns_fn())

    def _on_pose_valid(self, msg: Bool) -> None:
        self._pose_valid = bool(msg.data)
        self._watchdog.touch("pose_tf", self._now_ns_fn())

    def _on_candidate(self, msg: Twist) -> None:
        self._candidate = Twist2D(float(msg.linear.x), float(msg.angular.z))
        self._watchdog.touch("nav2_heartbeat", self._now_ns_fn())

    def _on_camera(self, _msg: Image) -> None:
        self._watchdog.touch("camera", self._now_ns_fn())

    def _tick(self) -> None:
        now_ns = self._now_ns_fn()
        wd = self._watchdog.evaluate(now_ns)
        inputs = ArbiterInputs(
            estop=self._estop,
            watchdog_tripped=wd.tripped,
            watchdog_reasons=wd.reasons,
            perception_degraded=self._perception_degraded,
            pose_valid=self._pose_valid,
            candidate=self._candidate,
        )
        decision = decide(inputs)

        dt_s = (now_ns - self._last_tick_ns) / _NS if self._last_tick_ns is not None else 0.0
        self._last_tick_ns = now_ns
        target = decision.output
        published = self._ramp.step(target, dt_s if dt_s > 0 else 1e-3)

        twist_msg = Twist()
        twist_msg.linear.x = published.linear_x
        twist_msg.angular.z = published.angular_z
        self._pub_cmd.publish(twist_msg)

        status = String()
        status.data = f"level={decision.level} hold={decision.hold} reason={decision.reason}"
        self._pub_status.publish(status)


def main() -> None:
    rclpy.init()
    node = SafetyArbiterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
