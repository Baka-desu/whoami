"""Safety arbiter node: the only publisher of the base /cmd_vel (architecture.md §3.1).

Subscribes : /cmd_vel_nav2             geometry_msgs/Twist  Nav2 candidate (Dev 4)
             /ugv/e_stop               std_msgs/Bool        operator kill, latched until released (volatile + transient-local)
             /ugv/pose_valid           std_msgs/Bool        Dev 2, 20 Hz heartbeat
             /ugv/perception_degraded  std_msgs/Bool        Dev 1, every processed frame
             /ugv/nav2_heartbeat       std_msgs/Bool        Dev 4, 20 Hz heartbeat
             /camera/camera_info       sensor_msgs/CameraInfo  one per image (camera liveness)
Publishes  : /cmd_vel                  geometry_msgs/Twist  every tick, whatever the inputs do
             /ugv/safety_status        std_msgs/String      `L<level> <NAME>[: reasons]`, on change
Params     : config_path (required, see config/safety/safety_timeouts.yaml)
             estop_state_path (default ~/.ros/ugv/estop_latched; empty disables): the e-stop survives a restart
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, String

from ugv_safety.arbiter_core import SafetyArbiter, load_config
from ugv_safety.estop_store import EstopLatchStore

_NS = 1_000_000_000


class SafetyArbiterNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_arbiter")
        path = self.declare_parameter("config_path", "").value
        if not path:
            raise RuntimeError("config_path parameter is required")
        cfg = load_config(path)
        self._arb = SafetyArbiter(cfg)

        # Start from the e-stop state we last held, not from "released": a latched message dies with its
        # publisher (the UI disconnecting), so after a restart nothing on the bus remembers the kill.
        state_path = self.declare_parameter("estop_state_path", "~/.ros/ugv/estop_latched").value
        self._store = EstopLatchStore(state_path) if state_path else None
        if self._store is not None:
            asserted, note = self._store.load()
            if asserted:
                self._arb.on_estop(True, self._now_ns())
            if note:
                self.get_logger().error(note) if asserted else self.get_logger().warning(note)

        reliable = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # A latched CameraInfo must not read as a fresh frame to a late joiner, so volatile here;
        # compatible with the driver's reliable + transient-local publisher.
        self.create_subscription(Twist, "/cmd_vel_nav2", self._on_candidate, reliable)
        # The kill switch is subscribed twice on purpose. A volatile subscription hears every kind of
        # publisher (the `ros2 topic pub` CLI, rosbridge) and may assert or release. A transient-local one
        # additionally receives the last latched value from a still-living latching publisher, so an
        # arbiter that starts late sees an e-stop asserted before it existed. That replayed history may
        # only ASSERT: it can be older than the persisted state, and must never release a newer e-stop.
        self.create_subscription(Bool, "/ugv/e_stop", self._on_estop, reliable)
        self.create_subscription(
            Bool, "/ugv/e_stop", self._on_estop_replay,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(Bool, "/ugv/pose_valid", self._on_pose, reliable)
        self.create_subscription(Bool, "/ugv/perception_degraded", self._on_perception, reliable)
        self.create_subscription(Bool, "/ugv/nav2_heartbeat", self._on_nav2, reliable)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_camera, reliable)

        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self._pub_status = self.create_publisher(
            String, "/ugv/safety_status",
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._last_status: str | None = None
        self.create_timer(1.0 / cfg.publish_rate_hz, self._tick)
        self.get_logger().info(f"safety arbiter up, config {path}; holding until every watched source has spoken")

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_candidate(self, m: Twist) -> None:
        self._arb.on_candidate(m.linear.x, m.angular.z, self._now_ns())

    def _set_estop(self, asserted: bool) -> None:
        changed = asserted != self._arb.estop_asserted
        self._arb.on_estop(asserted, self._now_ns())
        if changed and self._store is not None and not self._store.save(asserted):
            self.get_logger().error(
                f"cannot persist the e-stop state to {self._store.path}; it will not survive a restart"
            )

    def _on_estop(self, m: Bool) -> None:
        self._set_estop(m.data)

    def _on_estop_replay(self, m: Bool) -> None:
        if m.data:
            self._set_estop(True)

    def _on_pose(self, m: Bool) -> None:
        self._arb.on_pose_valid(m.data, self._now_ns())

    def _on_perception(self, m: Bool) -> None:
        self._arb.on_perception_degraded(m.data, self._now_ns())

    def _on_nav2(self, m: Bool) -> None:
        self._arb.on_nav2_heartbeat(m.data, self._now_ns())

    def _on_camera(self, m: CameraInfo) -> None:
        self._arb.on_camera(m.header.stamp.sec * _NS + m.header.stamp.nanosec, self._now_ns())

    def _tick(self) -> None:
        d = self._arb.step(self._now_ns())
        out = Twist()
        out.linear.x = d.linear
        out.angular.z = d.angular
        self._pub_cmd.publish(out)
        if d.status != self._last_status:
            self._last_status = d.status
            self._pub_status.publish(String(data=d.status))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SafetyArbiterNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
