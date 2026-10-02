"""wheel/odom + odom_visual → one gated, continuous odom->base_link TF + /odom (mindmap D5b, D6).

Subscribes : wheel/odom   nav_msgs/Odometry  Dev 5 wheel odometry (no TF from Dev 5)
             odom_visual  nav_msgs/Odometry  rtabmap_odom/rgbd_odometry (publish_tf:=false)
Publishes  : odom                          nav_msgs/Odometry (selected source; source stamp)
             TF odom->base_link            stamp = source odometry stamp, never "now"
             /ugv/localization/odom_source std_msgs/String  active source, latched, on change
Params     : profile_path (config/odom_select.yaml), odom_source (auto | wheel | visual)
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from ugv_localization.odom import OdomGate, OdomSelector, Source, load_odom_select_config, parse_odom_source
from ugv_localization.rosconv import odom_msg_to_sample


class OdomSelectorNode(Node):
    def __init__(self) -> None:
        super().__init__("odom_selector")
        profile_path = self.declare_parameter("profile_path", "").value
        if not profile_path:
            raise RuntimeError("profile_path parameter is required")
        policy = parse_odom_source(self.declare_parameter("odom_source", "auto").value)
        overlay_path = self.declare_parameter("profile_overlay_path", "").value  # e.g. odom_select_laptop.yaml
        cfg = load_odom_select_config(profile_path, overlay_path or None)
        self._odom_frame = cfg.wheel_gate.odom_frame
        self._base_frame = cfg.wheel_gate.base_frame
        self._gates = {Source.WHEEL: OdomGate(cfg.wheel_gate), Source.VISUAL: OdomGate(cfg.visual_gate)}
        self._selector = OdomSelector(cfg.selector, policy)
        self._tf = TransformBroadcaster(self)
        self._pub = self.create_publisher(Odometry, "odom", 20)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub_source = self.create_publisher(String, "/ugv/localization/odom_source", latched)
        self.create_subscription(Odometry, "wheel/odom", lambda m: self._on_odom(Source.WHEEL, m), 20)
        self.create_subscription(Odometry, "odom_visual", lambda m: self._on_odom(Source.VISUAL, m), 20)
        self._last_now_ns: int | None = None
        self.get_logger().info(f"odom selector: odom_source={policy.value}")

    def _on_odom(self, source: Source, msg: Odometry) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns <= 0:
            return  # use_sim_time and /clock not received yet
        if self._last_now_ns is not None and now_ns < self._last_now_ns:
            self.get_logger().warning("clock jumped backwards (bag loop / sim reset): resetting odom selector")
            for gate in self._gates.values():
                gate.reset()
            self._selector.reset()
        self._last_now_ns = now_ns

        try:
            sample = odom_msg_to_sample(msg)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"dropping {source.value} odom: {exc}", throttle_duration_sec=2.0)
            return
        gated = self._gates[source].accept(sample, now_ns)
        if gated.edge is None:
            self.get_logger().warning(
                f"dropping {source.value} odom: {gated.reason.value if gated.reason else 'unknown'}",
                throttle_duration_sec=2.0,
            )
            return

        res = self._selector.on_sample(source, gated.edge, sample.pose_covariance, now_ns)
        if res.event is not None:
            prev = res.event.previous.value if res.event.previous else "none"
            self.get_logger().warning(
                f"odom source {prev} -> {res.event.current.value} ({res.event.reason})"
            )
            self._pub_source.publish(String(data=res.event.current.value))
        if res.dropped == "visual_lost":
            self.get_logger().warning("visual odometry lost (rgbd_odometry)", throttle_duration_sec=2.0)
        out = res.output
        if out is None:
            return

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self._odom_frame
        t.child_frame_id = self._base_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = out.translation
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = out.rotation
        self._tf.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        p.x, p.y, p.z = out.translation
        q.x, q.y, q.z, q.w = out.rotation
        odom.pose.covariance = list(out.pose_covariance)
        odom.twist = msg.twist  # body-frame twist: unaffected by the odom-frame re-anchor
        self._pub.publish(odom)


def main(args: list[str] | None = None) -> None:
    # Context manager owns shutdown; calling destroy_node() after SIGINT raises on Lyrical.
    try:
        with rclpy.init(args=args):
            rclpy.spin(OdomSelectorNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
