"""Distance travelled along /odom — an odometry ESTIMATE (visual odometry when there is no wheel sensor).

Subscribes : odom                            nav_msgs/Odometry  selected odom from odom_selector
             /ugv/localization/odom_source   std_msgs/String    latched; attributes each metre
Publishes  : /ugv/localization/distance_travelled  std_msgs/Float64  metres since start / reset, every tick
             /ugv/localization/distance_basis      std_msgs/String   latched, on change: none |
                 visual_odometry_estimate | wheel_odometry | odometry_estimate (mixed / unattributed)
Services   : /ugv/localization/reset_distance      std_srvs/Empty    zero the total (e.g. per mission)
Params     : profile_path (config/distance.yaml)
"""

from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64, String
from std_srvs.srv import Empty

from ugv_localization.odom import DistanceTracker, Source, load_distance_profile
from ugv_localization.rosconv import stamp_to_ns


class DistanceTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("distance_tracker")
        profile_path = self.declare_parameter("profile_path", "").value
        if not profile_path:
            raise RuntimeError("profile_path parameter is required")
        profile = load_distance_profile(profile_path)
        self._tracker = DistanceTracker(profile)
        self._last_now_ns: int | None = None
        self._last_basis: str | None = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub_total = self.create_publisher(Float64, "/ugv/localization/distance_travelled", 10)
        self._pub_basis = self.create_publisher(String, "/ugv/localization/distance_basis", latched)
        self.create_subscription(Odometry, "odom", self._on_odom, 20)
        self.create_subscription(String, "/ugv/localization/odom_source", self._on_odom_source, latched)
        self.create_service(Empty, "/ugv/localization/reset_distance", self._on_reset)
        self.create_timer(1.0 / profile.publish_rate_hz, self._tick)
        self.get_logger().info("distance tracker: publishing /ugv/localization/distance_travelled (odometry estimate)")

    def _on_odom(self, msg: Odometry) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns <= 0:
            return  # use_sim_time and /clock not received yet
        if self._last_now_ns is not None and now_ns < self._last_now_ns:
            self.get_logger().warning("clock jumped backwards (bag loop / sim reset): distance reset to 0")
            self._tracker.reset()
        self._last_now_ns = now_ns
        try:
            ns = stamp_to_ns(msg.header.stamp)
        except (TypeError, ValueError):
            return
        p = msg.pose.pose.position
        jumps = self._tracker.jumps_rejected
        self._tracker.on_odom(ns, float(p.x), float(p.y))
        if self._tracker.jumps_rejected != jumps:
            self.get_logger().warning(
                "odom jump faster than max_speed_mps: not counted as distance", throttle_duration_sec=2.0
            )

    def _on_odom_source(self, msg: String) -> None:
        try:
            self._tracker.on_source(Source(msg.data.strip().lower()))
        except ValueError:
            self.get_logger().warning(f"unknown odom source {msg.data!r}: keeping the previous one")

    def _on_reset(self, request: Empty.Request, response: Empty.Response) -> Empty.Response:
        self.get_logger().info(f"distance reset (was {self._tracker.total_m:.2f} m)")
        self._tracker.reset()
        self._tick()
        return response

    def _tick(self) -> None:
        self._pub_total.publish(Float64(data=self._tracker.total_m))
        basis = self._tracker.basis.value
        if basis != self._last_basis:
            self._last_basis = basis
            self._pub_basis.publish(String(data=basis))


def main(args: list[str] | None = None) -> None:
    # Context manager owns shutdown; calling destroy_node() after SIGINT raises on Lyrical.
    try:
        with rclpy.init(args=args):
            rclpy.spin(DistanceTrackerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
