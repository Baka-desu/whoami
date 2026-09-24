"""E-stop CLI utility (dev.md Dev 5 task 6).

Publishes /ugv/e_stop once, latched via TRANSIENT_LOCAL so a late-joining
arbiter still sees the current state, then exits. Level 1 in the 4-tier
arbiter -- this is the operator hard kill switch.

Usage:
    ros2 run ugv_eval ugv-estop --set true
    ros2 run ugv_eval ugv-estop --set false
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from ugv_eval.bool_arg import parse_bool


def _latched() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class EstopPublisher(Node):
    def __init__(self) -> None:
        super().__init__("ugv_estop_cli")
        self._pub = self.create_publisher(Bool, "/ugv/e_stop", _latched())

    def publish_once(self, value: bool) -> None:
        msg = Bool()
        msg.data = value
        self._pub.publish(msg)
        # Give the publisher's discovery a moment before we tear down.
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Toggle the UGV hard e-stop.")
    parser.add_argument("--set", type=parse_bool, required=True, help="true|false")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    node = EstopPublisher()
    try:
        node.publish_once(args.set)
        node.get_logger().info(f"/ugv/e_stop <- {args.set}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
