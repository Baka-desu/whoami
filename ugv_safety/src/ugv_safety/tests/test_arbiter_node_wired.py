"""Optional ROS contract for the safety arbiter node. Skip if rclpy/msgs are
missing -- not a failure, mirrors turing/src/ugv_perception/tests/test_port_wired_ros.py.

Exercises the actual rclpy node (topics, QoS, timer tick), not just the pure
decide()/TwistRamp kernels those already cover in test_arbiter.py /
test_safety_precedence.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("geometry_msgs")
pytest.importorskip("sensor_msgs")
pytest.importorskip("std_msgs")

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from ugv_safety.node.arbiter_node import SafetyArbiterNode


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def test_fresh_node_holds_zero_by_default() -> None:
    """No upstream has reported in yet -- fail closed, never publish motion."""
    rclpy.init()
    node = SafetyArbiterNode()
    cmds: list[Twist] = []
    helper = Node("t_safety_fresh_helper")
    helper.create_subscription(Twist, "/cmd_vel", cmds.append, 10)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        for _ in range(10):
            ex.spin_once(timeout_sec=0.1)
        assert cmds, "expected /cmd_vel to be published"
        assert all(c.linear.x == 0.0 and c.angular.z == 0.0 for c in cmds)
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_estop_overrides_a_moving_candidate() -> None:
    rclpy.init()
    node = SafetyArbiterNode()
    cmds: list[Twist] = []
    helper = Node("t_safety_estop_helper")
    helper.create_subscription(Twist, "/cmd_vel", cmds.append, 10)
    pub_estop = helper.create_publisher(Bool, "/ugv/e_stop", _latched_qos())
    pub_candidate = helper.create_publisher(Twist, "/cmd_vel_nav2", 10)

    estop_msg = Bool(data=True)
    candidate_msg = Twist()
    candidate_msg.linear.x = 1.0
    candidate_msg.angular.z = 0.5

    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        pub_estop.publish(estop_msg)
        for _ in range(20):
            pub_candidate.publish(candidate_msg)
            ex.spin_once(timeout_sec=0.1)
        assert cmds, "expected /cmd_vel to be published"
        assert all(c.linear.x == 0.0 and c.angular.z == 0.0 for c in cmds), (
            "estop must hold /cmd_vel at zero even with a moving Nav2 candidate"
        )
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_all_clear_ramps_toward_candidate() -> None:
    rclpy.init()
    node = SafetyArbiterNode()
    cmds: list[Twist] = []
    helper = Node("t_safety_clear_helper")
    helper.create_subscription(Twist, "/cmd_vel", cmds.append, 10)
    pub_degraded = helper.create_publisher(Bool, "/ugv/perception_degraded", 10)
    pub_pose_valid = helper.create_publisher(Bool, "/ugv/pose_valid", 10)
    pub_candidate = helper.create_publisher(Twist, "/cmd_vel_nav2", 10)
    pub_image = helper.create_publisher(Image, "/camera/image_raw", 10)

    candidate_msg = Twist()
    candidate_msg.linear.x = 0.3
    candidate_msg.angular.z = 0.0
    image_msg = Image()

    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        for _ in range(200):
            pub_degraded.publish(Bool(data=False))
            pub_pose_valid.publish(Bool(data=True))
            pub_candidate.publish(candidate_msg)
            pub_image.publish(image_msg)
            ex.spin_once(timeout_sec=0.02)
        assert cmds, "expected /cmd_vel to be published"
        assert cmds[-1].linear.x == pytest.approx(0.3, abs=0.05), (
            "expected published twist to ramp up toward the Nav2 candidate once all-clear"
        )
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
