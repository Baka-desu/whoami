#!/usr/bin/env python3
"""
TEST-ONLY kinematic diff-drive base. NOT the product safety path.

Stands in for Dev 5 (safety authority + base) and Dev 2 (odometry / odom TF) so the
Dev 4 stack can drive in closed loop: integrates the candidate twist on /cmd_vel_nav2
as a unicycle and publishes /odom plus TF odom -> base_link. It never publishes
/cmd_vel. Like a watchdog, a candidate older than `cmd_timeout` is treated as zero.
"""

import math
import sys

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

RATE_HZ = 50.0


class FakeBase(Node):

    def __init__(self):
        super().__init__('test_fake_base')
        self.cmd_timeout = self.declare_parameter('cmd_timeout', 0.5).value
        self.x = self.y = self.yaw = 0.0
        self.cmd = Twist()
        self.cmd_time = None
        self.tf = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_subscription(Twist, '/cmd_vel_nav2', self.on_cmd, 10)
        self.last = self.get_clock().now()
        self.create_timer(1.0 / RATE_HZ, self.step)

    def on_cmd(self, msg):
        self.cmd = msg
        self.cmd_time = self.get_clock().now()

    def step(self):
        now = self.get_clock().now()
        dt = (now - self.last).nanoseconds * 1e-9
        self.last = now
        v = w = 0.0
        if self.cmd_time is not None and \
                (now - self.cmd_time).nanoseconds * 1e-9 <= self.cmd_timeout:
            v, w = self.cmd.linear.x, self.cmd.angular.z
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw = math.atan2(math.sin(self.yaw + w * dt), math.cos(self.yaw + w * dt))
        qz, qw = math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf.sendTransform(t)

        odom = Odometry()
        odom.header = t.header
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)


def main():
    rclpy.init(args=sys.argv)
    node = FakeBase()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
