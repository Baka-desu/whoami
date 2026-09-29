#!/usr/bin/env python3
"""
TEST-ONLY: publish a closed-loop scenario map (stands in for Dev 3's costmap output).

Publishes the scenario OccupancyGrid on /test/scenario_map (transient local). For a
scenario with late boxes it republishes the map with them added once the robot's
odometry x passes the trigger, so the planner and controller must react to a hazard
that appears mid-run.
"""

import sys

from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scenarios import build_grid, HEIGHT, ORIGIN, RESOLUTION, SCENARIOS, WIDTH

MAP_TOPIC = '/test/scenario_map'


class ScenarioMap(Node):

    def __init__(self):
        super().__init__('test_scenario_map')
        name = self.declare_parameter('scenario', 'open').value
        self.scenario = SCENARIOS[name]
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(OccupancyGrid, MAP_TOPIC, qos)
        self.late_published = False
        self.publish(self.scenario.boxes)
        if self.scenario.late_boxes:
            self.create_subscription(Odometry, '/odom', self.on_odom, 10)

    def publish(self, boxes):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = RESOLUTION
        msg.info.width = WIDTH
        msg.info.height = HEIGHT
        msg.info.origin.position.x = ORIGIN[0]
        msg.info.origin.position.y = ORIGIN[1]
        msg.info.origin.orientation.w = 1.0
        msg.data = build_grid(boxes)
        self.pub.publish(msg)
        self.get_logger().info(f'scenario "{self.scenario.name}": published {len(boxes)} boxes')

    def on_odom(self, msg):
        if not self.late_published and msg.pose.pose.position.x >= self.scenario.trigger_x:
            self.late_published = True
            self.publish(self.scenario.boxes + self.scenario.late_boxes)


def main():
    rclpy.init(args=sys.argv)
    node = ScenarioMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
