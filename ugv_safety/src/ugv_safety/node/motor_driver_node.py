"""rclpy motor driver node: converts final /cmd_vel into wheel actuator commands.

Real-hardware seam -- swap the publish-only body for a serial/CAN write once
physical wheels exist. Not used by the `sim` profile: the Gazebo diff_drive
plugin (ugv_robot_description/urdf/ugv.urdf.xacro) subscribes to /cmd_vel
directly and is the "Base Wheels / Sim" consumer for that profile.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ugv_safety.arbiter.types import Twist2D
from ugv_safety.motor.kinematics import DiffDriveGeometry, twist_to_wheel_speeds


class MotorDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("ugv_motor_driver")
        self.declare_parameter("wheel_radius_m", 0.08)
        self.declare_parameter("track_width_m", 0.40)
        self._geometry = DiffDriveGeometry(
            wheel_radius_m=float(self.get_parameter("wheel_radius_m").value),
            track_width_m=float(self.get_parameter("track_width_m").value),
        )
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self._pub_wheel_cmd = self.create_publisher(Float64MultiArray, "/ugv/wheel_cmd", 10)

    def _on_cmd_vel(self, msg: Twist) -> None:
        wheels = twist_to_wheel_speeds(
            Twist2D(float(msg.linear.x), float(msg.angular.z)), self._geometry
        )
        out = Float64MultiArray()
        out.data = [wheels.left_rad_s, wheels.right_rad_s]
        self._pub_wheel_cmd.publish(out)


def main() -> None:
    rclpy.init()
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
