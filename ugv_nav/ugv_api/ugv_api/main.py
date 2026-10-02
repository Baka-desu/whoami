"""Entry point: rclpy spins on a background executor thread, uvicorn serves HTTP on the main thread.

    ros2 launch ugv_api api.launch.py            # 127.0.0.1:8080
    ros2 run ugv_api api_gateway --ros-args -p port:=8080
"""

from __future__ import annotations

import threading

import rclpy
import uvicorn
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from ugv_api.app import create_app
from ugv_api.goals import GoalRegistry
from ugv_api.mapstore import MapStore
from ugv_api.ros_node import GatewayNode
from ugv_api.state import StateStore


def main(args: list[str] | None = None) -> None:
    # uvicorn owns SIGINT/SIGTERM; when it stops serving, ROS is shut down below.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    store, goals, maps = StateStore(), GoalRegistry(), MapStore()
    node = GatewayNode(store, goals, maps)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def spin() -> None:
        try:
            executor.spin()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass

    spinner = threading.Thread(target=spin, name="ugv_api_ros", daemon=True)
    spinner.start()
    app = create_app(node, store, goals, telemetry_hz=node.telemetry_hz, cors_origins=node.cors_origins, maps=maps,
                     **node.map_cfg.app_kwargs())
    try:
        uvicorn.run(app, host=node.host, port=node.port, log_level="info")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spinner.join(timeout=2.0)
