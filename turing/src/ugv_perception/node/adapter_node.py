"""ROS 2 perception node: subscribe Image+CameraInfo, compose_tick, publish port."""

from __future__ import annotations

import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64MultiArray

from ugv_perception.compose.load import load_compose_configs
from ugv_perception.ingest.msgs import CameraInfoView, ImageView
from ugv_perception.ingest.ros_bridge import camera_info_msg_to_view, image_msg_to_view
from ugv_perception.node.cycle import perception_cycle
from ugv_perception.node.wire import wire_compose_out

_ROOT = Path(__file__).resolve().parents[3]


class PerceptionAdapterNode(Node):
    def __init__(
        self,
        *,
        adapter: object,
        remap_path: Path | None = None,
        gates_path: Path | None = None,
        freshness_path: Path | None = None,
        now_ns_fn: object | None = None,
    ) -> None:
        super().__init__("ugv_perception")
        self.declare_parameter("adapter", "yoloe")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        adapter_name = self.get_parameter("adapter").get_parameter_value().string_value
        if adapter_name != "yoloe":
            raise ValueError("adapter:=onnx is illegal until T09; only yoloe")
        remap_path = remap_path or _ROOT / "config" / "ontologies" / "yoloe.yaml"
        gates_path = gates_path or _ROOT / "config" / "perception" / "yoloe.yaml"
        freshness_path = freshness_path or _ROOT / "config" / "perception" / "port.yaml"
        table, gates, fresh = load_compose_configs(
            remap_path=remap_path,
            gates_path=gates_path,
            freshness_path=freshness_path,
        )
        self._adapter = adapter
        self._table = table
        self._gates = gates
        self._fresh = fresh
        self._now_ns_fn = now_ns_fn or time.time_ns
        self._last_image: ImageView | None = None
        self._last_info: CameraInfoView | None = None
        self._last_camera_info_msg: CameraInfo | None = None
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_subscription(CameraInfo, info_topic, self._on_info, 10)
        self._pub_degraded = self.create_publisher(Bool, "/ugv/perception_degraded", 10)
        self._pub_mask = self.create_publisher(Image, "/segmentation/mask", 10)
        self._pub_conf = self.create_publisher(Image, "/segmentation/confidence", 10)
        self._pub_meta = self.create_publisher(Float64MultiArray, "/segmentation/port_meta", 10)
        self._pub_cinfo = self.create_publisher(CameraInfo, "/segmentation/camera_info", 10)

    def _on_image(self, msg: Image) -> None:
        self._last_image = image_msg_to_view(msg)
        self._tick()

    def _on_info(self, msg: CameraInfo) -> None:
        self._last_info = camera_info_msg_to_view(msg)
        self._last_camera_info_msg = msg
        self._tick()

    def _tick(self) -> None:
        now_ns = self._now_ns_fn()
        if type(now_ns) is not int or now_ns <= 0:
            raise TypeError("now_ns must be a Python int > 0")
        out = perception_cycle(
            image=self._last_image,
            camera_info=self._last_info,
            now_ns=now_ns,
            adapter=self._adapter,
            remap_table=self._table,
            gate_profile=self._gates,
            freshness_profile=self._fresh,
        )
        wired = wire_compose_out(out)
        self._pub_degraded.publish(wired.degraded)
        if wired.mask is not None:
            self._pub_mask.publish(wired.mask)
            self._pub_conf.publish(wired.confidence)
            self._pub_meta.publish(wired.port_meta)
            if self._last_camera_info_msg is not None:
                self._pub_cinfo.publish(self._last_camera_info_msg)


def main() -> None:
    from ugv_perception.adapter.prompts import load_prompts
    from ugv_perception.adapter.yoloe import YoloeAdapter, load_adapter_config
    from ugv_perception.backend.factory import build_backend

    cfg = load_adapter_config(_ROOT / "config" / "adapters" / "yoloe.yaml")
    prompts = load_prompts(
        _ROOT / "config" / "perception" / "yoloe_prompts.yaml",
        _ROOT / "config" / "ontologies" / "yoloe.yaml",
    )
    backend = build_backend(cfg["backend"], str(_ROOT / cfg["weights"]), prompts)
    adapter = YoloeAdapter(backend, prompts)
    rclpy.init()
    node = PerceptionAdapterNode(adapter=adapter)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
