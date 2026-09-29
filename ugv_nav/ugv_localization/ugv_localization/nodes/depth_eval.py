"""DA3 depth vs ground-truth depth, paired by exact stamp → report.json on exit (sensor honesty).

Subscribes : depth     sensor_msgs/Image 32FC1 m   (DA3 depth as converted from Dev 1's cloud: /rtabmap/depth/image)
             gt_depth  sensor_msgs/Image 32FC1 m   (Dev 5 sim depth camera; same camera, same size)
Params     : out_dir, stride (every Nth pixel row/column), duration_s (0 = until Ctrl-C),
             bins_m (range bins for abs_rel_by_range)
Output     : <out_dir>/depth_report.json   (depth.metrics.DepthErrorAccumulator.report())

    ros2 run ugv_localization depth_eval --ros-args -r depth:=/rtabmap/depth/image -r gt_depth:=/camera/depth/image_raw \
        -p use_sim_time:=true -p out_dir:=eval_out/depth_run1
"""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

from ugv_localization.depth import DepthErrorAccumulator, depth_values
from ugv_localization.rosconv import stamp_to_ns

_PENDING_MAX = 60  # unmatched frames kept per stream (DA3 drops frames; GT does not)


def _decode(msg: Image, stride: int):
    if msg.encoding != "32FC1":
        raise ValueError(f"encoding {msg.encoding!r} (need 32FC1 meters)")
    return depth_values(
        bytes(msg.data),
        width=int(msg.width),
        height=int(msg.height),
        step=int(msg.step),
        is_bigendian=bool(msg.is_bigendian),
        stride=stride,
    )


class DepthEval(Node):
    def __init__(self) -> None:
        super().__init__("depth_eval")
        self._out = Path(self.declare_parameter("out_dir", "eval_out/depth").value)
        self._stride = int(self.declare_parameter("stride", 4).value)
        self._duration_ns = int(self.declare_parameter("duration_s", 0.0).value * 1e9)
        bins = [float(b) for b in self.declare_parameter("bins_m", [0.0, 2.0, 4.0, 8.0]).value]
        self._acc = DepthErrorAccumulator(bins_m=bins)
        self._pending: dict[str, dict[int, Image]] = {"est": {}, "gt": {}}
        self._start_ns: int | None = None
        self.done = False
        self.create_subscription(Image, "depth", lambda m: self._on(m, "est"), 10)
        self.create_subscription(Image, "gt_depth", lambda m: self._on(m, "gt"), 10)

    def _on(self, msg: Image, which: str) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp)
        except (TypeError, ValueError):
            return
        other = "gt" if which == "est" else "est"
        match = self._pending[other].pop(ns, None)
        if match is None:
            mine = self._pending[which]
            mine[ns] = msg
            while len(mine) > _PENDING_MAX:
                mine.pop(min(mine))
            return
        est, gt = (msg, match) if which == "est" else (match, msg)
        try:
            self._acc.add(_decode(est, self._stride), _decode(gt, self._stride))
        except ValueError as exc:
            self.get_logger().warning(f"skipping pair: {exc}", throttle_duration_sec=2.0)
            return
        if self._start_ns is None:
            self._start_ns = ns
        if self._duration_ns and ns - self._start_ns >= self._duration_ns:
            self.done = True

    def write(self) -> None:
        self._out.mkdir(parents=True, exist_ok=True)
        report = self._acc.report()
        (self._out / "depth_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.get_logger().info(f"depth report → {self._out / 'depth_report.json'}: {report}")


def main(args: list[str] | None = None) -> None:
    node: DepthEval | None = None
    try:
        with rclpy.init(args=args):
            node = DepthEval()
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.write()  # plain file I/O; safe after rclpy shutdown
