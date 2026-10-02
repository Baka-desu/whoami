"""/ugv/map/stats publisher: the map's statistics for the web viewer (docs/localization/interfaces.md).

Subscribes : graph_topic  rtabmap_msgs/MapGraph  pose graph: keyframe count, path length, closure links,
                                                 "did the map change"
             Not /rtabmap/cloud_map and not /rtabmap/mapData: a subscriber on either makes RTAB-Map assemble
             and send the whole map every step. The graph is poses and links only. Not /rtabmap/info either.
Publishes  : /ugv/map/stats  std_msgs/String  one JSON object of scalars (keys: ugv_localization.mapstats),
             every 1/publish_rate_hz s, reliable + transient local (a late subscriber gets the latest)
Params     : mode (mapping | localize, required), database_path (file whose size is reported; "" = none),
             calibration_file (camera calibration YAML; its `placeholder` flag is reported; "" = none),
             publish_rate_hz (1.0), graph_topic (/rtabmap/mapGraph)
"""

from __future__ import annotations

import json
import math
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import MapGraph
from std_msgs.msg import String

from ugv_localization.camera import load_calibration
from ugv_localization.mapstats import MapStats, regular_file_size
from ugv_localization.modes import parse_mode

_NS_PER_S = 1_000_000_000


class MapStatsNode(Node):
    def __init__(self) -> None:
        super().__init__("map_stats")
        mode = parse_mode(self.declare_parameter("mode", "").value)
        db = self.declare_parameter("database_path", "").value
        self._db_path = db or ""  # regular_file_size() expands ~
        calibration_file = self.declare_parameter("calibration_file", "").value
        rate_hz = float(self.declare_parameter("publish_rate_hz", 1.0).value)
        if not (math.isfinite(rate_hz) and rate_hz > 0.0):
            raise RuntimeError(f"publish_rate_hz must be > 0, got {rate_hz}")
        graph_topic = self.declare_parameter("graph_topic", "/rtabmap/mapGraph").value

        # A graph that arrives before the sim clock starts (localize sends it once) is kept and applied on the
        # first tick with a clock, so it is stamped when the clock starts instead of being lost.
        self._pending: tuple[list[int], list[tuple[float, ...]], list[tuple[int, int, int]]] | None = None
        self._stats = MapStats(mode, self._calibration_is_placeholder(calibration_file))

        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._pub = self.create_publisher(String, "/ugv/map/stats", latched)
        # Same QoS as the rtabmap publishers (reliable, transient local).
        self.create_subscription(MapGraph, graph_topic, self._on_graph, latched)
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"map stats: mode={mode.value}, database_path={self._db_path or '-'}, publishing /ugv/map/stats"
        )
        self._tick()  # a subscriber that connects before the first tick still finds a sample

    def _calibration_is_placeholder(self, path: str) -> bool:
        if not path:
            return False
        try:
            return load_calibration(os.path.expanduser(path)).placeholder
        except Exception as exc:  # noqa: BLE001  any failure to read it means "not known to be a placeholder"
            self.get_logger().warning(
                f"calibration_file {path!r} not loaded ({exc}); reporting calibration_placeholder=false"
            )
            return False

    def _now_s(self) -> float | None:
        now_ns = self.get_clock().now().nanoseconds
        return now_ns / _NS_PER_S if now_ns > 0 else None  # None: use_sim_time and /clock not received yet

    def _on_graph(self, msg: MapGraph) -> None:
        poses = [
            (p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
            for p in msg.poses
        ]
        links = [(int(k.from_id), int(k.to_id), int(k.type)) for k in msg.links]
        self._pending = (list(msg.poses_id), poses, links)
        self._apply_pending()

    def _apply_pending(self) -> None:
        now_s = self._now_s()
        if now_s is None or self._pending is None:
            return  # no clock yet: keep the graph, the next tick applies it
        ids, poses, links = self._pending
        self._pending = None
        try:
            self._stats.on_graph(ids, poses, now_s, links)
        except ValueError as exc:
            self.get_logger().warning(f"dropping malformed map graph: {exc}", throttle_duration_sec=5.0)

    def _db_bytes(self) -> int | None:
        return regular_file_size(self._db_path)

    def _tick(self) -> None:
        self._apply_pending()
        snapshot = self._stats.snapshot(self._now_s() or 0.0, self._db_bytes())
        self._pub.publish(String(data=json.dumps(snapshot, allow_nan=False, separators=(",", ":"))))


def main(args: list[str] | None = None) -> None:
    # Context manager owns shutdown; calling destroy_node() after SIGINT raises on Lyrical.
    try:
        with rclpy.init(args=args):
            rclpy.spin(MapStatsNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
