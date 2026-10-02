"""ROS side of the operator gateway.

Subscribes : /camera/camera_info          sensor_msgs/CameraInfo  stamp only (§12 camera)
             /segmentation/mask           sensor_msgs/Image       stamp only (§12 perception)
             /ugv/perception_degraded     std_msgs/Bool           (Dev 1 -> Dev 5)
             /ugv/pose_valid              std_msgs/Bool           (Dev 2 -> Dev 5)
             /ugv/localization_status     std_msgs/String
             /ugv/nav2_heartbeat          std_msgs/Bool           (Dev 4 -> Dev 5)
             /ugv/nav2_status             std_msgs/String         transient local
             /ugv/e_stop                  std_msgs/Bool           any publisher (CLI, this gateway)
             /ugv/safety_status           std_msgs/String         Dev 5 arbiter (not implemented yet)
             /cmd_vel                     geometry_msgs/Twist     final command, read only
             TF map->base_link (polled; the pose is what GET /map/pose and the SSE `pose` event report)
             map viewer inputs, on a node of their own (ugv_api_map) in a second rclpy context (class _MapInputs):
               always on : /ugv/map/stats, /ugv/perception/stats        std_msgs/String (JSON)
               on demand : /rtabmap/cloud_map, /rtabmap/mapPath, /ugv/elevation/cloud + /ugv/elevation/obstacles,
                           /global_costmap/costmap, /perception/depth/image (depth + live), /image_raw/compressed
Publishes  : /ugv/e_stop                  std_msgs/Bool           latched; re-published while asserted
Clients    : /navigate_to_pose            nav2_msgs/action/NavigateToPose (map-frame goals, §11)
             <rtabmap ns>/set_mode_mapping, set_mode_localization  std_srvs/Empty (§10)

Never publishes /cmd_vel or /cmd_vel_nav2 (architecture §3.1).
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from ugv_api import mapsources as ms
from ugv_api import state as k
from ugv_api.errors import ServiceUnavailable
from ugv_api.goals import GoalRecord, GoalRegistry, GoalState, state_from_status, yaw_to_quaternion
from ugv_api.mapsources import MapConfig
from ugv_api.mapstore import MapStore
from ugv_api.state import StateStore
from ugv_api.watches import Timeouts


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class GatewayNode(Node):
    def __init__(self, store: StateStore, goals: GoalRegistry, maps: MapStore | None = None, **node_kwargs) -> None:
        """`maps` is the store the 3D map layers are put into; without one the gateway has no map inputs.
        `node_kwargs` go to rclpy's Node (tests pass `parameter_overrides`)."""
        super().__init__("ugv_api", **node_kwargs)
        p = self.declare_parameter
        self.host = str(p("host", "127.0.0.1").value)
        self.port = int(p("port", 8080).value)
        self.telemetry_hz = float(p("telemetry_hz", 5.0).value)
        self.cors_origins = [str(o) for o in p("cors_origins", [""]).value if str(o)]
        self.timeouts = Timeouts(
            camera=float(p("timeouts.camera", 0.5).value),
            perception=float(p("timeouts.perception", 0.5).value),
            localization=float(p("timeouts.localization", 0.5).value),
            tf=float(p("timeouts.tf", 0.5).value),
            nav2=float(p("timeouts.nav2", 0.5).value),
        )
        self._map = str(p("map_frame", "map").value)
        self._base = str(p("base_frame", "base_link").value)
        self._rtabmap_ns = str(p("rtabmap_service_ns", "/rtabmap/rtabmap").value).rstrip("/")
        estop_hz = float(p("e_stop_republish_hz", 5.0).value)
        if not self.telemetry_hz > 0 or not estop_hz > 0:
            raise RuntimeError("telemetry_hz and e_stop_republish_hz must be > 0")
        cam_topic = str(p("camera_info_topic", "/camera/camera_info").value)
        self.map_cfg = self._declare_map_config()

        self._store = store
        self._goals = goals
        self._lock = threading.Lock()  # guards e-stop state and the goal-handle table
        self._estop_asserted = False
        self._handles: dict[str, object] = {}
        self.requested_mode: str | None = None
        # (K row-major, width, height) of the last CameraInfo, for the live scan. Written by the camera_info
        # callback here, read by the map node's thread: one reference swap, never mutated in place.
        self.camera_k: tuple[tuple[float, ...], int, int] | None = None

        best_effort = qos_profile_sensor_data  # matches reliable and best-effort publishers
        flags = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(CameraInfo, cam_topic, self._on_camera_info, best_effort)
        self.create_subscription(Image, "/segmentation/mask", self._stamped(k.MASK), best_effort)
        self.create_subscription(Bool, "/ugv/perception_degraded", self._value(k.PERCEPTION_DEGRADED), flags)
        self.create_subscription(Bool, "/ugv/pose_valid", self._value(k.POSE_VALID), flags)
        self.create_subscription(String, "/ugv/localization_status", self._value(k.LOCALIZATION_STATUS), 10)
        self.create_subscription(Bool, "/ugv/nav2_heartbeat", self._value(k.NAV2_HEARTBEAT), flags)
        self.create_subscription(String, "/ugv/nav2_status", self._value(k.NAV2_STATUS), latched)
        # Volatile + reliable matches both `ros2 topic pub` (volatile) and latched publishers.
        self.create_subscription(Bool, "/ugv/e_stop", self._value(k.E_STOP), 10)
        self.create_subscription(String, "/ugv/safety_status", self._value(k.SAFETY_STATUS), 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, flags)

        self._pub_estop = self.create_publisher(Bool, "/ugv/e_stop", latched)
        self.create_timer(1.0 / estop_hz, self._republish_estop)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_timer(0.1, self._poll_tf)

        self._map_inputs = (
            _MapInputs(self, maps, self.map_cfg, self._tf_buffer, self._map) if maps is not None else None
        )

        self._nav = ActionClient(self, NavigateToPose, str(p("navigate_action", "/navigate_to_pose").value))
        self._mode_clients = {
            "mapping": self.create_client(Empty, f"{self._rtabmap_ns}/set_mode_mapping"),
            "localize": self.create_client(Empty, f"{self._rtabmap_ns}/set_mode_localization"),
        }
        self.get_logger().info(f"operator gateway on http://{self.host}:{self.port}/api/v1")

    def destroy_node(self) -> None:
        map_inputs = getattr(self, "_map_inputs", None)
        if map_inputs is not None:
            map_inputs.close()
        super().destroy_node()

    # ---- inputs -------------------------------------------------------------------------------
    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _value(self, key: str):
        def cb(msg) -> None:
            self._store.put(key, msg.data, self.now_ns())

        return cb

    def _stamped(self, key: str):
        def cb(msg) -> None:
            self._store.put(key, None, self.now_ns(), _stamp_ns(msg.header.stamp))

        return cb

    def _on_cmd_vel(self, msg: Twist) -> None:
        lin, ang = msg.linear, msg.angular
        value = ((lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z))
        self._store.put(k.CMD_VEL, value, self.now_ns())

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._store.put(k.CAMERA_INFO, None, self.now_ns(), _stamp_ns(msg.header.stamp))  # §12 camera watch
        self.camera_k = (tuple(map(float, msg.k)), int(msg.width), int(msg.height))

    def _poll_tf(self) -> None:
        try:
            t = self._tf_buffer.lookup_transform(self._map, self._base, Time())
        except TransformException:
            return  # absence shows up as a missing / stale TF watch
        tr, q = t.transform.translation, t.transform.rotation
        pose = (tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)  # what GET /map/pose reports; the §12 watch reads the stamp
        self._store.put(k.TF_MAP_BASE, pose, self.now_ns(), _stamp_ns(t.header.stamp))

    # ---- 3D map inputs ------------------------------------------------------------------------
    def _declare_map_config(self) -> MapConfig:
        """Declare every `map.*` parameter (MapConfig holds the defaults and the types) and validate the lot."""
        values = {}
        for f in dataclasses.fields(MapConfig):
            values[f.name] = type(f.default)(self.declare_parameter(f"map.{f.name}", f.default).value)
        return MapConfig(**values)

    @property
    def map_rejects(self) -> dict[str, int]:
        """How many messages (or frames) each map input has refused or skipped, by input name."""
        return dict(self._map_inputs.rejects) if self._map_inputs is not None else {}

    # ---- e-stop (§3.1 level 1) ----------------------------------------------------------------
    @property
    def estop_asserted(self) -> bool:
        with self._lock:
            return self._estop_asserted

    def set_estop(self, asserted: bool) -> None:
        """Assert: publish true now and keep re-publishing. Release: publish false once."""
        with self._lock:
            self._estop_asserted = asserted
        self._pub_estop.publish(Bool(data=asserted))
        self.get_logger().warning(f"operator e-stop {'ASSERTED' if asserted else 'released'}")

    def _republish_estop(self) -> None:
        if self.estop_asserted:
            self._pub_estop.publish(Bool(data=True))

    # ---- localization mode (§10) --------------------------------------------------------------
    def set_mode(self, mode: str, timeout_s: float = 3.0) -> None:
        client = self._mode_clients[mode]
        if not client.wait_for_service(timeout_sec=0.5):
            raise ServiceUnavailable(f"{client.srv_name} is not available (is RTAB-Map running?)")
        done = threading.Event()
        future = client.call_async(Empty.Request())
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout_s):
            future.cancel()
            raise ServiceUnavailable(f"{client.srv_name} did not answer within {timeout_s:.1f} s")
        if future.exception() is not None:
            raise ServiceUnavailable(f"{client.srv_name} failed: {future.exception()}")
        self.requested_mode = mode

    # ---- navigation goals (§10, §11) ----------------------------------------------------------
    def action_ready(self) -> bool:
        return self._nav.server_is_ready()

    def send_goal(self, rec: GoalRecord) -> None:
        if not self._nav.wait_for_server(timeout_sec=0.5):
            raise ServiceUnavailable("/navigate_to_pose action server is not available (is Nav2 running?)")
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = rec.frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = rec.x
        goal.pose.pose.position.y = rec.y
        q = goal.pose.pose.orientation
        q.x, q.y, q.z, q.w = yaw_to_quaternion(rec.yaw)

        def on_feedback(fb) -> None:
            f = fb.feedback
            fields = {"distance_remaining": float(f.distance_remaining), "recoveries": int(f.number_of_recoveries)}
            cur = self._goals.get(rec.id)
            if cur is not None and cur.state == GoalState.PENDING:  # never undo a CANCELING
                fields["state"] = GoalState.EXECUTING
            self._goals.update(rec.id, **fields)

        future = self._nav.send_goal_async(goal, feedback_callback=on_feedback)
        future.add_done_callback(lambda f: self._on_goal_response(rec.id, f))

    def _on_goal_response(self, goal_id: str, future) -> None:
        if future.exception() is not None:
            self._goals.update(goal_id, state=GoalState.FAILED, error_message=str(future.exception()))
            return
        handle = future.result()
        if not handle.accepted:
            self._goals.update(goal_id, state=GoalState.REJECTED, error_message="rejected by Nav2")
            return
        with self._lock:
            self._handles[goal_id] = handle
        self._goals.preempt_active(except_id=goal_id)
        self._goals.update(goal_id, state=GoalState.EXECUTING)
        handle.get_result_async().add_done_callback(lambda f: self._on_result(goal_id, f))

    def _on_result(self, goal_id: str, future) -> None:
        with self._lock:
            self._handles.pop(goal_id, None)
        if future.exception() is not None:
            self._goals.update(goal_id, state=GoalState.FAILED, error_message=str(future.exception()))
            return
        res = future.result()
        result = res.result
        code = int(getattr(result, "error_code", 0))
        msg = str(getattr(result, "error_msg", "")) or None
        state = state_from_status(res.status)
        if res.status == GoalStatus.STATUS_SUCCEEDED:
            code, msg = None, None
        self._goals.update(goal_id, state=state, error_code=code, error_message=msg)

    def cancel_goal(self, goal_id: str) -> bool:
        """Request cancel. False if the goal has no live Nav2 handle (unknown, pending or finished)."""
        with self._lock:
            handle = self._handles.get(goal_id)
        if handle is None:
            return False
        self._goals.update(goal_id, state=GoalState.CANCELING)
        handle.cancel_goal_async()
        return True


class _MapInputs:
    """The inputs of the 3D map viewer, kept away from the §12 watch inputs.

    Isolation. The map inputs run on a node of their own, `ugv_api_map`, in a second rclpy context (a second DDS
    participant) spun by a single-threaded executor on a thread of its own; every subscription and the 1 Hz
    demand timer share one mutually exclusive callback group on it. Callback groups and executor threads alone
    were measured not to be enough: with only a separate group on the gateway node, a 1 M point (32 MB) cloud
    at 2 Hz made the tf watch stale for 1 to 3 s in three of six 30 s runs (worst age 3.07 s), although no
    callback ran long. The delay sits below rclpy, in the participant that receives the big sample: reliable
    topics of the same participant (TF) wait behind it, a probe process on its own participant saw none of it.
    With the map inputs on their own participant the same six runs had a worst tf age of 0.20 s. The node
    shares nothing with the gateway node but the TF buffer (thread-safe), the store, and the clock reading.

    The executor must stay single-threaded: callbacks (including the timer that destroys subscriptions) then
    run on the thread that builds the wait set. With a MultiThreadedExecutor a `destroy_subscription` from a
    worker thread can land between rclpy marking a subscription's QoS event handler in use and adding it to
    the wait set; the second entry raises InvalidHandle out of `spin` (reproduced within seconds with a
    toggling demand) and ends the thread. One group and one thread also mean these callbacks never run
    concurrently, so `_subs` and the elevation pairer need no lock.

    A callback does the minimum: copy out what the layer's source needs (ugv_api.mapsources, pure functions)
    and `MapStore.put` it. Encoding happens later, on the HTTP thread, only for a layer somebody requests.

    Demand: the heavy subscriptions exist only while `maps.wanted(now, idle_timeout_s)`, which GET /api/v1/map
    keeps true. The timer creates them when it becomes true and destroys them when it stops. The two stats
    subscriptions and the TF lookups are always on.

    Durability: a TRANSIENT_LOCAL subscription only matches a latched publisher, a VOLATILE one matches both
    but misses the latched sample. So each subscription takes the durability its publishers offer (latched only
    when all of them are), VOLATILE when there is none yet, and the timer re-creates a subscription whose
    publisher turned out to offer something else. Losing the publisher does not change anything: the
    subscription is kept for when it comes back.
    """

    ELEVATION_FIELDS = ("x", "y", "z", "confidence", "obstacle_h")

    def __init__(self, gateway: GatewayNode, maps: MapStore, cfg: MapConfig, tf_buffer: Buffer,
                 map_frame: str) -> None:
        self._gw = gateway  # logger, clock (sim-time aware, the one the HTTP layer touches with), camera K
        self._maps = maps
        self._cfg = cfg
        self._tf = tf_buffer
        self._frame = map_frame
        self._closing = threading.Event()
        self._context = Context()
        rclpy.init(context=self._context, domain_id=gateway.context.get_domain_id(),
                   signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node("ugv_api_map", context=self._context, enable_rosout=False,
                                       start_parameter_services=False)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._group = MutuallyExclusiveCallbackGroup()
        self._pairer = ms.ElevationPairer()
        self._subs: dict[str, tuple[Any, DurabilityPolicy]] = {}
        self._gateway_stats: dict[str, int] = {}
        self._stats_seen_ns: dict[str, int] = {}
        self.rejects: dict[str, int] = {}

        c = cfg
        guard = self._guarded
        self._specs: dict[str, tuple[str, type, Callable[[Any], None]]] = {
            "cloud": (c.cloud_topic, PointCloud2, guard("cloud", self._on_cloud)),
            "trajectory": (c.trajectory_topic, Path, guard("trajectory", self._on_path)),
            "elevation_cloud": (c.elevation_cloud_topic, PointCloud2, guard("elevation_cloud", self._on_elevation_cloud)),
            "elevation_obstacles": (c.elevation_obstacles_topic, OccupancyGrid,
                                    guard("elevation_obstacles", self._on_elevation_grid)),
            "grid": (c.grid_topic, OccupancyGrid, guard("grid", self._on_grid)),
            "depth": (c.depth_topic, Image, guard("depth", self._on_depth)),
            "camera": (c.camera_topic, CompressedImage, guard("camera", self._on_camera)),
        }
        # reliable + volatile matches a latched publisher (/ugv/map/stats) and a plain one alike
        stats_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        for source, topic in (("map", c.map_stats_topic), ("perception", c.perception_stats_topic)):
            self._node.create_subscription(String, topic, guard(f"{source}_stats", self._stats_callback(source)),
                                           stats_qos, callback_group=self._group)
        self._node.create_timer(1.0, guard("demand_timer", self._tick), callback_group=self._group)
        self._thread = threading.Thread(target=self._spin, name="ugv_api_map_ros", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        while not self._closing.is_set():
            try:
                self._executor.spin()
                return  # shut down
            except (KeyboardInterrupt, ExternalShutdownException):
                return
            except Exception as exc:  # noqa: BLE001 - the map inputs are not worth a dead thread, and not a dead gateway
                self._reject("map_executor", f"{type(exc).__name__}: {exc}")
                self._closing.wait(1.0)

    def close(self) -> None:
        """Stop the map executor and take the second participant down (GatewayNode.destroy_node)."""
        self._closing.set()
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self._node.destroy_node()
        self._context.try_shutdown()

    # ---- bookkeeping --------------------------------------------------------------------------
    def _reject(self, name: str, why: str) -> None:
        """Count a refused message or skipped frame; log the first one of each kind only."""
        n = self.rejects.get(name, 0) + 1
        self.rejects[name] = n
        if n == 1:
            self._gw.get_logger().warning(f"map input '{name}': {why} (later ones are counted, not logged)")

    def _guarded(self, name: str, fn: Callable[..., None]) -> Callable[..., None]:
        """A bad message is counted and logged, never raised: an exception escaping a callback would end the
        executor's spin and take the §12 watches down with it."""

        def run(*args: Any) -> None:
            try:
                fn(*args)
            except Exception as exc:  # noqa: BLE001 - see the docstring
                self._reject(name, f"{type(exc).__name__}: {exc}")

        return run

    def _stamp_s(self, msg: Any) -> float:
        return ms.stamp_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec, fallback_s=self._gw.now_ns() / 1e9)

    def _count(self, key: str, value: int) -> None:
        self._gateway_stats[key] = int(value)
        self._maps.put_stats("gateway", dict(self._gateway_stats))

    # ---- demand timer -------------------------------------------------------------------------
    def _tick(self) -> None:
        now_ns = self._gw.now_ns()
        if self._maps.wanted(now_ns / 1e9, self._cfg.idle_timeout_s):
            self._ensure_subscriptions()
        else:
            self._drop_subscriptions()
        self._expire_stats(now_ns)

    def _offered_durability(self, topic: str) -> DurabilityPolicy | None:
        """What the topic's publishers offer: None without a publisher, TRANSIENT_LOCAL when all of them latch."""
        infos = self._node.get_publishers_info_by_topic(topic)
        if not infos:
            return None
        if all(i.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for i in infos):
            return DurabilityPolicy.TRANSIENT_LOCAL
        return DurabilityPolicy.VOLATILE

    def _ensure_subscriptions(self) -> None:
        for key, (topic, msg_type, callback) in self._specs.items():
            try:
                offered = self._offered_durability(topic)
                current = self._subs.get(key)
                if current is not None and (offered is None or offered == current[1]):
                    continue
                if current is not None:  # the publisher offers something else than we asked for
                    self._node.destroy_subscription(current[0])
                    del self._subs[key]
                durability = offered or DurabilityPolicy.VOLATILE
                qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=durability)
                sub = self._node.create_subscription(msg_type, topic, callback, qos, callback_group=self._group)
                self._subs[key] = (sub, durability)
            except Exception as exc:  # noqa: BLE001 - one bad topic must not stop the others
                self._reject(f"subscribe_{key}", f"{type(exc).__name__}: {exc}")

    def _drop_subscriptions(self) -> None:
        for sub, _durability in self._subs.values():
            self._node.destroy_subscription(sub)
        self._subs.clear()
        self._pairer.reset()

    def _expire_stats(self, now_ns: int) -> None:
        limit_ns = int(self._cfg.stats_stale_s * 1e9)
        for source, seen_ns in list(self._stats_seen_ns.items()):
            if now_ns - seen_ns > limit_ns:
                self._maps.put_stats(source, {})  # a source that went quiet must not keep showing old numbers
                del self._stats_seen_ns[source]

    # ---- statistics ---------------------------------------------------------------------------
    def _stats_callback(self, source: str) -> Callable[[String], None]:
        def on_stats(msg: String) -> None:
            values = ms.parse_stats(msg.data)
            if values is None:
                self._reject(f"{source}_stats", "not a JSON object, ignored")
                return
            self._maps.put_stats(source, values)
            self._stats_seen_ns[source] = self._gw.now_ns()

        return on_stats

    # ---- layers -------------------------------------------------------------------------------
    @staticmethod
    def _cloud_args(msg: PointCloud2) -> dict[str, Any]:
        return {
            "fields": [(f.name, f.offset, f.datatype, f.count) for f in msg.fields],
            "point_step": msg.point_step,
            "n_points": msg.width * msg.height,
            "is_bigendian": msg.is_bigendian,
            "data": msg.data,
        }

    def _on_cloud(self, msg: PointCloud2) -> None:
        source = ms.cloud_source(**self._cloud_args(msg))  # a view of msg.data, which rclpy never reuses
        self._maps.put("cloud", source, self._stamp_s(msg))
        self._count("cloud_source_points", source["n_points"])

    def _on_path(self, msg: Path) -> None:
        rows = [(p.pose.position.x, p.pose.position.y, p.pose.position.z, p.pose.orientation.x,
                 p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w) for p in msg.poses]
        self._maps.put("trajectory", ms.trajectory_source(rows), self._stamp_s(msg))

    def _on_grid(self, msg: OccupancyGrid) -> None:
        info, o = msg.info, msg.info.origin
        source = ms.grid_source(
            data=msg.data, width=info.width, height=info.height, resolution=info.resolution,
            origin_x=o.position.x, origin_y=o.position.y,
            origin_q=(o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w))
        self._maps.put("grid", source, self._stamp_s(msg))

    def _on_elevation_cloud(self, msg: PointCloud2) -> None:
        columns = ms.cloud_columns(**self._cloud_args(msg), names=self.ELEVATION_FIELDS)
        self._count("elevation_known_cells", len(columns["x"]))
        source = self._pairer.add_cloud(_stamp_ns(msg.header.stamp), columns)
        if source is not None:
            self._maps.put("elevation", source, self._stamp_s(msg))

    def _on_elevation_grid(self, msg: OccupancyGrid) -> None:
        info = msg.info
        source = self._pairer.add_grid(
            _stamp_ns(msg.header.stamp), resolution=info.resolution, width=info.width, height=info.height,
            origin_x=info.origin.position.x, origin_y=info.origin.position.y)
        if source is not None:
            self._maps.put("elevation", source, self._stamp_s(msg))

    def _on_camera(self, msg: CompressedImage) -> None:
        self._maps.put("camera", ms.jpeg_source(msg.data), self._stamp_s(msg))

    def _on_depth(self, msg: Image) -> None:
        depth = ms.depth_source(encoding=msg.encoding, height=msg.height, width=msg.width, step=msg.step,
                                is_bigendian=msg.is_bigendian, data=msg.data)
        stamp_s = self._stamp_s(msg)
        self._maps.put("depth", depth, stamp_s)
        self._put_live(msg, depth["depth_m"], stamp_s)

    def _put_live(self, msg: Image, depth_m: Any, stamp_s: float) -> None:
        """The depth image back-projected with CameraInfo K and moved into the map frame. Without K or without
        any TF map <- camera the layer is skipped for this frame: camera-frame points are never published as
        map-frame points."""
        cam = self._gw.camera_k
        intr = None
        if cam is not None:
            intr = ms.intrinsics(cam[0], width=msg.width, height=msg.height, info_width=cam[1], info_height=cam[2])
        if intr is None:
            self._reject("live_no_intrinsics", "no usable CameraInfo K yet, live layer skipped")
            return
        transform = self._camera_to_map(msg.header.frame_id, msg.header.stamp)
        if transform is None:
            self._reject("live_no_transform", f"no TF {self._frame} <- '{msg.header.frame_id}', live layer skipped")
            return
        cfg = self._cfg
        points = ms.backproject(depth_m, intr, stride=cfg.live_stride, min_range_m=cfg.live_range_min_m,
                                max_range_m=cfg.live_range_max_m)
        self._maps.put("live", {"xyz": ms.transform_points(points, *transform)}, stamp_s)

    def _camera_to_map(self, frame_id: str, stamp: Any) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """(translation, quaternion) of `frame_id` in the map frame at the image stamp, else the latest one."""
        for when in (Time.from_msg(stamp), Time()):
            try:
                t = self._tf.lookup_transform(self._frame, frame_id, when)
            except TransformException:
                continue
            tr, q = t.transform.translation, t.transform.rotation
            return (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w)
        return None
