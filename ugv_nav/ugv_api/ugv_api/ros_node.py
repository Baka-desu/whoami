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
             TF map->base_link (polled)
Publishes  : /ugv/e_stop                  std_msgs/Bool           latched; re-published while asserted
Clients    : /navigate_to_pose            nav2_msgs/action/NavigateToPose (map-frame goals, §11)
             <rtabmap ns>/set_mode_mapping, set_mode_localization  std_srvs/Empty (§10)

Never publishes /cmd_vel or /cmd_vel_nav2 (architecture §3.1).
"""

from __future__ import annotations

import threading

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from ugv_api import state as k
from ugv_api.errors import ServiceUnavailable
from ugv_api.goals import GoalRecord, GoalRegistry, GoalState, state_from_status, yaw_to_quaternion
from ugv_api.state import StateStore
from ugv_api.watches import Timeouts


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class GatewayNode(Node):
    def __init__(self, store: StateStore, goals: GoalRegistry) -> None:
        super().__init__("ugv_api")
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

        self._store = store
        self._goals = goals
        self._lock = threading.Lock()  # guards e-stop state and the goal-handle table
        self._estop_asserted = False
        self._handles: dict[str, object] = {}
        self.requested_mode: str | None = None

        best_effort = qos_profile_sensor_data  # matches reliable and best-effort publishers
        flags = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(CameraInfo, cam_topic, self._stamped(k.CAMERA_INFO), best_effort)
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

        self._nav = ActionClient(self, NavigateToPose, str(p("navigate_action", "/navigate_to_pose").value))
        self._mode_clients = {
            "mapping": self.create_client(Empty, f"{self._rtabmap_ns}/set_mode_mapping"),
            "localize": self.create_client(Empty, f"{self._rtabmap_ns}/set_mode_localization"),
        }
        self.get_logger().info(f"operator gateway on http://{self.host}:{self.port}/api/v1")

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

    def _poll_tf(self) -> None:
        try:
            t = self._tf_buffer.lookup_transform(self._map, self._base, Time())
        except TransformException:
            return  # absence shows up as a missing / stale TF watch
        self._store.put(k.TF_MAP_BASE, None, self.now_ns(), _stamp_ns(t.header.stamp))

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
