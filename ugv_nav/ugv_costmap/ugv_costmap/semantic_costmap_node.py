"""Dev 3 semantic costmap node: /segmentation/mask -> /semantic_costmap/grid (nav_msgs/OccupancyGrid).

    ros2 run ugv_costmap semantic_costmap_node --ros-args --params-file config/semantic_costmap.yaml

Runs costmap_core's pipeline (mask -> ground projection -> semantic costmap) on every current mask and
publishes the result in the robot frame. Nav2's global and local costmaps read it with a StaticLayer
(src/ugv_navigation/config/costmaps.yaml) and add the Depth Anything geometry with a VoxelLayer whose
Max combination keeps geometry lethal over semantic traversable (architecture §9).

Inputs:  /segmentation/mask (mono8 {0,1,2}), camera_info_topic (K; the driver's /camera/camera_info),
         TF robot_frame <- mask frame (the static camera mount).
         Dev 1 publishes a mask only when it is valid, so validity is judged on the mask itself (its stamp age),
         not on /segmentation/port_meta: that topic carries no stamp and arrives after its mask, so pairing it
         would judge each mask by the previous frame's flag.
Output:  output_topic, frame robot_frame, stamp = mask stamp, reliable + transient local.

Fail-safe (architecture §8.4, §8.6): a mask whose stamp is older than max_mask_age_s (or in the future), or no
mask for longer than max_mask_age_s, publishes the camera's whole field of view as lethal. Unseen cells are -1 (unknown); observed class 0 is
`unknown_occupancy` (never free).
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from costmap_core.contracts import (
    CameraGroundInput, CameraIntrinsicsInput, CostmapCoreInputs, GridInput, SemanticMaskInput,
)
from costmap_core.grid import CostmapGridGeometry
from costmap_core.mask_projection import project_mask_to_costmap
from costmap_core.pipeline import run_costmap_pipeline
from ugv_costmap.adapter import (
    AdapterError, all_traversable, costs_to_occupancy, front_roi_lethal, mask_is_fresh, mount_from_tf, pool_mask,
    scale_intrinsics,
)

_GROUND = "camera_ground"  # core-internal frame: the ground point under the camera, x along its heading


def _latched() -> QoSProfile:
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class SemanticCostmapNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_costmap")
        p = self.declare_parameter
        self._robot_frame = p("robot_frame", "base_link").value
        self._res = float(p("resolution", 0.1).value)
        self._range = float(p("max_range", 6.0).value)
        self._half_w = float(p("half_width", 4.0).value)
        self._factor = int(p("mask_downsample", 4).value)
        self._unknown_occ = int(p("unknown_occupancy", 50).value)
        self._max_age = float(p("max_mask_age_s", 0.5).value)
        info_topic = p("camera_info_topic", "/camera/camera_info").value
        self._out_topic = p("output_topic", "/semantic_costmap/grid").value

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._lock = threading.Lock()
        self._busy = threading.Lock()
        self._info: CameraInfo | None = None
        self._model = None  # (key, intrinsics, mount, grid, coverage), rebuilt only if camera or mount change
        self._last_good_ns = 0
        self._failsafe_sent = False

        self._pub = self.create_publisher(OccupancyGrid, self._out_topic, _latched())
        self.create_subscription(CameraInfo, info_topic, self._on_info, _latched())
        self.create_subscription(Image, "/segmentation/mask", self._on_mask, 1)
        self.create_timer(0.1, self._watchdog)
        self.get_logger().info(
            f"semantic costmap: /segmentation/mask -> {self._out_topic} in {self._robot_frame}, "
            f"{self._range:g} m ahead x {2 * self._half_w:g} m wide @ {self._res:g} m, mask / {self._factor}"
        )

    # --- inputs -------------------------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._info = msg

    def _on_mask(self, msg: Image) -> None:
        if not self._busy.acquire(blocking=False):
            return  # still projecting the previous mask: keep only the newest
        try:
            self._process(msg)
        except (AdapterError, ValueError) as exc:
            self.get_logger().error(f"mask refused: {exc}", throttle_duration_sec=5.0)
        finally:
            self._busy.release()

    # --- camera model ------------------------------------------------------------------------------
    def _camera_model(self, mask_frame: str, mask_hw: tuple[int, int]):
        with self._lock:
            info = self._info
        if info is None:
            raise AdapterError("no CameraInfo yet")
        if info.header.frame_id != mask_frame or (info.height, info.width) != mask_hw:
            raise AdapterError(
                f"mask {mask_frame} {mask_hw[1]}x{mask_hw[0]} does not match CameraInfo "
                f"{info.header.frame_id} {info.width}x{info.height}"
            )
        try:
            t = self._tf.lookup_transform(self._robot_frame, mask_frame, Time(), timeout=Duration(seconds=0.2))
        except TransformException as exc:
            raise AdapterError(f"no TF {self._robot_frame} <- {mask_frame}: {exc}") from exc
        tr, rot = t.transform.translation, t.transform.rotation
        key = (tuple(info.k), info.width, info.height, mask_frame,
               *(round(v, 4) for v in (tr.x, tr.y, tr.z, rot.x, rot.y, rot.z, rot.w)))
        if self._model is not None and self._model[0] == key:
            return self._model
        mount = mount_from_tf((tr.x, tr.y, tr.z), (rot.x, rot.y, rot.z, rot.w))
        k = info.k
        h, w = mask_hw[0] // self._factor, mask_hw[1] // self._factor
        intr = CameraIntrinsicsInput(intrinsics=scale_intrinsics(k[0], k[4], k[2], k[5], self._factor),
                                     image_width=w, image_height=h, frame_id=mask_frame)
        # Grid in the core's ground frame (origin under the camera); published shifted by the mount offset.
        n_x = int(round(self._range / self._res))
        n_y = int(round(2 * self._half_w / self._res))
        grid = GridInput(CostmapGridGeometry(resolution=self._res, origin_x=0.0, origin_y=-self._half_w,
                                             width=n_x, height=n_y), frame_id=_GROUND)
        coverage = project_mask_to_costmap(all_traversable((h, w)), intr.intrinsics, mount.geometry,
                                           grid.geometry) != 255
        self._model = (key, intr, mount, grid, coverage)
        self.get_logger().info(
            f"camera mount: {mount.geometry.camera_height:.3f} m high, pitch "
            f"{np.degrees(mount.geometry.pitch_rad):.1f} deg down, at ({mount.x:.3f}, {mount.y:.3f}); "
            f"{int(coverage.sum())} cells in view"
        )
        return self._model

    # --- one update --------------------------------------------------------------------------------
    def _process(self, msg: Image) -> None:
        if msg.encoding != "mono8":
            raise AdapterError(f"mask encoding {msg.encoding!r}, expected mono8")
        classes = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width]
        _, intr, mount, grid, coverage = self._camera_model(msg.header.frame_id, (msg.height, msg.width))
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        now_ns = self.get_clock().now().nanoseconds
        if not mask_is_fresh(now_ns, stamp_ns, self._max_age):
            self._publish(front_roi_lethal(coverage), msg.header.stamp, mount)
            self.get_logger().warning(
                f"mask stamp {(now_ns - stamp_ns) / 1e9:.3f} s old (limit {self._max_age:g} s): field of view "
                "published lethal", throttle_duration_sec=5.0)
            return
        inputs = CostmapCoreInputs(
            mask=SemanticMaskInput(classes=pool_mask(classes, self._factor), stamp_ns=stamp_ns,
                                   frame_id=msg.header.frame_id, valid=True),
            intrinsics=intr,
            camera_ground=CameraGroundInput(geometry=mount.geometry, camera_frame_id=msg.header.frame_id,
                                            ground_frame_id=_GROUND, stamp_ns=stamp_ns),
            grid=grid,
        )
        # Inflation is Nav2's InflationLayer (it inflates semantic and geometry lethal alike), so 0 here.
        result = run_costmap_pipeline(inputs, inflation_radius=0.0)
        self._publish(costs_to_occupancy(result.final, coverage, self._unknown_occ), msg.header.stamp, mount)
        self._last_good_ns = self.get_clock().now().nanoseconds
        self._failsafe_sent = False

    def _publish(self, occ: np.ndarray, stamp, mount) -> None:
        g = self._model[3].geometry
        m = OccupancyGrid()
        m.header.stamp = stamp
        m.header.frame_id = self._robot_frame
        m.info.map_load_time = stamp
        m.info.resolution = g.resolution
        m.info.width, m.info.height = g.width, g.height
        m.info.origin.position.x = g.origin_x + mount.x
        m.info.origin.position.y = g.origin_y + mount.y
        m.info.origin.orientation.w = 1.0
        m.data = occ.reshape(-1).tolist()
        self._pub.publish(m)

    def _watchdog(self) -> None:
        """No valid mask for max_mask_age_s: publish the fail-safe once (stale is never current, §8.4)."""
        if self._model is None or self._failsafe_sent:
            return
        if self.get_clock().now().nanoseconds - self._last_good_ns > self._max_age * 1e9:
            _, _, mount, _, coverage = self._model
            self._publish(front_roi_lethal(coverage), self.get_clock().now().to_msg(), mount)
            self._failsafe_sent = True
            self.get_logger().warning("no fresh valid mask: field of view published lethal")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SemanticCostmapNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
