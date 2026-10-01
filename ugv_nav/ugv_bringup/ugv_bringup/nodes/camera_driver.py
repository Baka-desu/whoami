"""Camera driver: one V4L2 capture, published for both the system and the UI (no second pipeline).

Publishes (stamp = grab time, frame_id = optical frame; one CameraInfo per image, same stamp):
  /camera/image_raw           sensor_msgs/Image            rgb8 - Dev 1 perception, Dev 2 RTAB-Map
  /camera/camera_info         sensor_msgs/CameraInfo       reliable + transient local - Dev 1, Dev 2, safety
  /image_raw/compressed       sensor_msgs/CompressedImage  jpeg, rate-limited - the web UI (rosbridge)
  /camera_info                sensor_msgs/CameraInfo       same message, with each compressed frame - the web UI
Fails closed: no valid calibration, or a camera whose resolution differs from it, means no frames at all
(a driver that publishes a fake K would corrupt DA3 depth and RTAB-Map geometry).

Params: calibration_file (required) device frame_id fps image_topic info_topic compressed_topic
        ui_info_topic compressed_rate_hz jpeg_quality   (empty compressed_topic disables the UI stream)
"""

from __future__ import annotations

import sys

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from ugv_localization.camera import CalibrationError, load_calibration

from ugv_bringup.camera_core import CaptureError, RatePacer, camera_info_fields, check_capture_size, parse_device

_MAX_FAILED_READS_BEFORE_ERROR = 30


def _live_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=1,
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
    )


def _info_qos() -> QoSProfile:
    # Dev 1 subscribes transient-local, which only matches a transient-local publisher.
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=1,
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class CameraDriver(Node):
    def __init__(self) -> None:
        super().__init__("camera_driver")
        cal_path = self.declare_parameter("calibration_file", "").value
        device = self.declare_parameter("device", "/dev/video0").value
        self._frame_id = self.declare_parameter("frame_id", "camera_optical_frame").value
        fps = float(self.declare_parameter("fps", 30.0).value)
        image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        info_topic = self.declare_parameter("info_topic", "/camera/camera_info").value
        ui_image_topic = self.declare_parameter("compressed_topic", "/image_raw/compressed").value
        ui_info_topic = self.declare_parameter("ui_info_topic", "/camera_info").value
        ui_rate = float(self.declare_parameter("compressed_rate_hz", 5.0).value)
        self._jpeg_quality = int(self.declare_parameter("jpeg_quality", 80).value)

        if not cal_path:
            raise RuntimeError("calibration_file parameter is required (capture one with camera_info_to_yaml)")
        if not fps > 0.0 or not 1 <= self._jpeg_quality <= 100:
            raise RuntimeError("fps must be > 0 and jpeg_quality in 1..100")
        try:
            cal = load_calibration(cal_path)
        except CalibrationError as exc:
            raise RuntimeError(f"refusing to start: {exc}") from exc
        self._info_fields = camera_info_fields(cal)

        try:
            src = parse_device(device)
        except CaptureError as exc:
            raise RuntimeError(str(exc)) from exc
        v4l2 = isinstance(src, int) or str(src).startswith("/dev/")
        self._cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if v4l2 else cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal.height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            check_capture_size(
                cal.width, cal.height,
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        except CaptureError as exc:
            self._cap.release()
            raise RuntimeError(str(exc)) from exc

        self._pub_image = self.create_publisher(Image, image_topic, _live_qos())
        self._pub_info = self.create_publisher(CameraInfo, info_topic, _info_qos())
        self._pub_ui_image = self._pub_ui_info = self._pacer = None
        if ui_image_topic:
            self._pub_ui_image = self.create_publisher(CompressedImage, ui_image_topic, _live_qos())
            self._pub_ui_info = self.create_publisher(CameraInfo, ui_info_topic, _info_qos())
            self._pacer = RatePacer(ui_rate, slack_s=0.5 / fps)
        self._failed = 0
        self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            f"camera {device!r} {cal.width}x{cal.height} @ {fps:g} fps -> {image_topic}, {info_topic}"
            + (f", {ui_image_topic}, {ui_info_topic}" if ui_image_topic else "")
        )

    def _info(self, stamp) -> CameraInfo:
        f = self._info_fields
        m = CameraInfo()
        m.header.stamp = stamp
        m.header.frame_id = self._frame_id
        m.width, m.height = f.width, f.height
        m.distortion_model = f.distortion_model
        m.d, m.k, m.r, m.p = list(f.d), list(f.k), list(f.r), list(f.p)
        return m

    def _tick(self) -> None:
        stamp = self.get_clock().now().to_msg()
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            # Publish nothing: silence is what lets the safety arbiter see a dead camera.
            self._failed += 1
            if self._failed == _MAX_FAILED_READS_BEFORE_ERROR:
                self.get_logger().error("camera is not delivering frames")
            return
        self._failed = 0
        if bgr.shape[1] != self._info_fields.width or bgr.shape[0] != self._info_fields.height:
            self.get_logger().error("frame size changed after start; dropping frames", throttle_duration_sec=5.0)
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self._frame_id
        img.height, img.width = rgb.shape[0], rgb.shape[1]
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = rgb.shape[1] * 3
        img.data = rgb.tobytes()
        self._pub_image.publish(img)
        self._pub_info.publish(self._info(stamp))

        if self._pacer is not None and self._pacer.due(self.get_clock().now().nanoseconds / 1e9):
            ok_jpg, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            if ok_jpg:
                c = CompressedImage()
                c.header.stamp = stamp
                c.header.frame_id = self._frame_id
                c.format = "jpeg"
                c.data = jpg.tobytes()
                self._pub_ui_image.publish(c)
                self._pub_ui_info.publish(self._info(stamp))

    def close(self) -> None:
        self._cap.release()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = CameraDriver()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        print(f"camera_driver: {exc}", file=sys.stderr)
        code = 1
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
