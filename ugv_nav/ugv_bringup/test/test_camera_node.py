"""The camera driver over real ROS 2 topics. Skipped without ROS 2 / OpenCV (runs under colcon test).

The "camera" is a generated video file (OpenCV opens it exactly like a device), and the calibration is a
test fixture - never a product calibration.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml

rclpy = pytest.importorskip("rclpy")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CameraInfo, CompressedImage, Image  # noqa: E402

from ugv_localization.camera import calibration_to_yaml_dict  # noqa: E402
from test_camera_core import fixture_calibration  # noqa: E402


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "cam.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))
    for i in range(300):
        frame = np.full((480, 640, 3), (i % 255, 80, 200 - i % 200), np.uint8)
        w.write(frame)
    w.release()
    return path


def write_cal(tmp_path, **kw) -> Path:
    path = tmp_path / "cal.yaml"
    path.write_text(yaml.safe_dump(calibration_to_yaml_dict(fixture_calibration(**kw))), encoding="utf-8")
    return path


def start(args: list[str]):
    os.environ["ROS_DOMAIN_ID"] = str(90 + os.getpid() % 9)
    rclpy.init(args=["--ros-args", *args])
    from ugv_bringup.nodes.camera_driver import CameraDriver

    return CameraDriver()


def test_publishes_system_and_ui_topics_from_one_capture(tmp_path, video):
    cal = write_cal(tmp_path)
    node = start(["-p", f"calibration_file:={cal}", "-p", f"device:={video}", "-p", "fps:=30.0", "-p", "compressed_rate_hz:=10.0"])
    probe = rclpy.create_node("camera_probe")
    got: dict[str, list] = {"img": [], "info": [], "ui_img": [], "ui_info": []}
    live = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE)
    latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    probe.create_subscription(Image, "/camera/image_raw", got["img"].append, live)
    probe.create_subscription(CameraInfo, "/camera/camera_info", got["info"].append, latched)
    probe.create_subscription(CompressedImage, "/image_raw/compressed", got["ui_img"].append, live)
    probe.create_subscription(CameraInfo, "/camera_info", got["ui_info"].append, latched)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    try:
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.02)

        assert got["img"] and got["info"] and got["ui_img"] and got["ui_info"]
        img, info = got["img"][-1], got["info"][-1]
        assert img.encoding == "rgb8" and (img.width, img.height) == (640, 480)
        assert img.header.frame_id == info.header.frame_id == "camera_optical_frame"
        assert info.k[0] == 500.0 and info.k[2] == 320.0 and (info.width, info.height) == (640, 480)
        # one CameraInfo per image, identical stamp (rgbd_sync pairs by exact stamp)
        stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in got["img"]}
        assert (info.header.stamp.sec, info.header.stamp.nanosec) in stamps
        # UI stream: jpeg that decodes, with its own CameraInfo of the same K and a matching stamp
        ui = got["ui_img"][-1]
        assert ui.format == "jpeg"
        decoded = cv2.imdecode(np.frombuffer(bytes(ui.data), np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape[:2] == (480, 640)
        ui_info = got["ui_info"][-1]
        assert ui_info.k[0] == 500.0
        assert len(got["ui_img"]) < len(got["img"])  # rate limited
        assert probe.count_publishers("/camera/image_raw") == 1  # a single pipeline
    finally:
        ex.shutdown()
        node.close()
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_refuses_to_start_without_a_valid_calibration(tmp_path, video):
    bad = calibration_to_yaml_dict(fixture_calibration())
    bad["camera_matrix"]["data"] = [0.0] * 9
    path = tmp_path / "zero.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to start"):
        try:
            start(["-p", f"calibration_file:={path}", "-p", f"device:={video}"])
        finally:
            rclpy.shutdown()


def test_refuses_a_camera_whose_resolution_differs_from_the_calibration(tmp_path, video):
    cal = write_cal(tmp_path, width=1280, height=720,
                    k=(900.0, 0.0, 640.0, 0.0, 900.0, 360.0, 0.0, 0.0, 1.0),
                    p=(900.0, 0.0, 640.0, 0.0, 0.0, 900.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    with pytest.raises(RuntimeError, match="calibration is for 1280x720"):
        try:
            start(["-p", f"calibration_file:={cal}", "-p", f"device:={video}"])
        finally:
            rclpy.shutdown()
