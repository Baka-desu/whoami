"""Every ugv_localization node, run as its real entry point in a subprocess and driven over ROS.

Each test gets its own ROS_DOMAIN_ID, publishes synthetic inputs from an in-process probe node,
and asserts on what the node publishes / writes / exits with. Skipped without ROS 2 (runs under
colcon test in WSL Lyrical). Synthetic data here is test input only — never a product calibration.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("rtabmap_msgs")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from rtabmap_msgs.msg import Info  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from std_srvs.srv import Empty  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from ugv_localization.camera import load_calibration  # noqa: E402

_PKG = Path(__file__).resolve().parents[1]
_CFG = _PKG / "config"
_DOMAINS = itertools.count(40 + (os.getpid() % 40))
_COV = [0.01 if i in (0, 7, 14, 21, 28, 35) else 0.0 for i in range(36)]
_LOST = [9999.0 if i in (0, 7, 14, 21, 28, 35) else 0.0 for i in range(36)]
_K = [500.0, 0.0, 32.0, 0.0, 500.0, 24.0, 0.0, 0.0, 1.0]
_LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
_W, _H, _FRAME = 64, 48, "camera_optical"


# ------------------------------------------------------------------------------ harness
class Harness:
    def __init__(self, domain: int) -> None:
        self.domain = domain
        self.ctx = rclpy.Context()
        rclpy.init(context=self.ctx, domain_id=domain)
        self.node = rclpy.create_node("probe", context=self.ctx)
        self.executor = rclpy.executors.SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.node)
        self.procs: list[subprocess.Popen] = []

    def start(self, module: str, *args: str) -> subprocess.Popen:
        env = {**os.environ, "ROS_DOMAIN_ID": str(self.domain), "PYTHONPATH": f"{_PKG}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        code = f"import sys; from {module} import main; main(sys.argv[1:])"
        p = subprocess.Popen([sys.executable, "-c", code, *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p

    def spin(self, seconds: float, *, until=None, each=None, period: float = 0.05) -> bool:
        end = time.monotonic() + seconds
        nxt = time.monotonic()
        while time.monotonic() < end:
            if each is not None and time.monotonic() >= nxt:
                each()
                nxt = time.monotonic() + period
            self.executor.spin_once(timeout_sec=0.01)
            if until is not None and until():
                return True
        return until is None

    def now(self):
        return self.node.get_clock().now().to_msg()

    def close(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown(context=self.ctx)


@pytest.fixture
def h():
    harness = Harness(next(_DOMAINS))
    yield harness
    harness.close()


def _output(p: subprocess.Popen) -> str:
    return p.stdout.read() if p.stdout else ""


def _odom(h: Harness, frame: str, x: float, cov=_COV) -> Odometry:
    m = Odometry()
    m.header.stamp = h.now()
    m.header.frame_id = frame
    m.child_frame_id = "base_link"
    m.pose.pose.position.x = x
    m.pose.pose.orientation.w = 1.0
    m.pose.covariance = list(cov)
    return m


def _cinfo(h: Harness, stamp=None, k=_K) -> CameraInfo:
    m = CameraInfo()
    m.header.stamp = stamp or h.now()
    m.header.frame_id = _FRAME
    m.width, m.height = _W, _H
    m.distortion_model = "plumb_bob"
    m.d = [0.0] * 5
    m.k = list(k)
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.p = [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
    return m


def _depth(h: Harness, value: float = 3.0, encoding: str = "32FC1", stamp=None) -> Image:
    m = Image()
    m.header.stamp = stamp or h.now()
    m.header.frame_id = _FRAME
    m.height, m.width = _H, _W
    m.encoding = encoding
    m.is_bigendian = 0
    if encoding == "32FC1":
        arr = np.full((_H, _W), value, dtype="<f4")
        arr[:, : _W // 5] = np.nan  # 20 % holes (sky)
        m.step, m.data = 4 * _W, arr.tobytes()
    else:
        m.step, m.data = 2 * _W, bytes(2 * _W * _H)
    return m


def _tf(h: Harness, parent: str, child: str, x: float = 0.0, stamp=None) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = stamp or h.now()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x = x
    t.transform.rotation.w = 1.0
    return t


# ------------------------------------------------------------------------------ odom_selector
def _selector(h: Harness, source: str) -> subprocess.Popen:
    return h.start(
        "ugv_localization.nodes.odom_selector",
        "--ros-args", "-p", f"profile_path:={_CFG / 'odom_select.yaml'}", "-p", f"odom_source:={source}",
        # same remaps as localization.launch.py
        "-r", "wheel/odom:=/wheel/odom", "-r", "odom_visual:=/rtabmap/odom_visual", "-r", "odom:=/odom",
    )


def test_n1_selector_wheel_to_odom_tf_and_source(h: Harness) -> None:
    _selector(h, "auto")
    pub = h.node.create_publisher(Odometry, "/wheel/odom", 20)
    outs: list[Odometry] = []
    tfs: list[TransformStamped] = []
    src: list[str] = []
    h.node.create_subscription(Odometry, "/odom", outs.append, 20)
    h.node.create_subscription(TFMessage, "/tf", lambda m: tfs.extend(m.transforms), 50)
    h.node.create_subscription(String, "/ugv/localization/odom_source", lambda m: src.append(m.data), _LATCHED)
    sent: list[Odometry] = []

    def each() -> None:
        m = _odom(h, "odom", 1.5)
        sent.append(m)
        pub.publish(m)

    assert h.spin(15.0, each=each, until=lambda: len(outs) >= 5 and tfs and src)
    o = outs[-1]
    assert (o.header.frame_id, o.child_frame_id) == ("odom", "base_link")
    assert o.pose.pose.position.x == pytest.approx(1.5)
    assert (o.header.stamp.sec, o.header.stamp.nanosec) in {(s.header.stamp.sec, s.header.stamp.nanosec) for s in sent}
    t = tfs[-1]
    assert (t.header.frame_id, t.child_frame_id) == ("odom", "base_link")
    assert src[-1] == "wheel"


def test_n2_selector_visual_lost_dropped_then_recovers(h: Harness) -> None:
    _selector(h, "visual")
    pub = h.node.create_publisher(Odometry, "/rtabmap/odom_visual", 20)
    outs: list[Odometry] = []
    h.node.create_subscription(Odometry, "/odom", outs.append, 20)
    h.spin(6.0, each=lambda: pub.publish(_odom(h, "odom_visual", 0.0, _LOST)))
    assert outs == []  # rgbd_odometry lost marker never reaches /odom
    assert h.spin(10.0, each=lambda: pub.publish(_odom(h, "odom_visual", 2.0)), until=lambda: len(outs) >= 3)
    assert outs[-1].pose.pose.position.x == pytest.approx(2.0)


def test_n3_selector_rejects_wrong_frame(h: Harness) -> None:
    _selector(h, "wheel")
    pub = h.node.create_publisher(Odometry, "/wheel/odom", 20)
    outs: list[Odometry] = []
    h.node.create_subscription(Odometry, "/odom", outs.append, 20)
    h.spin(6.0, each=lambda: pub.publish(_odom(h, "world", 1.0)))
    assert outs == []


# ------------------------------------------------------------------------------ pose_validity
def _validity(h: Harness) -> subprocess.Popen:
    return h.start(
        "ugv_localization.nodes.pose_validity_node",
        "--ros-args", "-p", f"profile_path:={_CFG / 'pose_validity.yaml'}", "-p", "mode:=mapping",
    )


def test_n4_validity_heartbeat_and_depth_contract(h: Harness) -> None:
    _validity(h)
    valid: list[bool] = []
    status: list[str] = []
    h.node.create_subscription(Bool, "/ugv/pose_valid", lambda m: valid.append(m.data), 50)
    h.node.create_subscription(String, "/ugv/localization_status", lambda m: status.append(m.data), 10)
    cam = h.node.create_publisher(CameraInfo, "camera_info", 10)
    dep = h.node.create_publisher(Image, "depth", 10)
    assert h.spin(15.0, until=lambda: len(valid) > 0)  # node up
    n0 = len(valid)
    h.spin(1.0)
    assert len(valid) - n0 >= 15  # 20 Hz heartbeat (contract: Dev 5 watchdog)
    assert not any(valid)  # fail closed at startup

    def bad() -> None:
        s = h.now()
        cam.publish(_cinfo(h, s))
        dep.publish(_depth(h, encoding="16UC1", stamp=s))

    assert h.spin(8.0, each=bad, until=lambda: any("depth_invalid" in s for s in status))


def test_n5_validity_goes_true_with_all_inputs(h: Harness) -> None:
    proc = _validity(h)
    valid: list[bool] = []
    status: list[str] = []
    h.node.create_subscription(Bool, "/ugv/pose_valid", lambda m: valid.append(m.data), 50)
    h.node.create_subscription(String, "/ugv/localization_status", lambda m: status.append(m.data), 10)
    pubs = {
        "odom": h.node.create_publisher(Odometry, "odom", 10),
        "cam": h.node.create_publisher(CameraInfo, "camera_info", 10),
        "depth": h.node.create_publisher(Image, "depth", 10),
        "info": h.node.create_publisher(Info, "rtabmap/info", 10),
        "tf": h.node.create_publisher(TFMessage, "/tf", 50),
    }

    send_depth = [True]

    def each() -> None:
        s = h.now()
        pubs["tf"].publish(TFMessage(transforms=[_tf(h, "map", "odom", stamp=s), _tf(h, "odom", "base_link", 0.2, s)]))
        o = _odom(h, "odom", 0.2)
        o.header.stamp = s
        pubs["odom"].publish(o)
        pubs["cam"].publish(_cinfo(h, s))
        if send_depth[0]:
            pubs["depth"].publish(_depth(h, stamp=s))
        info = Info()
        info.header.stamp = s
        pubs["info"].publish(info)

    ok = h.spin(20.0, each=each, until=lambda: bool(valid) and valid[-1] and "valid" in status)
    assert ok, f"never valid; statuses={status}"
    assert "recovering" in status  # passed through the recover_hold_s hysteresis first
    # Regression: the node used to crash on its first valid→log (rclpy logger severity switch).
    h.spin(1.0, each=each)
    assert proc.poll() is None, _output(proc)
    send_depth[0] = False  # Dev 1 depth stops → fail closed, node still alive
    assert h.spin(5.0, each=each, until=lambda: not valid[-1] and any("depth_stale" in s for s in status[-2:])), status
    assert proc.poll() is None


# ------------------------------------------------------------------------------ depth_eval
def test_n6_depth_eval_pairs_by_stamp_and_reports(h: Harness, tmp_path: Path) -> None:
    p = h.start(
        "ugv_localization.nodes.depth_eval",
        "--ros-args", "-p", f"out_dir:={tmp_path}", "-p", "duration_s:=1.0", "-p", "stride:=2",
    )
    est = h.node.create_publisher(Image, "depth", 10)
    gt = h.node.create_publisher(Image, "gt_depth", 10)

    def each() -> None:
        s = h.now()
        gt.publish(_depth(h, 3.0, stamp=s))
        est.publish(_depth(h, 3.3, stamp=s))  # DA3 10 % long
        gt.publish(_depth(h, 3.0))  # unmatched GT frame: must not be paired

    assert h.spin(20.0, each=each, period=0.1, until=lambda: p.poll() is not None), _output(p)
    rep = json.loads((tmp_path / "depth_report.json").read_text(encoding="utf-8"))
    assert rep["frames"] >= 5
    assert rep["abs_rel"] == pytest.approx(0.1, rel=1e-4)
    assert rep["median_scale_gt_over_est"] == pytest.approx(3.0 / 3.3, rel=1e-4)


# ------------------------------------------------------------------------------ drift_eval
def test_n7_drift_eval_perfect_estimate(h: Harness, tmp_path: Path) -> None:
    p = h.start(
        "ugv_localization.nodes.drift_eval",
        "--ros-args", "-p", f"out_dir:={tmp_path}", "-p", "duration_s:=3.0", "-r", "ground_truth:=/ground_truth/odom",
    )
    gt = h.node.create_publisher(Odometry, "/ground_truth/odom", 20)
    tf = h.node.create_publisher(TFMessage, "/tf", 50)
    t0 = time.monotonic()

    def each() -> None:
        s = h.now()
        x = 0.5 * (time.monotonic() - t0)
        tf.publish(TFMessage(transforms=[_tf(h, "map", "base_link", x, s)]))
        o = _odom(h, "map", x)
        o.header.stamp = s
        gt.publish(o)

    assert h.spin(25.0, each=each, until=lambda: p.poll() is not None), _output(p)
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert rep["matched"] >= 20
    assert rep["ate_rmse_m"] == pytest.approx(0.0, abs=1e-6)
    assert rep["with_scale"] is False


# ------------------------------------------------------------------------------ tf_rate_check
@pytest.mark.parametrize("rate_hz, code", [(20.0, 0), (5.0, 1)])
def test_n8_tf_rate_check_exit_code(h: Harness, rate_hz: float, code: int) -> None:
    p = h.start("ugv_localization.nodes.tf_rate_check", "--ros-args", "-p", "duration_s:=2.0")
    tf = h.node.create_publisher(TFMessage, "/tf", 50)

    def each() -> None:
        s = h.now()
        tf.publish(TFMessage(transforms=[_tf(h, "map", "odom", stamp=s), _tf(h, "odom", "base_link", stamp=s)]))

    assert h.spin(20.0, each=each, period=1.0 / rate_hz, until=lambda: p.poll() is not None), _output(p)
    assert p.returncode == code, _output(p)


# ------------------------------------------------------------------------------ mode_cli
def test_n9_mode_cli_calls_rtabmap_service(h: Harness) -> None:
    calls: list[str] = []

    def handler(req, resp):
        calls.append("localize")
        return resp

    h.node.create_service(Empty, "/rtabmap/rtabmap/set_mode_localization", handler)
    p = h.start("ugv_localization.nodes.mode_cli", "localize", "--timeout", "10")
    assert h.spin(20.0, until=lambda: p.poll() is not None)
    assert p.returncode == 0 and calls == ["localize"], _output(p)


def test_n10_mode_cli_fails_without_service(h: Harness) -> None:
    p = h.start("ugv_localization.nodes.mode_cli", "backup", "--timeout", "1")
    assert h.spin(15.0, until=lambda: p.poll() is not None)
    assert p.returncode == 1
    assert "not available" in _output(p)


# ------------------------------------------------------------------------------ camera_info_to_yaml
def test_n11_camera_info_to_yaml_writes_loadable_file(h: Harness, tmp_path: Path) -> None:
    out = tmp_path / "cams" / "sim_front.yaml"
    p = h.start("ugv_localization.nodes.camera_info_to_yaml", "--topic", "/camera/camera_info", "--name", "sim_front", "--out", str(out))
    pub = h.node.create_publisher(CameraInfo, "/camera/camera_info", 10)
    assert h.spin(20.0, each=lambda: pub.publish(_cinfo(h)), until=lambda: p.poll() is not None)
    assert p.returncode == 0, _output(p)
    cal = load_calibration(out)
    assert (cal.camera_name, cal.width, cal.height) == ("sim_front", _W, _H)
    assert "Do not hand-edit" in out.read_text(encoding="utf-8")


def test_n12_camera_info_to_yaml_refuses_fake_k_and_overwrite(h: Harness, tmp_path: Path) -> None:
    out = tmp_path / "c.yaml"
    p = h.start("ugv_localization.nodes.camera_info_to_yaml", "--name", "c", "--out", str(out), "--timeout", "10")
    pub = h.node.create_publisher(CameraInfo, "/camera/camera_info", 10)
    assert h.spin(20.0, each=lambda: pub.publish(_cinfo(h, k=[0.0] * 9)), until=lambda: p.poll() is not None)
    assert p.returncode == 1 and "refusing" in _output(p) and not out.exists()
    out.write_text("existing", encoding="utf-8")
    p2 = h.start("ugv_localization.nodes.camera_info_to_yaml", "--name", "c", "--out", str(out))
    assert h.spin(10.0, until=lambda: p2.poll() is not None)
    assert p2.returncode == 1 and "--force" in _output(p2)
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == "existing"
