"""Live rclpy spin of PerceptionAdapterNode with fixture Image+CameraInfo."""

from __future__ import annotations

import json
import time
from collections import deque

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("sensor_msgs")
pytest.importorskip("std_msgs")
pytest.importorskip("builtin_interfaces")

from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Header, String

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from ugv_perception.ingest.ros_bridge import camera_info_msg_to_view, image_msg_to_view
from ugv_perception.node.adapter_node import PerceptionAdapterNode, camera_info_qos
from ugv_perception.node.metrics import STAGES, PerceptionMetrics
from ugv_perception.tests.test_node_cycle import SpyAdapter

_STAMP = 2_000_000_000
_K = (400.0, 0.0, 1.0, 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)


def _msgs() -> tuple[Image, CameraInfo]:
    t = Time()
    t.sec = _STAMP // 1_000_000_000
    t.nanosec = _STAMP % 1_000_000_000
    img = Image()
    img.header = Header(stamp=t, frame_id="camera_optical")
    img.height = 2
    img.width = 2
    img.encoding = "rgb8"
    img.step = 6
    img.data = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3])
    info = CameraInfo()
    info.header = Header(stamp=t, frame_id="camera_optical")
    info.height = 2
    info.width = 2
    info.k = list(_K)
    return img, info


def test_adapter_node_spin_fixture_topics() -> None:
    rclpy.init()
    spy = SpyAdapter()
    node = PerceptionAdapterNode(
        adapter=spy,
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 100_000_000,
    )
    helper = Node("test_cam_pub")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    masks: list[Image] = []
    flags: list[bool] = []
    helper.create_subscription(Image, "/segmentation/mask", masks.append, 10)
    helper.create_subscription(Bool, "/ugv/perception_degraded", lambda m: flags.append(m.data), 10)
    img, info = _msgs()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        for _ in range(80):
            pub_i.publish(img)
            pub_c.publish(info)
            ex.spin_once(timeout_sec=0.05)
            if spy.calls >= 1 and masks and flags:
                break
        assert spy.calls >= 1
        assert masks, "expected /segmentation/mask from adapter_node"
        assert masks[0].header.frame_id == "camera_optical"
        assert masks[0].header.stamp.sec == _STAMP // 1_000_000_000
        assert False in flags or True in flags
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


_MS = 1_000_000
_STAGE_KEYS = {"decode", "seg", "depth_infer", "depth_post", "cloud", "publish"}


class _Clock:
    """Injected monotonic clock. Time moves only when a test says so."""

    def __init__(self) -> None:
        self.ns = 1_000_000_000

    def __call__(self) -> int:
        return self.ns

    def advance_ms(self, ms: float) -> None:
        self.ns += int(ms * _MS)


class _TimedAdapter(SpyAdapter):
    def __init__(self, clock: _Clock) -> None:
        super().__init__()
        self._clock = clock

    def infer(self, frame):
        self._clock.advance_ms(20)
        return super().infer(frame)


class _FakeDepth:
    """Depth channel double honouring the optional timing hook."""

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self.calls = 0

    def maps(self, rgb, k, stage=None):
        self.calls += 1
        with stage("depth_infer"):
            self._clock.advance_ms(30)
        with stage("depth_post"):
            self._clock.advance_ms(4)
        return np.ones((2, 2), dtype=np.float32), np.zeros((4, 3), dtype=np.float32)


class _RaisingDepth:
    def maps(self, rgb, k, stage=None):
        raise RuntimeError("boom")


class _SpyLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.warnings.append(msg)


def test_latencies_ns_is_a_bounded_deque() -> None:
    m = PerceptionMetrics()
    assert isinstance(m.latencies_ns, deque)
    assert m.latencies_ns.maxlen == 600
    for i in range(1000):
        m.latencies_ns.append(i)
    assert len(m.latencies_ns) == 600
    assert m.latencies_ns[0] == 400
    assert m.latencies_ns[-1] == 999


def test_stage_timers_accumulate_per_frame_with_injected_clock() -> None:
    clock = _Clock()
    m = PerceptionMetrics(clock_ns=clock)
    # frame 1: seg 20 ms; publish is called twice in the frame and sums to 5 ms
    with m.stage("seg"):
        clock.advance_ms(20)
    with m.stage("publish"):
        clock.advance_ms(3)
    with m.stage("publish"):
        clock.advance_ms(2)
    m.end_frame()
    # frame 2: seg 40 ms, no publish
    with m.stage("seg"):
        clock.advance_ms(40)
    m.end_frame()
    stage_ms = m.snapshot()["stage_ms"]
    assert set(stage_ms) == _STAGE_KEYS == set(STAGES)
    assert stage_ms["seg"] == pytest.approx(30.0)
    assert stage_ms["publish"] == pytest.approx(5.0)
    assert stage_ms["decode"] == 0.0


def test_stage_timer_records_a_span_that_raises_and_rejects_unknown_names() -> None:
    clock = _Clock()
    m = PerceptionMetrics(clock_ns=clock)
    with pytest.raises(RuntimeError):
        with m.stage("depth_infer"):
            clock.advance_ms(7)
            raise RuntimeError("model fell over")
    m.end_frame()
    assert m.snapshot()["stage_ms"]["depth_infer"] == pytest.approx(7.0)
    with pytest.raises(ValueError):
        with m.stage("infer"):
            pass


def test_rates_and_stage_means_cover_a_recent_window_not_the_lifetime() -> None:
    clock = _Clock()
    m = PerceptionMetrics(clock_ns=clock)
    for _ in range(50):
        clock.advance_ms(100)
        with m.stage("seg"):
            clock.advance_ms(0)
        m.mark_mask()
        m.mark_depth()
        m.end_frame()
    snap = m.snapshot()
    assert snap["mask_hz"] == pytest.approx(10.0)
    assert snap["depth_hz"] == pytest.approx(10.0)
    clock.advance_ms(10_000)
    snap = m.snapshot()
    assert snap["mask_hz"] == 0.0
    assert snap["depth_hz"] == 0.0
    assert snap["stage_ms"]["seg"] == 0.0
    assert m.masks_published == 50


def test_depth_channel_timing_hook_sees_both_stages_and_changes_nothing() -> None:
    from contextlib import contextmanager

    from ugv_perception.backend.depth_live import DepthChannel

    class _Backend:
        def ensure_hw(self, h: int, w: int) -> None:
            pass

        def run_all(self, blob):
            hw = blob.shape[2:]
            return [np.full(hw, 0.5, dtype=np.float32), np.zeros(hw, dtype=np.float32)]

    seen: list[str] = []

    @contextmanager
    def hook(name: str):
        seen.append(name)
        yield

    rgb = np.full((28, 28, 3), 100, dtype=np.uint8)
    k = (200.0, 0.0, 13.5, 0.0, 200.0, 13.5, 0.0, 0.0, 1.0)
    plain = DepthChannel(_Backend()).maps(rgb, k)
    timed = DepthChannel(_Backend()).maps(rgb, k, stage=hook)
    assert seen == ["depth_infer", "depth_post"]
    assert np.array_equal(plain[0], timed[0], equal_nan=True)
    assert np.array_equal(plain[1], timed[1])


def _spin_until(ex, pub_i, pub_c, until, timeout_s: float = 10.0) -> None:
    """Feed one frame per pass until `until()`. Bounded by wall time, not pass count."""
    img, info = _msgs()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pub_i.publish(img)
        pub_c.publish(info)
        ex.spin_once(timeout_sec=0.05)
        if until():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached while spinning")


def test_depth_failure_is_counted_and_never_degrades_the_mask() -> None:
    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=SpyAdapter(),
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 100_000_000,
        depth=_RaisingDepth(),
    )
    helper = Node("test_depth_fail_pub")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    masks: list[Image] = []
    flags: list[bool] = []
    helper.create_subscription(Image, "/segmentation/mask", masks.append, 10)
    helper.create_subscription(Bool, "/ugv/perception_degraded", lambda m: flags.append(m.data), 10)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        _spin_until(ex, pub_i, pub_c, lambda: node.metrics.depth_errors >= 1)
        true_after_first_error = node.metrics.degraded_true
        _spin_until(
            ex, pub_i, pub_c,
            lambda: node.metrics.depth_errors >= 4 and masks and flags and flags[-1] is False,
        )
        assert node.metrics.last_depth_error == "RuntimeError: boom"
        assert masks, "a failing depth channel must not stop the mask"
        assert node.metrics.degraded_true == true_after_first_error, (
            "a failing depth channel must not degrade perception"
        )
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_depth_failure_logs_at_warn_first_then_every_100th() -> None:
    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=SpyAdapter(),
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 100_000_000,
        depth=_RaisingDepth(),
    )
    try:
        img, info = _msgs()
        node._last_image = image_msg_to_view(img)
        node._last_info = camera_info_msg_to_view(info)
        spy = _SpyLogger()
        node.get_logger = lambda: spy
        for _ in range(205):
            node._publish_depth()
        assert node.metrics.depth_errors == 205
        assert len(spy.warnings) == 3, spy.warnings
        assert all("RuntimeError: boom" in w for w in spy.warnings)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_stats_topic_publishes_stage_times_rates_and_depth_errors() -> None:
    rclpy.init()
    clock = _Clock()
    node = PerceptionAdapterNode(
        adapter=_TimedAdapter(clock),
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 100_000_000,
        monotonic_ns_fn=clock,
        depth=_FakeDepth(clock),
    )
    helper = Node("test_stats_pub")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    stats: list[dict] = []
    helper.create_subscription(
        String, "/ugv/perception/stats", lambda m: stats.append(json.loads(m.data)), 10
    )
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        # Warm up, then jump the injected clock past the 5 s window so the start-up tick
        # (CameraInfo not yet seen, no inference) is no longer in the recent window.
        _spin_until(ex, pub_i, pub_c, lambda: node.metrics.infer_calls >= 3)
        clock.advance_ms(10_000)
        _spin_until(
            ex, pub_i, pub_c,
            lambda: any(s["stage_ms"]["seg"] == pytest.approx(20.0) for s in stats),
        )
        got = next(s for s in stats if s["stage_ms"]["seg"] == pytest.approx(20.0))
        assert set(got) == {"mask_hz", "depth_hz", "stage_ms", "depth_errors", "last_depth_error"}
        assert set(got["stage_ms"]) == _STAGE_KEYS
        assert got["stage_ms"]["seg"] == pytest.approx(20.0)
        assert got["stage_ms"]["depth_infer"] == pytest.approx(30.0)
        assert got["stage_ms"]["depth_post"] == pytest.approx(4.0)
        assert got["mask_hz"] > 0.0
        assert got["depth_hz"] > 0.0
        assert got["depth_errors"] == 0
        assert got["last_depth_error"] is None
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
