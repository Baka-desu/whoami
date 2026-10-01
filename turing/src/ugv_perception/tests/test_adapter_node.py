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
from ugv_perception.node.cycle import decode_cycle_frame
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
        info_view = camera_info_msg_to_view(info)
        frame = decode_cycle_frame(image_msg_to_view(img), info_view)
        spy = _SpyLogger()
        node.get_logger = lambda: spy
        for _ in range(205):
            node._publish_depth(frame, info_view.k)
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


class _RecordingAdapter(SpyAdapter):
    """Keeps the pixels it was given, so a test can tell whether depth got the very same array."""

    def __init__(self) -> None:
        super().__init__()
        self.rgbs: list[np.ndarray] = []

    def infer(self, frame):
        self.rgbs.append(frame.rgb)
        return super().infer(frame)


class _RecordingDepth(_FakeDepth):
    def __init__(self, clock: _Clock) -> None:
        super().__init__(clock)
        self.rgbs: list[np.ndarray] = []

    def maps(self, rgb, k, stage=None):
        self.rgbs.append(rgb)
        return super().maps(rgb, k, stage)


def _node_with_one_pair(clock: _Clock, adapter, depth, *, now_ns: int):
    """A node that has seen CameraInfo, ready for `_on_image`. No executor, no wall-clock waits."""
    node = PerceptionAdapterNode(
        adapter=adapter,
        adapter_id="yoloe",
        now_ns_fn=lambda: now_ns,
        monotonic_ns_fn=clock,
        depth=depth,
    )
    img, info = _msgs()
    node._on_info(info)
    return node, img


def test_depth_is_published_when_the_mask_is_stale_and_not_published() -> None:
    rclpy.init()
    clock = _Clock()
    depth = _FakeDepth(clock)
    spy = SpyAdapter()
    node = PerceptionAdapterNode(
        adapter=spy,
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 10_000_000_000,  # the image is 10 s old: stale, no mask
        monotonic_ns_fn=clock,
        depth=depth,
    )
    helper = Node("test_depth_without_mask")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    masks: list[Image] = []
    depths: list[Image] = []
    flags: list[bool] = []
    helper.create_subscription(Image, "/segmentation/mask", masks.append, 10)
    helper.create_subscription(Image, "/perception/depth/image", depths.append, 10)
    helper.create_subscription(Bool, "/ugv/perception_degraded", lambda m: flags.append(m.data), 10)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        _spin_until(ex, pub_i, pub_c, lambda: bool(depths) and bool(flags))
        assert depths[0].header.stamp.sec == _STAMP // 1_000_000_000
        assert depths[0].header.frame_id == "camera_optical"
        assert not masks, "a stale frame must not produce a mask"
        assert spy.calls == 0, "a stale frame must not reach the adapter"
        assert flags and all(flags), "stale perception must still be degraded"
        assert node.metrics.masks_published == 0
        assert node.metrics.depth_errors == 0
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_depth_still_runs_when_the_adapter_fails_and_the_mask_does_not() -> None:
    from ugv_perception.adapter.output import AdapterError

    class _FailingAdapter:
        def infer(self, frame):
            raise AdapterError("model fell over")

    rclpy.init()
    clock = _Clock()
    depth = _FakeDepth(clock)
    node, img = _node_with_one_pair(
        clock, _FailingAdapter(), depth, now_ns=_STAMP + 100_000_000
    )
    try:
        node._on_image(img)
        assert depth.calls == 1
        assert node.metrics.masks_published == 0
        assert node.metrics.degraded_true == 1
        assert node.metrics.depth_errors == 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_depth_does_not_run_when_the_frame_cannot_be_decoded() -> None:
    """Nothing decoded means nothing to run depth on. That is the mask's degraded case, not a depth error."""
    rclpy.init()
    clock = _Clock()
    depth = _FakeDepth(clock)
    node = PerceptionAdapterNode(
        adapter=SpyAdapter(),
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 100_000_000,
        monotonic_ns_fn=clock,
        depth=depth,
    )
    try:
        img, info = _msgs()
        info.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # the reserved identity K: rejected
        node._on_info(info)
        node._on_image(img)
        assert depth.calls == 0
        assert node.metrics.depth_errors == 0
        assert node.metrics.degraded_true == 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_one_frame_is_decoded_once_and_depth_gets_the_cycle_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ugv_perception.ingest.decode as decode_mod
    import ugv_perception.node.cycle as cycle_mod

    calls: list[int] = []
    real = decode_mod.decode_frame

    def counting(image, info):
        calls.append(1)
        return real(image, info)

    monkeypatch.setattr(decode_mod, "decode_frame", counting)
    monkeypatch.setattr(cycle_mod, "decode_frame", counting)
    rclpy.init()
    clock = _Clock()
    adapter = _RecordingAdapter()
    depth = _RecordingDepth(clock)
    node, img = _node_with_one_pair(clock, adapter, depth, now_ns=_STAMP + 100_000_000)
    try:
        node._on_image(img)
        assert len(calls) == 1, f"decoded {len(calls)} times for one frame"
        assert len(adapter.rgbs) == 1 and len(depth.rgbs) == 1
        assert depth.rgbs[0] is adapter.rgbs[0], "depth must reuse the frame the cycle decoded"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_stage_attribution_decode_is_one_frame_decode_and_seg_excludes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """decode = ROS message to view plus the one frame decode. seg = the cycle without decoding."""
    import ugv_perception.ingest.decode as decode_mod
    import ugv_perception.node.cycle as cycle_mod

    rclpy.init()
    clock = _Clock()
    real = decode_mod.decode_frame

    def slow_decode(image, info):
        clock.advance_ms(7)
        return real(image, info)

    monkeypatch.setattr(decode_mod, "decode_frame", slow_decode)
    monkeypatch.setattr(cycle_mod, "decode_frame", slow_decode)
    node, img = _node_with_one_pair(
        clock, _TimedAdapter(clock), _FakeDepth(clock), now_ns=_STAMP + 100_000_000
    )
    try:
        node._on_image(img)
        stage_ms = node.metrics.snapshot()["stage_ms"]
        assert stage_ms["decode"] == pytest.approx(7.0)
        assert stage_ms["seg"] == pytest.approx(20.0)
        assert stage_ms["depth_infer"] == pytest.approx(30.0)
        assert stage_ms["depth_post"] == pytest.approx(4.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# --- Segmentation scheduling: depth on every frame, a mask at least every mask_period_s. ----------------


def _image_at(stamp_ns: int) -> Image:
    img, _ = _msgs()
    img.header.stamp.sec = stamp_ns // 1_000_000_000
    img.header.stamp.nanosec = stamp_ns % 1_000_000_000
    return img


def _capture(node: PerceptionAdapterNode) -> dict[str, list]:
    """Record what the node publishes, by wrapping the publishers it already has."""
    got: dict[str, list] = {"mask": [], "degraded": [], "depth": []}
    for key, pub in (
        ("mask", node._pub_mask),
        ("degraded", node._pub_degraded),
        ("depth", node._pub_depth),
    ):
        original = pub.publish

        def publish(msg, _key=key, _original=original) -> None:
            got[_key].append(msg)
            _original(msg)

        pub.publish = publish
    return got


class _Wall:
    """Injected wall clock for freshness. A test sets it relative to the stamp of the frame it feeds."""

    def __init__(self) -> None:
        self.ns = _STAMP + 100_000_000

    def __call__(self) -> int:
        return self.ns


def _scheduled_node(clock: _Clock, wall: _Wall, adapter, depth, **kwargs):
    node = PerceptionAdapterNode(
        adapter=adapter,
        adapter_id="yoloe",
        now_ns_fn=wall,
        monotonic_ns_fn=clock,
        depth=depth,
        **kwargs,
    )
    _, info = _msgs()
    node._on_info(info)
    return node


def _stamp_of(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def test_depth_runs_on_every_frame_and_segmentation_waits_for_the_mask_period() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth)
    got = _capture(node)
    try:
        first, second, third = _STAMP, _STAMP + 100_000_000, _STAMP + 200_000_000
        wall.ns = first + 50_000_000
        node._on_image(_image_at(first))
        assert (spy.calls, depth.calls, len(got["mask"]), len(got["depth"])) == (1, 1, 1, 1)
        assert _stamp_of(got["mask"][0]) == first
        degraded_after_first = len(got["degraded"])
        clock.advance_ms(100)  # 134 ms after the first segmentation started: not due
        wall.ns = second + 50_000_000
        node._on_image(_image_at(second))
        assert spy.calls == 1, "segmentation must wait for the mask period"
        assert depth.calls == 2 and len(got["depth"]) == 2
        assert _stamp_of(got["depth"][1]) == second
        assert len(got["mask"]) == 1
        assert len(got["degraded"]) == degraded_after_first, "a skipped frame publishes no degraded flag"
        clock.advance_ms(100)  # 268 ms: due
        wall.ns = third + 50_000_000
        node._on_image(_image_at(third))
        assert (spy.calls, depth.calls, len(got["mask"])) == (2, 3, 2)
        assert _stamp_of(got["mask"][1]) == third, "the mask carries the stamp of the image it came from"
        assert node.metrics.masks_published == 2 and node.metrics.depth_errors == 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_without_a_depth_channel_every_frame_is_segmented_as_before() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy = SpyAdapter()
    node = PerceptionAdapterNode(
        adapter=spy, adapter_id="yoloe", now_ns_fn=wall, monotonic_ns_fn=clock
    )
    _, info = _msgs()
    node._on_info(info)
    try:
        for i in range(3):
            stamp = _STAMP + i * 10_000_000
            wall.ns = stamp + 50_000_000
            node._on_image(_image_at(stamp))
            clock.advance_ms(10)
        assert spy.calls == 3
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_a_mask_period_of_zero_segments_every_frame() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth, mask_period_s=0.0)
    try:
        for i in range(3):
            stamp = _STAMP + i * 10_000_000
            wall.ns = stamp + 50_000_000
            node._on_image(_image_at(stamp))
        assert spy.calls == 3 and depth.calls == 3
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_a_stale_frame_inside_the_period_still_publishes_degraded_at_once() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth)
    got = _capture(node)
    try:
        wall.ns = _STAMP + 50_000_000
        node._on_image(_image_at(_STAMP))
        assert got["degraded"][-1].data is False
        clock.advance_ms(10)  # well inside the mask period
        stale_stamp = _STAMP + 10_000_000
        wall.ns = stale_stamp + 2_000_000_000  # the frame is 2 s old
        node._on_image(_image_at(stale_stamp))
        assert got["degraded"][-1].data is True, "a stale frame must degrade perception immediately"
        assert spy.calls == 1, "a stale frame costs no inference"
        assert len(got["mask"]) == 1
        assert len(got["depth"]) == 2, "depth is independent of the mask"
        # The stale frame did not use up the mask period, and the period still counts from the first mask.
        clock.advance_ms(10)
        fresh = _STAMP + 20_000_000
        wall.ns = fresh + 50_000_000
        node._on_image(_image_at(fresh))
        assert spy.calls == 1, "still inside the period that started with the first segmentation"
        clock.advance_ms(300)
        fresh2 = _STAMP + 400_000_000
        wall.ns = fresh2 + 50_000_000
        node._on_image(_image_at(fresh2))
        assert spy.calls == 2 and _stamp_of(got["mask"][-1]) == fresh2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_an_undecodable_frame_inside_the_period_still_publishes_degraded_at_once() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth)
    got = _capture(node)
    try:
        wall.ns = _STAMP + 50_000_000
        node._on_image(_image_at(_STAMP))
        clock.advance_ms(10)
        _, bad_info = _msgs()
        bad_info.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # the reserved identity K: rejected
        node._on_info(bad_info)
        node._on_image(_image_at(_STAMP + 10_000_000))
        assert got["degraded"][-1].data is True
        assert spy.calls == 1 and depth.calls == 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.parametrize("period", [0.3, 1.0, -0.1])
def test_a_mask_period_that_would_let_the_mask_go_stale_is_rejected(period: float) -> None:
    rclpy.init()
    try:
        with pytest.raises(ValueError):
            PerceptionAdapterNode(adapter=SpyAdapter(), adapter_id="yoloe", mask_period_s=period)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_the_default_mask_period_is_declared_as_a_parameter() -> None:
    rclpy.init()
    node = PerceptionAdapterNode(adapter=SpyAdapter(), adapter_id="yoloe")
    try:
        assert node.get_parameter("mask_period_s").value == 0.25
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
