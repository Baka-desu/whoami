"""Live rclpy spin of PerceptionAdapterNode with fixture Image+CameraInfo."""

from __future__ import annotations

import bisect
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
        ex.shutdown()  # before rclpy.shutdown(): a late Executor.__del__ would raise InvalidHandle
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
        ex.shutdown()  # before rclpy.shutdown(): a late Executor.__del__ would raise InvalidHandle
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
        ex.shutdown()  # before rclpy.shutdown(): a late Executor.__del__ would raise InvalidHandle
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
        ex.shutdown()  # before rclpy.shutdown(): a late Executor.__del__ would raise InvalidHandle
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


# --- Segmentation scheduling: depth on every frame, segmentation while the mask gap allows. -------------


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


def _log_publications(node: PerceptionAdapterNode, clock: "_Clock") -> list[tuple[str, int, object]]:
    """(kind, monotonic time, message) for every mask, degraded flag and depth image, in publication order."""
    log: list[tuple[str, int, object]] = []
    for kind, pub in (
        ("mask", node._pub_mask),
        ("degraded", node._pub_degraded),
        ("depth", node._pub_depth),
    ):
        original = pub.publish

        def publish(msg, _kind=kind, _original=original) -> None:
            log.append((_kind, clock.ns, msg))
            _original(msg)

        pub.publish = publish
    return log


class _Wall:
    """Injected wall clock for freshness. A test sets it relative to the stamp of the frame it feeds."""

    def __init__(self) -> None:
        self.ns = _STAMP + 100_000_000

    def __call__(self) -> int:
        return self.ns


_MONO0 = 1_000_000_000  # where _Clock starts


class _LiveWall:
    """Wall clock that follows the injected monotonic clock: the camera and the node share a time base."""

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock

    def __call__(self) -> int:
        return _STAMP + (self._clock.ns - _MONO0)


def _scheduled_node(clock: _Clock, wall, adapter, depth, **kwargs):
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


def _feed(node, clock: _Clock, wall: _Wall, t_ms: float, *, age_ms: float = 50.0) -> int:
    """Deliver the frame stamped `_STAMP + t_ms` the way the executor would, at monotonic `t_ms` after the start.

    The monotonic clock only moves forward (a tick that ran long keeps it ahead). The wall clock is `age_ms`
    after the stamp. Returns the stamp."""
    stamp = _STAMP + int(t_ms * _MS)
    clock.ns = max(clock.ns, _MONO0 + int(t_ms * _MS))
    wall.ns = stamp + int(age_ms * _MS)
    node._on_image(_image_at(stamp))
    return stamp


def _prime(node, clock: _Clock, wall: _Wall) -> None:
    """Two frames 100 ms apart. The first is segmented (nothing is known yet), the second because the first
    depth sample is not trusted. After them the camera interval (100 ms) and the depth tick (34 ms) are known,
    so a frame at 200 ms is depth-only."""
    _feed(node, clock, wall, 0)
    _feed(node, clock, wall, 100)


def _tear_down(node, *others) -> None:
    node.destroy_node()
    for other in others:
        other.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


class _FlakyAdapter(SpyAdapter):
    """Raises on the nth call (like a model that fell over once), or on every call from the nth on."""

    def __init__(self, fail_on: int, *, stay_down: bool = False) -> None:
        super().__init__()
        self._fail_on = fail_on
        self._stay_down = stay_down
        self.attempts = 0

    def infer(self, frame):
        from ugv_perception.adapter.output import AdapterError

        self.attempts += 1
        if self.attempts == self._fail_on or (self._stay_down and self.attempts > self._fail_on):
            raise AdapterError("model fell over")
        return super().infer(frame)


class _FlakyDepth(_FakeDepth):
    def __init__(self, clock: _Clock, fail_on: int) -> None:
        super().__init__(clock)
        self._fail_on = fail_on

    def maps(self, rgb, k, stage=None):
        if self.calls + 1 == self._fail_on:
            self.calls += 1
            raise RuntimeError("boom")
        return super().maps(rgb, k, stage)


def test_the_first_two_frames_are_segmented_and_the_third_is_depth_only() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth)
    got = _capture(node)
    try:
        s1 = _feed(node, clock, wall, 0)
        assert (spy.calls, depth.calls) == (1, 1), "nothing is known yet: segment"
        s2 = _feed(node, clock, wall, 100)
        assert (spy.calls, depth.calls) == (2, 2), "the first depth sample is not trusted: segment"
        s3 = _feed(node, clock, wall, 200)
        assert (spy.calls, depth.calls) == (2, 3), "interval and depth tick are known: depth only"
        assert [_stamp_of(m) for m in got["mask"]] == [s1, s2]
        assert [_stamp_of(m) for m in got["depth"]] == [s1, s2, s3]
        assert node.metrics.masks_published == 2 and node.metrics.depth_errors == 0
    finally:
        _tear_down(node)


def test_a_depth_only_frame_publishes_the_degraded_flag_before_its_depth() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    node = _scheduled_node(clock, wall, SpyAdapter(), _FakeDepth(clock))
    log = _log_publications(node, clock)
    try:
        _prime(node, clock, wall)
        log.clear()
        _feed(node, clock, wall, 200)
        kinds = [k for k, _, _ in log]
        assert kinds == ["degraded", "depth"], "no mask, but the flag still goes out, ahead of the depth"
        assert log[0][2].data is False
        assert node.metrics.degraded_false == 3 and node.metrics.degraded_true == 0
    finally:
        _tear_down(node)


def test_a_depth_only_frame_carries_a_degraded_last_decision() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    flaky = _FlakyAdapter(fail_on=2)  # the second segmentation fails: degraded, no mask
    node = _scheduled_node(clock, wall, flaky, _FakeDepth(clock))
    got = _capture(node)
    try:
        _prime(node, clock, wall)
        assert got["degraded"][-1].data is True and len(got["mask"]) == 1
        _feed(node, clock, wall, 200)
        assert flaky.attempts == 2, "the third frame is depth only"
        assert got["degraded"][-1].data is True, "never less conservative than the last segmented decision"
        assert len(got["degraded"]) == 3
    finally:
        _tear_down(node)


def test_a_depth_only_frame_with_no_mask_ever_published_is_degraded() -> None:
    from ugv_perception.adapter.output import AdapterError

    class _AlwaysFails:
        def infer(self, frame):
            raise AdapterError("model fell over")

    rclpy.init()
    clock, wall = _Clock(), _Wall()
    node = _scheduled_node(clock, wall, _AlwaysFails(), _FakeDepth(clock))
    got = _capture(node)
    try:
        _prime(node, clock, wall)
        _feed(node, clock, wall, 200)
        assert not got["mask"]
        assert len(got["degraded"]) == 3 and all(m.data is True for m in got["degraded"])
    finally:
        _tear_down(node)


def test_a_depth_failure_on_a_depth_only_frame_does_not_touch_the_flag() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    node = _scheduled_node(clock, wall, SpyAdapter(), _FlakyDepth(clock, fail_on=3))
    log = _log_publications(node, clock)
    try:
        _prime(node, clock, wall)
        log.clear()
        _feed(node, clock, wall, 200)
        assert node.metrics.depth_errors == 1
        assert [k for k, _, _ in log] == ["degraded"], "the flag went out, then the depth failed"
        assert log[0][2].data is False
    finally:
        _tear_down(node)


def test_a_stale_frame_inside_the_gap_is_never_skipped_and_publishes_degraded_at_once() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth)
    got = _capture(node)
    try:
        _prime(node, clock, wall)
        flags = len(got["degraded"])
        _feed(node, clock, wall, 200, age_ms=2_000)  # the frame is 2 s old
        assert len(got["degraded"]) == flags + 1 and got["degraded"][-1].data is True
        assert spy.calls == 2, "a stale frame costs no inference"
        assert len(got["mask"]) == 2
        assert depth.calls == 3, "depth is independent of the mask"
        # The stale frame did not count as a segmentation: the gap still runs from the one at 100 ms.
        _feed(node, clock, wall, 290, age_ms=50)
        assert spy.calls == 2, "a fresh frame that fits the bound is depth only"
        assert got["degraded"][-1].data is True, "carried from the stale frame's decision"
        _feed(node, clock, wall, 420, age_ms=50)  # 320 ms after the segmentation at 100 ms
        assert spy.calls == 3
        assert got["degraded"][-1].data is False and _stamp_of(got["mask"][-1]) == _STAMP + 420 * _MS
    finally:
        _tear_down(node)


def test_an_undecodable_frame_inside_the_gap_publishes_degraded_at_once() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth)
    got = _capture(node)
    try:
        _prime(node, clock, wall)
        _, bad_info = _msgs()
        bad_info.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # the reserved identity K: rejected
        node._on_info(bad_info)
        _feed(node, clock, wall, 200)
        assert got["degraded"][-1].data is True
        assert spy.calls == 2 and depth.calls == 2
    finally:
        _tear_down(node)


def test_after_a_long_gap_and_a_stale_burst_the_first_fresh_frame_is_segmented() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth)
    got = _capture(node)
    try:
        _prime(node, clock, wall)
        _feed(node, clock, wall, 200)  # depth only
        assert spy.calls == 2
        # The camera goes quiet for 3 s, then hands over frames it captured earlier: all stale.
        for t_ms in (3_210, 3_310, 3_410):
            _feed(node, clock, wall, t_ms, age_ms=2_500)
        assert spy.calls == 2, "stale frames reach no model"
        assert got["degraded"][-1].data is True
        _feed(node, clock, wall, 3_600, age_ms=50)
        assert spy.calls == 3, "the first fresh frame after the gap is segmented"
        assert got["degraded"][-1].data is False
    finally:
        _tear_down(node)


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
        for i in range(6):
            _feed(node, clock, wall, i * 10)
        assert spy.calls == 6
    finally:
        _tear_down(node)


def test_a_tiny_mask_max_gap_segments_every_frame() -> None:
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    spy, depth = SpyAdapter(), _FakeDepth(clock)
    node = _scheduled_node(clock, wall, spy, depth, mask_max_gap_s=0.001)
    try:
        for i in range(6):
            _feed(node, clock, wall, i * 100)
        assert spy.calls == 6 and depth.calls == 6
    finally:
        _tear_down(node)


# --- R-d: what the parameter guarantees, and what it does not. ----------------------------------------


@pytest.mark.parametrize("gap", [0.0, -0.1, 0.5, 0.75, 2.0])
def test_a_mask_max_gap_that_is_not_inside_zero_and_the_max_age_is_rejected(gap: float) -> None:
    rclpy.init()
    try:
        with pytest.raises(ValueError, match="mask_max_gap_s"):
            PerceptionAdapterNode(adapter=SpyAdapter(), adapter_id="yoloe", mask_max_gap_s=gap)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.parametrize("gap", [0.01, 0.3, 0.45, 0.499])
def test_any_mask_max_gap_inside_zero_and_the_max_age_is_accepted(gap: float) -> None:
    rclpy.init()
    node = PerceptionAdapterNode(adapter=SpyAdapter(), adapter_id="yoloe", mask_max_gap_s=gap)
    try:
        assert node.get_parameter("mask_max_gap_s").value == pytest.approx(gap)
    finally:
        _tear_down(node)


def test_the_default_mask_max_gap_is_declared_as_a_parameter() -> None:
    rclpy.init()
    node = PerceptionAdapterNode(adapter=SpyAdapter(), adapter_id="yoloe")
    try:
        assert node.get_parameter("mask_max_gap_s").value == 0.30
        assert not node.has_parameter("mask_period_s"), "the old minimum-spacing parameter is gone"
    finally:
        _tear_down(node)


# --- R-b: the watchdog also watches the newest mask. --------------------------------------------------


def _watchdog_node(adapter=None):
    rclpy.init()
    clock, wall = _Clock(), _Wall()
    node = _scheduled_node(clock, wall, adapter or SpyAdapter(), _FakeDepth(clock))
    node._stop.set()  # no background thread: the check is called directly, with the times we choose
    return node, clock, wall, _capture(node)


def test_the_watchdog_ignores_the_age_of_the_masks_image_while_masks_keep_coming() -> None:
    node, clock, wall, got = _watchdog_node()
    try:
        _prime(node, clock, wall)  # newest mask: stamped 100 ms, published at monotonic 100 ms
        _feed(node, clock, wall, 200)  # depth only: the newest IMAGE is stamped 200 ms
        got["degraded"].clear()
        # Wall 650 ms: image 450 ms old (live), the mask's image 550 ms old, but the mask was published 250 ms ago.
        assert node._watchdog_check(_STAMP + 650 * _MS, _MONO0 + 350 * _MS) is False
        assert got["degraded"] == []
    finally:
        _tear_down(node)


def test_the_watchdog_raises_degraded_when_masks_stop_while_fresh_images_keep_arriving() -> None:
    node, clock, wall, got = _watchdog_node()
    try:
        _prime(node, clock, wall)  # newest mask published at monotonic 100 ms
        _feed(node, clock, wall, 200)
        got["degraded"].clear()
        # Images are live (stamp 200 ms, wall 400 ms): only the mask stream can trip it.
        wall_now = _STAMP + 400 * _MS
        assert node._watchdog_check(wall_now, _MONO0 + 600 * _MS) is False  # 500 ms since the mask: at the limit
        assert node._watchdog_check(wall_now, _MONO0 + 600 * _MS + 1) is True
        assert [m.data for m in got["degraded"]] == [True]
        # The existing image rule is unchanged: a 550 ms old image trips it regardless of the masks.
        assert node._watchdog_check(_STAMP + 750 * _MS, _MONO0 + 150 * _MS) is True
        assert [m.data for m in got["degraded"]] == [True, True]
    finally:
        _tear_down(node)


def test_the_watchdog_raises_degraded_when_no_mask_comes_for_longer_than_the_max_age_since_the_first_image() -> None:
    node, clock, wall, got = _watchdog_node(_FlakyAdapter(fail_on=1, stay_down=True))
    try:
        _feed(node, clock, wall, 0)  # the first image, at monotonic 0; the adapter fails: no mask
        assert not got["mask"]
        got["degraded"].clear()
        wall_now = _STAMP + 400 * _MS  # the image stamped at 0 is 400 ms old: live
        assert node._watchdog_check(wall_now, _MONO0 + 500 * _MS) is False
        assert node._watchdog_check(wall_now, _MONO0 + 500 * _MS + 1) is True
    finally:
        _tear_down(node)


def test_the_watchdogs_image_rule_is_unchanged_when_no_image_has_arrived() -> None:
    node, clock, wall, got = _watchdog_node()
    try:
        tripped = node._watchdog_check(_STAMP + 100 * _MS, _MONO0 + 100 * _MS)
        assert tripped is True, "no image at all: the existing rule"
        assert [m.data for m in got["degraded"]] == [True]
    finally:
        _tear_down(node)


def _run_until_the_masks_stop(node, clock, wall, got, *, frame_ms=100, watchdog_ms=250, until_ms=2_000):
    """Fresh frames every `frame_ms`, a watchdog check every `watchdog_ms` (the node's real period), both on the
    monotonic clock. Returns (frame flags as (t, value), first watchdog publication time or None)."""
    frame_flags: list[tuple[int, bool]] = []
    first_watchdog = None
    next_check = watchdog_ms
    for t_ms in range(frame_ms * 2, until_ms + 1, frame_ms):
        while next_check <= t_ms:
            clock.ns = max(clock.ns, _MONO0 + next_check * _MS)
            wall_now = _STAMP + next_check * _MS + 50 * _MS  # the newest image is at most one frame old: live
            if node._watchdog_check(wall_now, clock.ns) and first_watchdog is None:
                first_watchdog = next_check
            next_check += watchdog_ms
        _feed(node, clock, wall, t_ms)
        frame_flags.append((t_ms, got["degraded"][-1].data))
    return frame_flags, first_watchdog


def test_when_the_adapter_fails_every_frame_the_watchdog_raises_degraded_within_the_limit_plus_one_period() -> None:
    node, clock, wall, got = _watchdog_node(_FlakyAdapter(fail_on=3, stay_down=True))
    try:
        _prime(node, clock, wall)  # the last mask is published at monotonic 100 ms
        _, first_watchdog = _run_until_the_masks_stop(node, clock, wall, got)
        assert len(got["mask"]) == 2
        # The limit is 500 ms after the last mask (600 ms); the watchdog runs every 250 ms: 250, 500, 750.
        assert first_watchdog is not None and 600 < first_watchdog <= 600 + 250
    finally:
        _tear_down(node)


def test_when_the_scheduler_is_forced_to_skip_the_watchdog_and_the_next_depth_only_frame_raise_degraded() -> None:
    node, clock, wall, got = _watchdog_node()
    try:
        _prime(node, clock, wall)  # the last mask is published at monotonic 100 ms
        node._sched.due = lambda start_ns: False  # the scheduler is forced to skip: masks stop
        frame_flags, first_watchdog = _run_until_the_masks_stop(node, clock, wall, got)
        assert len(got["mask"]) == 2
        assert first_watchdog is not None and 600 < first_watchdog <= 600 + 250
        # The depth-only frames: false while alive, true from the first one after the limit.
        assert all(v is False for t, v in frame_flags if t <= 600), frame_flags
        assert all(v is True for t, v in frame_flags if t > 600), frame_flags
        # A mask comes back: the flag is not latched.
        del node._sched.due
        _feed(node, clock, wall, frame_flags[-1][0] + 100)
        assert len(got["mask"]) == 3 and got["degraded"][-1].data is False
    finally:
        _tear_down(node)


def test_the_watchdog_thread_still_publishes_degraded_when_no_image_ever_arrived() -> None:
    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=SpyAdapter(), adapter_id="yoloe", now_ns_fn=lambda: _STAMP + 100_000_000
    )
    flags: list[bool] = []
    original = node._pub_degraded.publish
    node._pub_degraded.publish = lambda msg: (flags.append(msg.data), original(msg))[1]
    try:
        deadline = time.monotonic() + 3.0
        while not flags and time.monotonic() < deadline:
            time.sleep(0.05)
        assert flags and all(flags), "the thread keeps calling the check every perception_max_age / 2"
    finally:
        _tear_down(node)


# --- R-e: the real node, a camera at a fixed interval, injected clocks. -------------------------------


class _CostAdapter(SpyAdapter):
    """Segmentation that takes `ms` of the injected clock and remembers when it started."""

    def __init__(self, clock: _Clock, ms: float) -> None:
        super().__init__()
        self._clock, self._ms = clock, ms
        self.starts: list[int] = []

    def infer(self, frame):
        self.starts.append(self._clock.ns)
        self._clock.advance_ms(self._ms)
        return super().infer(frame)


class _CostDepth:
    def __init__(self, clock: _Clock, ms: float) -> None:
        self._clock, self._ms = clock, ms
        self.calls = 0

    def maps(self, rgb, k, stage=None):
        self.calls += 1
        with stage("depth_infer"):
            self._clock.advance_ms(self._ms - 4)
        with stage("depth_post"):
            self._clock.advance_ms(4)
        return np.ones((2, 2), dtype=np.float32), np.zeros((4, 3), dtype=np.float32)


def _simulate_node(
    interval_ms: float,
    *,
    seg_ms: float = 123,
    depth_ms: float = 80,
    seconds: float = 20,
    stall_ms: tuple[float, float] | None = None,
):
    """The real node, fed by a camera at a fixed interval, on injected clocks.

    The node takes the newest frame that has arrived (queue depth 1) or waits for the next one; the camera
    does not wait for it. `stall_ms` removes the frames inside that window. Returns the publication log,
    the adapter, the depth channel and the monotonic start (ms) of every processed frame."""
    rclpy.init()
    clock = _Clock()
    adapter, depth = _CostAdapter(clock, seg_ms), _CostDepth(clock, depth_ms)
    node = _scheduled_node(clock, _LiveWall(clock), adapter, depth)
    node._stop.set()  # the background watchdog reads real time; its rules are tested directly
    log = _log_publications(node, clock)
    arrivals = [
        k * interval_ms
        for k in range(int(seconds * 1000 / interval_ms) + 2)
        if stall_ms is None or not (stall_ms[0] <= k * interval_ms < stall_ms[1])
    ]
    processed: list[float] = []
    try:
        free_ms, last = 0.0, -1
        while True:
            newest = bisect.bisect_right(arrivals, free_ms + 1e-9) - 1
            k = newest if newest > last else last + 1
            if k >= len(arrivals):
                break
            start = max(free_ms, arrivals[k])
            if start >= seconds * 1000:
                break
            last = k
            clock.ns = max(clock.ns, _MONO0 + int(round(start * _MS)))
            node._on_image(_image_at(_STAMP + int(round(arrivals[k] * _MS))))
            processed.append(start)
            free_ms = (clock.ns - _MONO0) / _MS
    finally:
        _tear_down(node)
    return log, adapter, depth, processed


@pytest.mark.parametrize("interval", [83, 135, 200, 240])
def test_the_real_node_flags_and_depths_every_frame_it_takes(interval: int) -> None:
    # The gap and liveness bounds are the scheduler's (test_schedule), and the node schedules exactly like it (next
    # test); what is left to prove here is the node's own wiring.
    log, _, depth, processed = _simulate_node(interval)
    assert len(processed) > 40
    flags = [(t, m.data) for kind, t, m in log if kind == "degraded"]
    assert len(flags) == len(processed)  # the flag goes out on every processed frame
    # Depth is produced for every frame the node takes.
    assert depth.calls == len(processed)
    assert len([1 for kind, _, _ in log if kind == "depth"]) == len(processed)
    # A healthy run is never degraded: the carried value does not flicker.
    assert all(value is False for _, value in flags)


@pytest.mark.parametrize("interval", [83, 135, 200, 240])
def test_the_node_schedules_exactly_like_the_scheduler_alone(interval: int) -> None:
    from ugv_perception.tests.test_schedule import _simulate

    expected = [t.start_ms for t in _simulate(interval, seconds=20) if t.segmented]
    _, adapter, _, _ = _simulate_node(interval)
    assert [(s - _MONO0) / _MS for s in adapter.starts] == pytest.approx(expected)


@pytest.mark.parametrize("interval", [83, 135])
def test_after_a_quiet_camera_the_first_frame_is_segmented_in_the_real_node(interval: int) -> None:
    _, adapter, _, processed = _simulate_node(interval, seconds=30, stall_ms=(10_000, 13_000))
    first_after = next(t for t in processed if t >= 13_000)
    assert first_after * _MS + _MONO0 in adapter.starts
