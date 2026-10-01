"""The segmentation scheduling kernel. Pure function, injected clock, no rclpy, no GPU."""

from __future__ import annotations

import pytest

from ugv_perception.node.schedule import DEFAULT_MASK_PERIOD_S, seg_due

_NS = 1_000_000_000
_MS = 1_000_000


def test_the_default_mask_period_is_a_quarter_second() -> None:
    assert DEFAULT_MASK_PERIOD_S == 0.25


def test_the_first_frame_is_always_segmented() -> None:
    assert seg_due(5 * _NS, None) is True
    assert seg_due(1, None, 10.0) is True


def test_a_frame_inside_the_period_is_not_segmented_and_the_boundary_is() -> None:
    t0 = 10 * _NS
    assert seg_due(t0 + 1 * _MS, t0) is False
    assert seg_due(t0 + 249 * _MS, t0) is False
    assert seg_due(t0 + 250 * _MS, t0) is True
    assert seg_due(t0 + 400 * _MS, t0) is True


def test_the_period_is_a_parameter() -> None:
    t0 = 10 * _NS
    assert seg_due(t0 + 120 * _MS, t0, 0.1) is True
    assert seg_due(t0 + 120 * _MS, t0, 0.2) is False
    assert seg_due(t0 + 120 * _MS, t0, mask_period_s=0.2) is False


def test_a_period_of_zero_segments_every_frame() -> None:
    t0 = 10 * _NS
    assert seg_due(t0, t0, 0.0) is True
    assert seg_due(t0 + 1, t0, 0) is True


def test_a_clock_that_went_backwards_segments_rather_than_waits() -> None:
    assert seg_due(9 * _NS, 10 * _NS) is True


@pytest.mark.parametrize(
    "now_ns,last_ns,period",
    [
        (1.5, None, 0.25),
        (True, None, 0.25),
        ("1", None, 0.25),
        (1, 1.5, 0.25),
        (1, True, 0.25),
        (1, None, "0.25"),
        (1, None, True),
        (1, None, -0.1),
        (1, None, float("nan")),
        (1, None, float("inf")),
    ],
)
def test_bad_arguments_are_rejected(now_ns, last_ns, period) -> None:
    with pytest.raises((TypeError, ValueError)):
        seg_due(now_ns, last_ns, period)


def _simulate(
    frame_interval_ms: int,
    seg_cost_ms: int,
    depth_cost_ms: int,
    seconds: float,
    period_s: float = DEFAULT_MASK_PERIOD_S,
) -> tuple[list[int], int]:
    """Frames arrive no faster than they are processed. Returns (seg start times in ms, depth count)."""
    now_ms = 0
    last_seg_ns: int | None = None
    segs: list[int] = []
    depths = 0
    while now_ms < seconds * 1000:
        now_ns = now_ms * _MS
        cost = depth_cost_ms
        if seg_due(now_ns, last_seg_ns, period_s):
            last_seg_ns = now_ns
            segs.append(now_ms)
            cost += seg_cost_ms
        depths += 1
        now_ms += max(cost, frame_interval_ms)
    return segs, depths


def test_depth_runs_on_every_frame_and_a_mask_comes_at_least_every_period_plus_one_frame() -> None:
    # The measured costs on the RTX 4060: seg 123 ms, depth about 80 ms; camera at 7.4 Hz (135 ms).
    segs, depths = _simulate(frame_interval_ms=135, seg_cost_ms=123, depth_cost_ms=80, seconds=60)
    gaps = [b - a for a, b in zip(segs, segs[1:])]
    assert max(gaps) <= 250 + 135, gaps
    assert min(gaps) >= 250
    frames_without_seg = depths - len(segs)
    assert frames_without_seg > 0, "the scheduler never skipped a segmentation"
    assert depths >= 1.5 * len(segs), "depth must run on far more frames than segmentation"


def test_with_a_fast_camera_the_mask_rate_is_capped_and_the_depth_rate_is_not() -> None:
    segs, depths = _simulate(frame_interval_ms=33, seg_cost_ms=10, depth_cost_ms=10, seconds=10)
    assert len(segs) <= 10 / 0.25 + 1
    assert depths >= 10 * 1000 / 33 * 0.95


def test_a_slow_segmentation_never_waits_for_the_period() -> None:
    # seg takes longer than the period: every frame is due again as soon as it arrives.
    segs, depths = _simulate(frame_interval_ms=100, seg_cost_ms=400, depth_cost_ms=50, seconds=20)
    assert len(segs) == depths
