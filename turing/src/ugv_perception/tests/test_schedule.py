"""Segmentation scheduling: the pure decision, the estimates that feed it, and a simulation of a whole run.

Pure: injected times, no clock, no rclpy, no GPU.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import pytest

from ugv_perception.node.schedule import (
    DEFAULT_MASK_MAX_GAP_S,
    SegScheduler,
    carried_degraded,
    mask_stream_stalled,
    seg_due,
)

_NS = 1_000_000_000
_MS = 1_000_000
_MAX_AGE_S = 0.50


def ms(value: float) -> int:
    return int(round(value * _MS))


def due(now, last, arrival, interval, depth, gap=DEFAULT_MASK_MAX_GAP_S) -> bool:
    """seg_due with times in ms. last/depth may be None (no estimate yet)."""
    return seg_due(
        now_ns=ms(now),
        last_seg_ns=None if last is None else ms(last),
        arrival_ns=ms(arrival),
        frame_interval_ns=ms(interval),
        depth_tick_ns=None if depth is None else ms(depth),
        mask_max_gap_s=gap,
    )


# --- the pure decision ---------------------------------------------------------------------------------


def test_the_default_mask_max_gap_is_0_30_s() -> None:
    assert DEFAULT_MASK_MAX_GAP_S == 0.30


def test_with_no_estimate_yet_the_frame_is_segmented() -> None:
    assert due(5_000, None, 5_000, 100, 80) is True  # nothing segmented yet
    assert due(5_000, 4_950, 5_000, 100, None) is True  # no depth-only tick measured yet


def test_a_frame_is_skipped_exactly_when_the_predicted_gap_is_within_the_bound() -> None:
    # last segmentation at 0, no camera wait, depth tick 80: the next segmentation would start at now + 80.
    assert due(220, 0, 220, 0, 80) is False  # 300 ms: at the bound, skip
    assert due(221, 0, 221, 0, 80) is True  # 301 ms: over, segment
    assert due(300, 0, 300, 0, 80, gap=0.4) is False
    assert due(300, 0, 300, 0, 80, gap=0.2) is True


def test_the_next_start_is_the_later_of_the_depth_tick_end_and_the_next_frame() -> None:
    # depth tick end = 100 + 80 = 180; next frame = 100 + 150 = 250: the frame is later, 250 <= 300.
    assert due(100, 0, 100, 150, 80) is False
    # a slower camera: next frame at 100 + 250 = 350 > 300.
    assert due(100, 0, 100, 250, 80) is True
    # a slower depth tick: 100 + 250 = 350 > 300 although the camera is fast.
    assert due(100, 0, 100, 0, 250) is True


def test_a_frame_that_arrived_earlier_than_now_makes_a_skip_cheaper() -> None:
    # The node was busy: the frame arrived at 135 but is picked up at 203. Camera interval 135.
    assert due(203, 0, 135, 135, 80) is False  # next start max(283, 270) = 283
    assert due(203, 0, 203, 135, 80) is True  # if it had only just arrived: max(283, 338) = 338


def test_a_frame_cannot_arrive_after_its_own_tick_started() -> None:
    assert due(203, 0, 400, 135, 80) == due(203, 0, 203, 135, 80)


def test_at_12_hz_frames_alternate_whether_or_not_the_interval_is_known() -> None:
    for interval in (83, 0):  # a fast camera never yields a measured interval
        assert due(203, 0, 166, interval, 80) is False
        assert due(283, 0, 249, interval, 80) is True


def test_a_clock_that_went_backwards_counts_as_due() -> None:
    assert due(9_000, 10_000, 9_000, 100, 80) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"now_ns": 1.5},
        {"now_ns": True},
        {"last_seg_ns": 1.5},
        {"last_seg_ns": True},
        {"arrival_ns": None},
        {"arrival_ns": 1.5},
        {"frame_interval_ns": None},
        {"frame_interval_ns": -1},
        {"depth_tick_ns": 1.5},
        {"depth_tick_ns": -1},
        {"mask_max_gap_s": 0.0},
        {"mask_max_gap_s": -0.1},
        {"mask_max_gap_s": float("nan")},
        {"mask_max_gap_s": float("inf")},
        {"mask_max_gap_s": "0.3"},
        {"mask_max_gap_s": True},
    ],
)
def test_bad_arguments_are_rejected(kwargs) -> None:
    good = dict(
        now_ns=ms(100),
        last_seg_ns=ms(0),
        arrival_ns=ms(100),
        frame_interval_ns=ms(100),
        depth_tick_ns=ms(80),
        mask_max_gap_s=0.3,
    )
    good.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        seg_due(**good)


# --- the degraded value on a depth-only frame, and the liveness of the mask stream --------------------------
# "Published" times are the node's monotonic clock at the moment a mask went out. The age of the mask's image
# stamp is NOT used: it is latency plus the time the mask is held before the next one replaces it, which
# reaches 0.45 to 0.58 s in normal operation. Liveness asks only whether masks are still being produced.


def test_the_stream_is_stalled_when_the_newest_mask_was_published_too_long_ago() -> None:
    t = 10 * _NS
    kw = {"first_image_ns": t - 5 * _NS, "max_age_s": _MAX_AGE_S}
    assert mask_stream_stalled(last_published_ns=t, now_ns=t + ms(100), **kw) is False
    assert mask_stream_stalled(last_published_ns=t, now_ns=t + ms(500), **kw) is False  # at the limit
    assert mask_stream_stalled(last_published_ns=t, now_ns=t + ms(500) + 1, **kw) is True
    assert mask_stream_stalled(last_published_ns=t, now_ns=t - ms(5), **kw) is False  # a clock step back


def test_with_no_mask_yet_the_stream_is_stalled_only_after_the_limit_since_the_first_image() -> None:
    t = 10 * _NS
    kw = {"last_published_ns": None, "max_age_s": _MAX_AGE_S}
    assert mask_stream_stalled(first_image_ns=None, now_ns=t, **kw) is False  # no image, nothing to wait for
    assert mask_stream_stalled(first_image_ns=t, now_ns=t + ms(500), **kw) is False
    assert mask_stream_stalled(first_image_ns=t, now_ns=t + ms(500) + 1, **kw) is True


@pytest.mark.parametrize(
    "last_decision,published_ago_ms,expected",
    [
        (False, 100, False),  # nothing wrong
        (False, 500, False),  # exactly at the limit
        (True, 100, True),  # never less conservative than the last segmented decision
        (None, 100, True),  # no decision yet
        (False, 501, True),  # no mask published for longer than perception_max_age
        (False, None, True),  # no mask has ever been published
        (True, None, True),
    ],
)
def test_carried_degraded_value(last_decision, published_ago_ms, expected) -> None:
    now = 100 * _NS
    published = None if published_ago_ms is None else now - ms(published_ago_ms)
    assert (
        carried_degraded(
            last_decision_degraded=last_decision,
            last_published_ns=published,
            now_ns=now,
            max_age_s=_MAX_AGE_S,
        )
        is expected
    )


# --- the estimates --------------------------------------------------------------------------------------

_BASE = 5 * _NS  # camera clock: only differences between stamps are ever used


def stamp(t_ms: float) -> int:
    return _BASE + ms(t_ms)


def test_the_depth_only_tick_estimate_skips_the_first_sample_then_averages() -> None:
    s = SegScheduler()
    s.tick_done(ms(0), ms(1500), depth_ns=ms(1400))  # start-up: lazy init, not representative
    assert s.depth_tick_ns is None
    s.tick_done(ms(2000), ms(2100), depth_ns=ms(80))
    assert s.depth_tick_ns == ms(80)
    s.tick_done(ms(3000), ms(3100), depth_ns=ms(100))
    assert s.depth_tick_ns == pytest.approx(ms(86))  # 0.7 * 80 + 0.3 * 100
    s.tick_done(ms(4000), ms(4005), depth_ns=None)  # a frame without depth says nothing about it
    assert s.depth_tick_ns == pytest.approx(ms(86))


def test_an_idle_wait_measures_the_camera_interval_exactly_and_a_backlog_does_not() -> None:
    s = SegScheduler()
    s.frame_arrived(ms(0), stamp(0))
    s.tick_done(ms(0), ms(100), depth_ns=ms(80))
    # The node was back-to-back: the frame was already waiting, so the stamp delta is only a multiple.
    s.frame_arrived(ms(100), stamp(270))
    assert s.frame_interval_ns is None
    s.tick_done(ms(100), ms(180), depth_ns=ms(80))
    # The node waited 120 ms for this frame: it is the very next one, so the delta is the interval.
    s.frame_arrived(ms(300), stamp(405))
    assert s.frame_interval_ns == ms(135)
    assert s.arrival_ns == ms(300)  # an idle frame arrived when the callback began


def test_a_measured_interval_is_kept_until_a_shorter_stamp_delta_contradicts_it() -> None:
    s = SegScheduler()
    s.frame_arrived(ms(0), stamp(0))
    s.tick_done(ms(0), ms(10), depth_ns=ms(5))
    s.frame_arrived(ms(200), stamp(200))  # idle: interval 200
    assert s.frame_interval_ns == ms(200)
    # A backlogged stamp delta is a multiple of the interval (or jitter): the measurement stands, however long it is.
    s.tick_done(ms(200), ms(403), depth_ns=ms(80))
    s.frame_arrived(ms(403), stamp(400))
    assert s.frame_interval_ns == ms(200)
    s.tick_done(ms(403), ms(606), depth_ns=ms(80))
    s.frame_arrived(ms(606), stamp(790))  # 390 = about two intervals; 190 would be within jitter
    assert s.frame_interval_ns == ms(200)
    s.tick_done(ms(606), ms(809), depth_ns=ms(80))
    s.frame_arrived(ms(809), stamp(980))  # 190: jitter, not a faster camera
    assert s.frame_interval_ns == ms(200)
    # Two frames the node took only 100 ms apart: the camera cannot be at 200 ms any more.
    s.tick_done(ms(809), ms(1_012), depth_ns=ms(80))
    s.frame_arrived(ms(1_012), stamp(1_080))
    assert s.frame_interval_ns is None


def test_the_latest_idle_measurement_wins() -> None:
    s = SegScheduler()
    s.frame_arrived(ms(0), stamp(0))
    s.tick_done(ms(0), ms(10), depth_ns=ms(5))
    s.frame_arrived(ms(135), stamp(135))
    assert s.frame_interval_ns == ms(135)
    s.tick_done(ms(135), ms(145), depth_ns=ms(5))
    s.frame_arrived(ms(345), stamp(345))  # the camera slowed down: 210
    assert s.frame_interval_ns == ms(210)


def test_a_waiting_frames_arrival_is_projected_from_the_previous_one_inside_what_is_possible() -> None:
    s = SegScheduler()
    s.frame_arrived(ms(0), stamp(0))
    s.tick_done(ms(0), ms(203), depth_ns=ms(80))
    s.frame_arrived(ms(203), stamp(135))
    assert s.arrival_ns == ms(135)  # 0 + 135, within (tick start 0, now 203]
    s.tick_done(ms(203), ms(283), depth_ns=ms(80))
    s.frame_arrived(ms(283), stamp(270))
    assert s.arrival_ns == ms(270)
    s.tick_done(ms(283), ms(486), depth_ns=ms(80))
    # A big stamp delta (frames were dropped) cannot put the arrival after the callback began ...
    s.frame_arrived(ms(486), stamp(900))
    assert s.arrival_ns == ms(486)
    s.tick_done(ms(486), ms(500), depth_ns=ms(80))
    s.frame_arrived(ms(500), stamp(901))
    assert s.arrival_ns == ms(487)  # 486 + 1, inside (tick start 486, now 500]
    s.tick_done(ms(500), ms(520), depth_ns=ms(80))
    s.frame_arrived(ms(520), stamp(500))  # camera clock stepped back: no usable delta
    assert s.arrival_ns == ms(520)


def test_a_waiting_frame_arrived_after_the_previous_tick_began() -> None:
    s = SegScheduler()
    s.frame_arrived(ms(0), stamp(0))
    s.tick_done(ms(0), ms(1_000), depth_ns=ms(80))  # one long tick
    s.frame_arrived(ms(1_000), stamp(10))  # arrival 0 + 10 = 10 ...
    assert s.arrival_ns == ms(10)
    s.tick_done(ms(1_000), ms(1_100), depth_ns=ms(80))
    s.frame_arrived(ms(1_100), stamp(20))  # ... but this one arrived during the tick that began at 1000
    assert s.arrival_ns == ms(1_000)


def test_only_a_frame_that_was_segmented_restarts_the_gap() -> None:
    s = SegScheduler()
    assert s.last_seg_ns is None
    s.segmented(ms(100))
    assert s.last_seg_ns == ms(100)


# --- a whole run ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tick:
    start_ms: float
    end_ms: float
    segmented: bool
    flag_ms: float  # when the degraded flag goes out: after the mask, or at once on a depth-only tick
    mask_ms: float | None = None  # when the mask goes out (after the segmentation), None on a depth-only tick


def _simulate(
    frame_interval_ms: float,
    seg_cost_ms: float = 123,
    depth_cost_ms: float = 80,
    seconds: float = 30.0,
    gap_s: float = DEFAULT_MASK_MAX_GAP_S,
    stall_ms: tuple[float, float] | None = None,
) -> list[Tick]:
    """A camera at a fixed interval and a node that takes the newest frame that has arrived (queue depth 1),
    or waits for the next one. It drives SegScheduler exactly as the node does. The camera does not wait for
    the node, so frames that arrive while the node is busy are dropped. `stall_ms` removes the frames inside
    that window: a camera that went quiet."""
    arrivals = [
        k * frame_interval_ms
        for k in range(int(seconds * 1000 / frame_interval_ms) + 2)
        if stall_ms is None or not (stall_ms[0] <= k * frame_interval_ms < stall_ms[1])
    ]
    sched = SegScheduler(gap_s)
    ticks: list[Tick] = []
    free, last = 0.0, -1
    while True:
        newest = bisect.bisect_right(arrivals, free + 1e-9) - 1
        k = newest if newest > last else last + 1
        if k >= len(arrivals):
            break
        start = max(free, arrivals[k])
        if start >= seconds * 1000:
            break
        last = k
        sched.frame_arrived(ms(start), stamp(arrivals[k]))
        segment = sched.due(ms(start))
        cost = depth_cost_ms
        if segment:
            sched.segmented(ms(start))
            cost += seg_cost_ms
        end = start + cost
        sched.tick_done(ms(start), ms(end), depth_ns=ms(depth_cost_ms))
        out_at = start + seg_cost_ms
        ticks.append(Tick(start, end, segment, out_at if segment else start, out_at if segment else None))
        free = end
    return ticks


def _seg_gaps(ticks: list[Tick]) -> list[float]:
    starts = [t.start_ms for t in ticks if t.segmented]
    return [b - a for a, b in zip(starts, starts[1:])]


_CANONICAL = (83, 135, 200, 240)  # 12 Hz, 7.4 Hz, 5 Hz, 4.2 Hz
_TICK_MS = 123 + 80


@pytest.mark.parametrize("interval", (33, 83, 100, 135))
def test_a_fast_camera_gets_depth_on_every_frame_and_a_mask_about_every_other_one(interval: int) -> None:
    ticks = [t for t in _simulate(interval, seconds=30) if t.start_ms > 5_000]
    seconds = (ticks[-1].end_ms - ticks[0].start_ms) / 1000
    assert len(ticks) / seconds >= 5.0, "depth rate"
    depth_only = sum(not t.segmented for t in ticks)
    assert 0.35 * len(ticks) <= depth_only <= 0.65 * len(ticks), "depth-only frames are about every other one"
    assert max(_seg_gaps(ticks)) <= 300, "once the estimates have settled the bound holds exactly"


def test_at_7_4_and_12_hz_depth_only_frames_never_follow_each_other() -> None:
    for interval in (83, 135):
        kinds = [t.segmented for t in _simulate(interval, seconds=30) if t.start_ms > 5_000]
        assert not any(not a and not b for a, b in zip(kinds, kinds[1:]))


def test_a_240_ms_camera_gets_every_frame_segmented_and_no_mask_is_skipped() -> None:
    ticks = [t for t in _simulate(240, seconds=30) if t.start_ms > 5_000]
    assert all(t.segmented for t in ticks)
    assert max(_seg_gaps(ticks)) == 240


def test_a_200_ms_camera_gets_nearly_every_frame_segmented() -> None:
    # A tick (203 ms) a little longer than the camera interval (200 ms) slips by 3 ms a frame. Now and then the
    # waiting frame is already ~100 ms old, so the next one is due in 97 ms and a depth-only frame fits within
    # the bound. That is the rule working, not a skipped mask: the gap is still at most 300 ms.
    ticks = [t for t in _simulate(200, seconds=60) if t.start_ms > 5_000]
    assert sum(t.segmented for t in ticks) >= 0.9 * len(ticks)
    assert max(_seg_gaps(ticks)) <= 300


@pytest.mark.parametrize("interval", (83, 135))
def test_after_a_long_gap_the_first_frame_is_segmented(interval: int) -> None:
    ticks = _simulate(interval, seconds=30, stall_ms=(10_000, 13_000))
    after = [t for t in ticks if t.start_ms >= 13_000]
    before = [t for t in ticks if t.start_ms < 10_000]
    assert before[-1].start_ms > 9_000
    assert after[0].segmented, "the first frame after the gap must be segmented"
    assert after[0].start_ms - [t for t in ticks if t.segmented and t.start_ms < 10_000][-1].start_ms > 3_000


def test_a_depth_tick_too_slow_to_fit_a_skip_means_every_frame_is_segmented() -> None:
    # depth 130 + seg 123 = 253 ms ticks: a skip would push the next segmentation past 300 ms, so none happens.
    ticks = [t for t in _simulate(135, depth_cost_ms=130, seconds=30) if t.start_ms > 5_000]
    assert all(t.segmented for t in ticks)
    assert max(_seg_gaps(ticks)) <= 300 + 135


def _publication_gaps(ticks: list[Tick]) -> list[float]:
    times = [t.mask_ms for t in ticks if t.mask_ms is not None]
    return [b - a for a, b in zip(times, times[1:])]


@pytest.mark.parametrize("interval", _CANONICAL + (33, 100, 160, 300))
def test_the_mask_gap_bound_and_the_mask_stream_liveness(interval: int) -> None:
    """Segmentations start at most the bound plus one frame interval apart, and over the bound at most once (while
    the camera interval is unknown). Masks go out at most 400 ms apart: the liveness rules (0.5 s) never trip in
    normal alternation (the worst, 397 ms, is that one wrong skip at a 200 ms camera; in steady state 300 ms)."""
    ticks = _simulate(interval, seconds=60)
    gaps = _seg_gaps(ticks)
    assert max(gaps) <= 300 + interval, gaps
    if interval in _CANONICAL:
        assert len([g for g in gaps if g > 300]) <= 1, gaps
    assert max(_publication_gaps(ticks)) <= 400
