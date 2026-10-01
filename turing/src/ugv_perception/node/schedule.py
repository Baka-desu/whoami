"""Which frames get segmentation, and the estimates that decision rests on. No rclpy, no clock of its own.

With a depth channel the node runs depth on every frame it takes and segmentation on as many as the
mask gap allows. Segmentation has priority: a frame may be depth-only only if skipping it is predicted to
keep the start-to-start gap between segmentations at or below `mask_max_gap_s`.

`seg_due` is the decision, a pure function of its arguments. `SegScheduler` only keeps the estimates it needs.

What the node can and cannot observe, because it shapes the estimates:

* It sees the frames that survive its depth-1 queue, not every camera frame. When it is busy, frames
  are dropped, so the stamp delta between two frames it took is a multiple of the camera interval. A mean of
  those deltas follows the node's own cadence, not the camera's, and a fast camera (30 fps against a 200 ms
  tick) would look like a slow one and never get a depth-only frame.
* The camera interval is measured exactly in one situation only: the node was idle, waiting for a frame.
  Nothing arrived during the previous tick, so the frame that ended the wait is the very next one, and its
  stamp delta is the interval. A camera slower than the node is idle often, so it is measured quickly.
  A camera that never lets the node idle is faster than a depth tick, and then the term the interval feeds
  (the next frame arriving later than the depth tick ends) cannot be the larger one, so it is left out (0).
  If that is wrong the next start is later than predicted, the node idles, and the interval is measured:
  a wrong skip is paid for once per change of camera rate.
* A measured interval is kept until something contradicts it, not for a fixed time (a timeout would make a
  steady 5 Hz camera pay a wrong skip every few seconds). No two frames the node takes can be closer than
  the camera interval, so a stamp delta clearly shorter than the measured interval means the camera got
  faster: the measurement is dropped and the optimistic rule applies again.
"""

from __future__ import annotations

import math

DEFAULT_MASK_MAX_GAP_S = 0.30
_NS = 1_000_000_000
# A callback that begins this long after the previous tick ended was waiting for its frame. A callback that
# was already queued begins within microseconds of the previous tick's end.
_IDLE_NS = 2_000_000
# A stamp delta below this fraction of the measured interval contradicts it (allows for stamp jitter).
_CONTRADICTS = 0.9
# Weight of the newest depth-only tick duration in the running average.
_DEPTH_ALPHA = 0.3


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int")
    return value


def _require_gap(value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError("mask_max_gap_s must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("mask_max_gap_s must be finite and > 0")
    return float(value)


def seg_due(
    *,
    now_ns: int,
    last_seg_ns: int | None,
    arrival_ns: int,
    frame_interval_ns: int,
    depth_tick_ns: int | None,
    mask_max_gap_s: float = DEFAULT_MASK_MAX_GAP_S,
) -> bool:
    """True when this frame should be segmented, False when it may be depth-only.

    The next segmentation can start no earlier than the end of this frame's depth-only tick
    (`now + depth_tick_ns`) and no earlier than the next camera frame (`arrival + frame_interval_ns`).
    The frame is skipped only if the later of the two is within `mask_max_gap_s` of the last segmentation
    start. All times come from one monotonic clock.

    Segment when there is nothing to predict from (no segmentation yet, no depth-only tick measured), and
    when the clock went backwards. A frame cannot have arrived after its own tick began. A
    `frame_interval_ns` of 0 means no camera wait is expected.
    """
    now_ns = _require_int(now_ns, "now_ns")
    if last_seg_ns is not None:
        last_seg_ns = _require_int(last_seg_ns, "last_seg_ns")
    arrival_ns = _require_int(arrival_ns, "arrival_ns")
    frame_interval_ns = _require_int(frame_interval_ns, "frame_interval_ns")
    if frame_interval_ns < 0:
        raise ValueError("frame_interval_ns must be >= 0")
    if depth_tick_ns is not None:
        depth_tick_ns = _require_int(depth_tick_ns, "depth_tick_ns")
        if depth_tick_ns < 0:
            raise ValueError("depth_tick_ns must be >= 0")
    gap_ns = round(_require_gap(mask_max_gap_s) * _NS)
    if last_seg_ns is None or depth_tick_ns is None or now_ns < last_seg_ns:
        return True
    next_start = max(now_ns + depth_tick_ns, min(arrival_ns, now_ns) + frame_interval_ns)
    return next_start - last_seg_ns > gap_ns


def mask_expired(last_mask_stamp_ns: int | None, now_ns: int, max_age_s: float) -> bool:
    """True when the newest published mask is older than `max_age_s`. No mask yet: nothing to expire."""
    if last_mask_stamp_ns is None:
        return False
    return (now_ns - last_mask_stamp_ns) / _NS > max_age_s


def carried_degraded(
    *,
    last_decision_degraded: bool | None,
    last_mask_stamp_ns: int | None,
    now_ns: int,
    max_age_s: float,
) -> bool:
    """The flag a depth-only frame publishes. Never less conservative than the last segmented decision.

    Degraded if that decision was degraded (or there is none), if no mask has ever been published, or if the
    newest published mask is older than `max_age_s`.
    """
    return (
        last_decision_degraded is not False
        or last_mask_stamp_ns is None
        or mask_expired(last_mask_stamp_ns, now_ns, max_age_s)
    )


class SegScheduler:
    """The last segmentation start and the estimates `seg_due` needs. Times are one monotonic clock.

    Call order per frame: `frame_arrived` when its callback begins, `due`, `segmented` if it was segmented,
    `tick_done` when the tick ends.
    """

    def __init__(self, mask_max_gap_s: float = DEFAULT_MASK_MAX_GAP_S) -> None:
        self.mask_max_gap_s = _require_gap(mask_max_gap_s)
        self.last_seg_ns: int | None = None
        self.arrival_ns: int | None = None
        self._prev_start_ns: int | None = None
        self._prev_end_ns: int | None = None
        self._prev_stamp_ns: int | None = None
        self._interval_ns: int | None = None  # exact camera interval, measured while the node was idle
        self._depth_tick_ns: float | None = None
        self._depth_samples = 0

    @property
    def depth_tick_ns(self) -> int | None:
        """Running average of the time a frame takes without segmentation (decode and depth). None until known."""
        return None if self._depth_tick_ns is None else round(self._depth_tick_ns)

    @property
    def frame_interval_ns(self) -> int | None:
        """The camera interval as last measured exactly, until a shorter stamp delta contradicts it. Else None."""
        return self._interval_ns

    def frame_arrived(self, entry_ns: int, stamp_ns: int) -> None:
        """A frame's callback began at `entry_ns` (monotonic); `stamp_ns` is its header stamp (camera clock)."""
        entry_ns = _require_int(entry_ns, "entry_ns")
        stamp_ns = _require_int(stamp_ns, "stamp_ns")
        delta = None if self._prev_stamp_ns is None else stamp_ns - self._prev_stamp_ns
        if delta is not None and delta <= 0:
            delta = None
        idle = self._prev_end_ns is not None and entry_ns - self._prev_end_ns >= _IDLE_NS
        if idle:
            # Nothing arrived during the previous tick, so this is the very next frame and arrived just now.
            self.arrival_ns = entry_ns
            if delta is not None:
                self._interval_ns = delta
        elif delta is not None and self.arrival_ns is not None and self._prev_start_ns is not None:
            # Already waiting: it arrived during the previous tick, one stamp delta after the previous frame.
            low, high = self._prev_start_ns, entry_ns
            self.arrival_ns = min(high, max(low, self.arrival_ns + delta))
        else:
            self.arrival_ns = entry_ns
        if not idle and delta is not None and self._interval_ns is not None:
            if delta < _CONTRADICTS * self._interval_ns:
                self._interval_ns = None
        self._prev_stamp_ns = stamp_ns

    def due(self, start_ns: int) -> bool:
        """Should the frame whose tick began at `start_ns` be segmented? `frame_arrived` came first."""
        interval = self.frame_interval_ns
        return seg_due(
            now_ns=start_ns,
            last_seg_ns=self.last_seg_ns,
            arrival_ns=start_ns if self.arrival_ns is None else self.arrival_ns,
            frame_interval_ns=0 if interval is None else interval,
            depth_tick_ns=self.depth_tick_ns,
            mask_max_gap_s=self.mask_max_gap_s,
        )

    def segmented(self, start_ns: int) -> None:
        """The frame whose tick began at `start_ns` is going to be segmented."""
        self.last_seg_ns = _require_int(start_ns, "start_ns")

    def tick_done(self, start_ns: int, end_ns: int, depth_ns: int | None) -> None:
        """A tick ended. `depth_ns` is its decode and depth time (None when depth did not complete)."""
        self._prev_start_ns = _require_int(start_ns, "start_ns")
        self._prev_end_ns = _require_int(end_ns, "end_ns")
        if depth_ns is None:
            return
        depth_ns = max(0, _require_int(depth_ns, "depth_ns"))
        self._depth_samples += 1
        if self._depth_samples == 1:
            return  # the first call pays for lazy initialisation: not representative
        if self._depth_tick_ns is None:
            self._depth_tick_ns = float(depth_ns)
        else:
            self._depth_tick_ns += _DEPTH_ALPHA * (depth_ns - self._depth_tick_ns)
