"""In-process T11 counters and stage timing. Not a ROS msg. Not PortMeta."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

_NS = 1_000_000_000
_MS = 1_000_000
_MAX_ERROR_CHARS = 200

STAGES = ("decode", "seg", "depth_infer", "depth_post", "cloud", "publish")


class _Window:
    """(time, value) samples kept for the last `span_ns` only."""

    def __init__(self, span_ns: int) -> None:
        self._span_ns = span_ns
        self._items: deque[tuple[int, int]] = deque()

    def add(self, now_ns: int, value: int = 0) -> None:
        self._items.append((now_ns, value))
        self.trim(now_ns)

    def trim(self, now_ns: int) -> None:
        cutoff = now_ns - self._span_ns
        while self._items and self._items[0][0] <= cutoff:
            self._items.popleft()

    def count(self) -> int:
        return len(self._items)

    def total(self) -> int:
        return sum(value for _, value in self._items)


@dataclass
class PerceptionMetrics:
    frames_in: int = 0
    masks_published: int = 0
    infer_calls: int = 0
    degraded_true: int = 0
    degraded_false: int = 0
    latencies_ns: deque[int] = field(default_factory=lambda: deque(maxlen=600))
    depth_errors: int = 0
    last_depth_error: str | None = None
    clock_ns: Callable[[], int] = field(default=time.monotonic_ns, repr=False)
    window_s: float = 5.0
    _born_ns: int = field(init=False, repr=False)
    _mask_times: _Window = field(init=False, repr=False)
    _depth_times: _Window = field(init=False, repr=False)
    _stage_windows: dict[str, _Window] = field(init=False, repr=False)
    _frame_ns: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.clock_ns):
            raise TypeError("clock_ns must be callable")
        if type(self.window_s) not in (int, float) or self.window_s <= 0:
            raise TypeError("window_s must be a number > 0")
        span_ns = int(self.window_s * _NS)
        self._born_ns = self.clock_ns()
        self._mask_times = _Window(span_ns)
        self._depth_times = _Window(span_ns)
        self._stage_windows = {name: _Window(span_ns) for name in STAGES}
        self._frame_ns = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one span. Spans of the same stage within a frame add up."""
        if name not in self._stage_windows:
            raise ValueError(f"unknown stage {name!r}; expected one of {STAGES}")
        start = self.clock_ns()
        try:
            yield
        finally:
            self._frame_ns[name] = self._frame_ns.get(name, 0) + (self.clock_ns() - start)

    def end_frame(self) -> None:
        """Close the current frame: one sample per stage that ran in it."""
        now = self.clock_ns()
        for name, ns in self._frame_ns.items():
            self._stage_windows[name].add(now, ns)
        self._frame_ns.clear()

    def mark_mask(self) -> None:
        self.masks_published += 1
        self._mask_times.add(self.clock_ns())

    def mark_depth(self) -> None:
        self._depth_times.add(self.clock_ns())

    def record_depth_error(self, exc: BaseException) -> None:
        self.depth_errors += 1
        self.last_depth_error = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]

    def snapshot(self) -> dict[str, object]:
        """Recent-window view for /ugv/perception/stats. Rates are per second."""
        now = self.clock_ns()
        span_s = min(self.window_s, (now - self._born_ns) / _NS)
        stage_ms: dict[str, float] = {}
        for name, window in self._stage_windows.items():
            window.trim(now)
            n = window.count()
            stage_ms[name] = round(window.total() / n / _MS, 3) if n else 0.0
        return {
            "mask_hz": self._hz(self._mask_times, now, span_s),
            "depth_hz": self._hz(self._depth_times, now, span_s),
            "stage_ms": stage_ms,
            "depth_errors": self.depth_errors,
            "last_depth_error": self.last_depth_error,
        }

    @staticmethod
    def _hz(window: _Window, now_ns: int, span_s: float) -> float:
        window.trim(now_ns)
        return round(window.count() / span_s, 3) if span_s > 0 else 0.0
