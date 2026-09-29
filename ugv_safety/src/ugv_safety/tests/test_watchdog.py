import pytest

from ugv_safety.watchdog.monitor import WatchdogMonitor
from ugv_safety.watchdog.table import WatchdogProfile

_NS = 1_000_000_000


def _profile() -> WatchdogProfile:
    return WatchdogProfile(timeouts_s={"camera": 0.5, "nav2_heartbeat": 0.5})


def test_never_seen_trips() -> None:
    wd = WatchdogMonitor(_profile())
    result = wd.evaluate(now_ns=1 * _NS)
    assert result.tripped
    assert any("never_seen" in r for r in result.reasons)


def test_fresh_touch_clears() -> None:
    wd = WatchdogMonitor(_profile())
    wd.touch("camera", now_ns=1 * _NS)
    wd.touch("nav2_heartbeat", now_ns=1 * _NS)
    result = wd.evaluate(now_ns=1 * _NS + int(0.1 * _NS))
    assert not result.tripped


def test_stale_touch_trips() -> None:
    wd = WatchdogMonitor(_profile())
    wd.touch("camera", now_ns=1 * _NS)
    wd.touch("nav2_heartbeat", now_ns=1 * _NS)
    result = wd.evaluate(now_ns=1 * _NS + int(0.6 * _NS))
    assert result.tripped
    assert any("camera" in r for r in result.reasons)
    assert any("nav2_heartbeat" in r for r in result.reasons)


def test_touch_rejects_unknown_watch() -> None:
    wd = WatchdogMonitor(_profile())
    with pytest.raises(KeyError):
        wd.touch("not_a_real_watch", now_ns=0)


def test_one_watch_stale_still_trips_whole_result() -> None:
    wd = WatchdogMonitor(_profile())
    wd.touch("camera", now_ns=1 * _NS)
    wd.touch("nav2_heartbeat", now_ns=1 * _NS + int(0.6 * _NS))
    result = wd.evaluate(now_ns=1 * _NS + int(0.6 * _NS))
    assert result.tripped
    assert any("camera" in r for r in result.reasons)
    assert not any("nav2_heartbeat" in r for r in result.reasons)
