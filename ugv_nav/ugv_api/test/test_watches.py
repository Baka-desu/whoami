"""§12 watch table: every row fails closed (missing, stale, bad value), and the goal gate follows it."""

from __future__ import annotations

import pytest

from ugv_api import state as k
from ugv_api.state import StateStore
from ugv_api.watches import WATCH_NAMES, Timeouts, evaluate, gate_reasons

S = 1_000_000_000
NOW = 100 * S


def healthy() -> StateStore:
    st = StateStore()
    st.put(k.CAMERA_INFO, None, NOW, NOW - S // 10)
    st.put(k.MASK, None, NOW, NOW - S // 10)
    st.put(k.PERCEPTION_DEGRADED, False, NOW)
    st.put(k.POSE_VALID, True, NOW)
    st.put(k.TF_MAP_BASE, None, NOW, NOW - S // 20)
    st.put(k.NAV2_HEARTBEAT, True, NOW)
    return st


def by_name(st: StateStore, estop: bool = False):
    return {w.name: w for w in evaluate(st, NOW, Timeouts(), estop_asserted=estop)}


def test_table_order_and_all_ok_when_healthy():
    ws = evaluate(healthy(), NOW, Timeouts(), estop_asserted=False)
    assert tuple(w.name for w in ws) == WATCH_NAMES
    assert all(w.ok for w in ws), [w for w in ws if not w.ok]
    assert gate_reasons(ws) == []


def test_empty_store_trips_every_input_watch():
    ws = by_name(StateStore())
    for name in ("camera", "perception", "localization", "tf", "nav2"):
        assert not ws[name].ok and ws[name].age_s is None
    assert ws["e_stop"].ok  # nothing asserted


@pytest.mark.parametrize(
    "key,value,stamp,watch",
    [
        (k.CAMERA_INFO, None, NOW - S, "camera"),
        (k.MASK, None, NOW - S, "perception"),
        (k.TF_MAP_BASE, None, NOW - S, "tf"),
    ],
)
def test_stale_stamp_trips(key, value, stamp, watch):
    st = healthy()
    st.put(key, value, NOW, stamp)  # received now, but the source stamp is 1 s old (§8.4 age = now - stamp)
    w = by_name(st)[watch]
    assert not w.ok and "stale" in w.reason


def test_future_stamp_trips():
    st = healthy()
    st.put(k.CAMERA_INFO, None, NOW, NOW + 2 * S)
    assert "future" in by_name(st)["camera"].reason


@pytest.mark.parametrize(
    "key,bad,watch,detail_key,detail",
    [
        (k.PERCEPTION_DEGRADED, True, "perception", None, None),
        (k.POSE_VALID, False, "localization", k.LOCALIZATION_STATUS, "tf_missing"),
        (k.NAV2_HEARTBEAT, False, "nav2", k.NAV2_STATUS, "controller_server stale"),
    ],
)
def test_bad_flag_trips_with_reason(key, bad, watch, detail_key, detail):
    st = healthy()
    st.put(key, bad, NOW)
    if detail_key:
        st.put(detail_key, detail, NOW)
    w = by_name(st)[watch]
    assert not w.ok
    if detail:
        assert detail in w.reason


@pytest.mark.parametrize("key,watch", [(k.POSE_VALID, "localization"), (k.NAV2_HEARTBEAT, "nav2"),
                                       (k.PERCEPTION_DEGRADED, "perception")])
def test_heartbeat_that_stops_trips(key, watch):
    st = healthy()
    st.put(key, st.get(key).value, NOW - S)  # last good value, received 1 s ago
    assert not by_name(st)[watch].ok


def test_estop_from_gateway_or_graph_trips_and_closes_gate():
    assert not by_name(healthy(), estop=True)["e_stop"].ok
    st = healthy()
    st.put(k.E_STOP, True, NOW)
    ws = evaluate(st, NOW, Timeouts(), estop_asserted=False)
    assert gate_reasons(ws) == ["e_stop: asserted on /ugv/e_stop"]
    st.put(k.E_STOP, False, NOW)
    assert by_name(st)["e_stop"].ok


def test_timeouts_must_be_positive():
    with pytest.raises(ValueError):
        Timeouts(nav2=0.0)
