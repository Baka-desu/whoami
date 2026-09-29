"""Odom source selector: wheel | visual | auto → one continuous odom->base_link. Clocks + poses."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ugv_localization.odom import (
    OdomSelector,
    SelectorProfile,
    Source,
    SourcePolicy,
    TfEdge,
    load_odom_select_config,
    needs_visual_odometry,
    parse_odom_source,
    rtabmap_subscribes_odom_info,
)

_NS = 1_000_000_000
_MS = 1_000_000
_T0 = 100 * _NS
_COV = tuple(0.01 if i in (0, 7, 14, 21, 28, 35) else 0.0 for i in range(36))
_LOST_COV = tuple(9999.0 if i in (0, 7, 14, 21, 28, 35) else 0.0 for i in range(36))
_PRODUCT = Path(__file__).resolve().parents[1] / "config" / "odom_select.yaml"


def _yaw_q(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _edge(stamp_ns: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> TfEdge:
    return TfEdge(stamp_ns=stamp_ns, parent="odom", child="base_link", translation=(x, y, 0.0), rotation=_yaw_q(yaw))


def _yaw(rot: tuple[float, float, float, float]) -> float:
    x, y, z, w = rot
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _sel(policy: SourcePolicy = SourcePolicy.AUTO, **over) -> OdomSelector:
    base = dict(wheel_timeout_s=0.25, visual_timeout_s=0.5, switch_back_hold_s=1.0, visual_lost_variance=1000.0)
    base.update(over)
    return OdomSelector(SelectorProfile(**base), policy)


def _wheel(s: OdomSelector, t: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, cov=_COV):
    return s.on_sample(Source.WHEEL, _edge(t, x, y, yaw), cov, now_ns=t)


def _visual(s: OdomSelector, t: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, cov=_COV, now=None):
    return s.on_sample(Source.VISUAL, _edge(t, x, y, yaw), cov, now_ns=t if now is None else now)


# --- forced policies ----------------------------------------------------------------------


def test_s1_wheel_forced_passes_wheel_through_unchanged() -> None:
    s = _sel(SourcePolicy.WHEEL)
    res = _wheel(s, _T0, x=1.0, y=2.0, yaw=0.3)
    out = res.output
    assert out is not None and out.source is Source.WHEEL
    assert out.stamp_ns == _T0  # never restamped
    assert out.translation == pytest.approx((1.0, 2.0, 0.0))
    assert _yaw(out.rotation) == pytest.approx(0.3)
    assert out.pose_covariance == _COV
    assert res.event is not None and res.event.current is Source.WHEEL and res.event.previous is None


def test_s2_wheel_forced_ignores_visual() -> None:
    s = _sel(SourcePolicy.WHEEL)
    res = _visual(s, _T0, x=5.0)
    assert res.output is None and res.dropped == "inactive_source"


def test_s3_visual_forced_passes_visual_and_ignores_wheel() -> None:
    s = _sel(SourcePolicy.VISUAL)
    assert _wheel(s, _T0, x=1.0).dropped == "inactive_source"
    out = _visual(s, _T0 + 10 * _MS, x=3.0).output
    assert out is not None and out.source is Source.VISUAL
    assert out.translation == pytest.approx((3.0, 0.0, 0.0))


# --- auto ---------------------------------------------------------------------------------


def test_s4_auto_prefers_wheel_when_both_alive() -> None:
    s = _sel()
    assert _wheel(s, _T0, x=1.0).output.source is Source.WHEEL
    assert _visual(s, _T0 + 10 * _MS, x=9.0).dropped == "inactive_source"
    assert _wheel(s, _T0 + 50 * _MS, x=1.1).output.source is Source.WHEEL


def test_s4b_auto_without_wheel_starts_on_visual() -> None:
    res = _visual(_sel(), _T0, x=2.0)
    assert res.output is not None and res.output.source is Source.VISUAL
    assert res.output.translation == pytest.approx((2.0, 0.0, 0.0))
    assert res.event is not None and (res.event.previous, res.event.reason) == (None, "start")


def test_s5_auto_falls_back_to_visual_without_a_jump() -> None:
    s = _sel()
    _wheel(s, _T0, x=2.0, y=1.0, yaw=0.5)
    _visual(s, _T0 + 10 * _MS, x=-7.0, y=3.0, yaw=-1.0)  # different origin, inactive
    # wheel silent > wheel_timeout_s; next visual sample takes over
    res = _visual(s, _T0 + 400 * _MS, x=-7.0, y=3.0, yaw=-1.0)
    assert res.event is not None
    assert (res.event.previous, res.event.current, res.event.reason) == (Source.WHEEL, Source.VISUAL, "wheel_lost")
    out = res.output
    assert out is not None and out.source is Source.VISUAL
    assert out.translation == pytest.approx((2.0, 1.0, 0.0), abs=1e-9)  # continues from last output
    assert _yaw(out.rotation) == pytest.approx(0.5)


def test_s6_visual_motion_is_applied_in_the_continued_frame() -> None:
    s = _sel()
    _wheel(s, _T0, x=0.0, y=0.0, yaw=math.pi / 2)  # facing +y in odom
    _visual(s, _T0 + 400 * _MS, x=10.0, y=0.0, yaw=0.0)  # takeover; visual frame faces +x
    out = _visual(s, _T0 + 500 * _MS, x=11.0, y=0.0, yaw=0.0).output  # 1 m forward in visual
    assert out is not None
    assert out.translation == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)  # forward = +y in odom
    assert _yaw(out.rotation) == pytest.approx(math.pi / 2)


def test_s7_switch_back_to_wheel_needs_hold_and_does_not_jump() -> None:
    s = _sel()
    _wheel(s, _T0, x=0.0)
    _visual(s, _T0 + 400 * _MS, x=0.0)  # takeover
    _visual(s, _T0 + 500 * _MS, x=0.5)  # moved 0.5 m (odom x = 0.5)
    # wheel comes back (its own frame drifted: reports x=3.0)
    assert _wheel(s, _T0 + 600 * _MS, x=3.0).dropped == "inactive_source"
    t = _T0 + 600 * _MS
    while t < _T0 + 1500 * _MS:  # both healthy, but wheel hold (1.0 s) not yet satisfied
        t += 50 * _MS
        assert _wheel(s, t, x=3.0).dropped == "inactive_source"
        assert _visual(s, t + 10 * _MS, x=0.5).output.source is Source.VISUAL
    t2 = _T0 + 1700 * _MS
    res = _wheel(s, t2, x=3.0)
    assert res.event is not None and res.event.reason == "wheel_recovered"
    assert res.output is not None and res.output.source is Source.WHEEL
    assert res.output.translation == pytest.approx((0.5, 0.0, 0.0), abs=1e-9)
    out = _wheel(s, t2 + 50 * _MS, x=3.2).output
    assert out.translation == pytest.approx((0.7, 0.0, 0.0), abs=1e-9)


def test_s8_dead_visual_switches_back_to_wheel_immediately() -> None:
    s = _sel()
    _wheel(s, _T0)
    _visual(s, _T0 + 400 * _MS)  # takeover
    res = _wheel(s, _T0 + 1000 * _MS, x=0.2)  # visual silent 0.6 s > visual_timeout_s
    assert res.event is not None and res.event.current is Source.WHEEL
    assert res.output is not None


def test_s9_lost_visual_sample_is_dropped_and_not_alive() -> None:
    s = _sel()
    _wheel(s, _T0)
    res = _visual(s, _T0 + 400 * _MS, cov=_LOST_COV)  # rgbd_odometry null / reset sample
    assert res.output is None and res.dropped == "visual_lost"
    # wheel alive again: still wheel, never switched
    assert _wheel(s, _T0 + 450 * _MS).output.source is Source.WHEEL


def test_s10_only_lost_samples_produce_nothing() -> None:
    s = _sel(SourcePolicy.VISUAL)
    for i in range(5):
        assert _visual(s, _T0 + i * 100 * _MS, cov=_LOST_COV).output is None


def test_s11_output_stamps_strictly_increase_across_switch() -> None:
    s = _sel()
    _wheel(s, _T0 + 300 * _MS)
    # DA3 latency: visual sample stamped before the last wheel output, received later
    res = _visual(s, _T0 + 200 * _MS, now=_T0 + 600 * _MS)
    assert res.output is None and res.dropped == "stale_stamp"
    assert _visual(s, _T0 + 700 * _MS, now=_T0 + 750 * _MS).output is not None


def test_s12_reset_clears_state() -> None:
    s = _sel()
    _wheel(s, _T0, x=4.0)
    s.reset()
    res = _wheel(s, _T0 - 50 * _NS, x=1.0)  # clock went backwards (bag loop)
    assert res.output is not None and res.output.translation == pytest.approx((1.0, 0.0, 0.0))
    assert res.event is not None and res.event.previous is None


def test_s13_bad_arguments() -> None:
    with pytest.raises(TypeError):
        OdomSelector("auto", SourcePolicy.AUTO)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _sel().on_sample("wheel", _edge(_T0), _COV, now_ns=_T0)  # type: ignore[arg-type]


# --- parsing + config ---------------------------------------------------------------------


def test_s14_parse_odom_source() -> None:
    assert parse_odom_source("auto") is SourcePolicy.AUTO
    assert parse_odom_source(" Wheel ") is SourcePolicy.WHEEL
    assert parse_odom_source("visual") is SourcePolicy.VISUAL
    with pytest.raises(ValueError, match="odom_source"):
        parse_odom_source("gps")


def test_s15_product_config_loads() -> None:
    cfg = load_odom_select_config(_PRODUCT)
    assert cfg.wheel_gate.odom_frame == "odom"
    assert cfg.visual_gate.odom_frame != cfg.wheel_gate.odom_frame  # rgbd_odometry label frame
    assert cfg.wheel_gate.base_frame == cfg.visual_gate.base_frame == "base_link"
    assert cfg.selector.wheel_timeout_s < cfg.selector.visual_timeout_s


def test_s16_config_strict(tmp_path: Path) -> None:
    text = _PRODUCT.read_text(encoding="utf-8")
    p = tmp_path / "s.yaml"
    p.write_text(text.replace("switch_back_hold_s: 1.0", "switch_back_hold_s: 1"), encoding="utf-8")
    with pytest.raises(TypeError, match="switch_back_hold_s"):
        load_odom_select_config(p)
    p.write_text(text + "\nsurprise: 1.0\n", encoding="utf-8")
    with pytest.raises(KeyError, match="surprise"):
        load_odom_select_config(p)


def test_s17_launch_decisions_per_policy() -> None:
    assert [needs_visual_odometry(p) for p in SourcePolicy] == [True, False, True]  # auto, wheel, visual
    assert [rtabmap_subscribes_odom_info(p) for p in SourcePolicy] == [False, False, True]
