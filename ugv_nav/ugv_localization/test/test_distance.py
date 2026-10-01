"""Distance travelled along /odom: jitter floor, jump rejection, source attribution, strict profile."""

from __future__ import annotations

from pathlib import Path

import pytest

from ugv_localization.odom import Basis, DistanceProfile, DistanceTracker, Source, load_distance_profile

_NS = 1_000_000_000
_MS = 1_000_000
_T0 = 100 * _NS
_PRODUCT = Path(__file__).resolve().parents[1] / "config" / "distance.yaml"


def _tracker(**over) -> DistanceTracker:
    base = dict(min_step_m=0.05, max_speed_mps=3.0, publish_rate_hz=5.0)
    base.update(over)
    t = DistanceTracker(DistanceProfile(**base))
    t.on_source(Source.VISUAL)
    return t


def _drive(t: DistanceTracker, xs, ys=None, *, t0: int = _T0, dt_ns: int = 100 * _MS) -> int:
    ys = ys if ys is not None else [0.0] * len(xs)
    for i, (x, y) in enumerate(zip(xs, ys)):
        t.on_odom(t0 + i * dt_ns, x, y)
    return t0 + len(xs) * dt_ns


def test_d1_straight_line_counts_full_length() -> None:
    t = _tracker()
    _drive(t, [0.1 * i for i in range(51)])  # 5 m at 1 m/s
    assert t.total_m == pytest.approx(5.0)
    assert t.basis is Basis.VISUAL


def test_d2_back_and_forth_counts_path_not_displacement() -> None:
    t = _tracker()
    xs = [0.1 * i for i in range(21)] + [2.0 - 0.1 * i for i in range(1, 21)]  # 2 m out, 2 m back
    _drive(t, xs)
    assert t.total_m == pytest.approx(4.0)


def test_d3_parked_jitter_never_adds_up() -> None:
    t = _tracker()
    xs = [0.02 if i % 2 else -0.02 for i in range(600)]  # ±2 cm wobble for a minute
    _drive(t, xs, [0.01 if i % 3 else -0.01 for i in range(600)])
    assert t.total_m == 0.0 and type(t.total_m) is float  # std_msgs/Float64 aborts on int
    assert t.basis is Basis.NONE


def test_d4_slow_motion_below_floor_per_sample_still_accumulates() -> None:
    t = _tracker()
    _drive(t, [0.005 * i for i in range(201)])  # 5 mm steps (0.05 m/s), 1 m total
    assert t.total_m == pytest.approx(1.0, abs=0.05)


def test_d5_jump_is_rejected_and_reanchored() -> None:
    t = _tracker()
    end = _drive(t, [0.1 * i for i in range(11)])  # 1 m
    t.on_odom(end, 20.0, 0.0)  # 19 m in 0.1 s: artefact
    _drive(t, [20.0 + 0.1 * i for i in range(1, 11)], t0=end + 100 * _MS)  # +1 m after
    assert t.total_m == pytest.approx(2.0)
    assert t.jumps_rejected == 1


def test_d6_duplicate_reordered_and_nonfinite_samples_ignored() -> None:
    t = _tracker()
    t.on_odom(_T0, 0.0, 0.0)
    t.on_odom(_T0 + _NS, 1.0, 0.0)
    t.on_odom(_T0 + _NS, 5.0, 0.0)  # same stamp
    t.on_odom(_T0 + _NS // 2, 9.0, 0.0)  # older stamp
    t.on_odom(_T0 + 2 * _NS, float("nan"), 0.0)
    t.on_odom(_T0 + 3 * _NS, 2.0, 0.0)
    assert t.total_m == pytest.approx(2.0)


def test_d7_attribution_and_basis() -> None:
    t = DistanceTracker(DistanceProfile(min_step_m=0.05, max_speed_mps=3.0, publish_rate_hz=5.0))
    end = _drive(t, [0.0, 0.5], dt_ns=_NS)  # before any odom_source message: unattributed
    assert t.basis is Basis.MIXED
    t.reset()
    t.on_source(Source.WHEEL)
    end = _drive(t, [0.0, 0.5, 1.0], t0=end, dt_ns=_NS)
    assert t.basis is Basis.WHEEL
    t.on_source(Source.VISUAL)
    _drive(t, [1.5, 2.0], t0=end, dt_ns=_NS)
    assert t.basis is Basis.MIXED
    assert t.by_source_m == {Source.WHEEL: pytest.approx(1.0), Source.VISUAL: pytest.approx(1.0)}


def test_d7b_startup_race_metres_go_to_the_first_announced_source() -> None:
    t = DistanceTracker(DistanceProfile(min_step_m=0.05, max_speed_mps=3.0, publish_rate_hz=5.0))
    end = _drive(t, [0.0, 0.5], dt_ns=_NS)  # /odom delivered before the latched odom_source
    t.on_source(Source.VISUAL)
    assert t.basis is Basis.VISUAL and t.by_source_m == {Source.VISUAL: pytest.approx(0.5)}
    t.on_source(Source.WHEEL)  # a later switch never re-attributes what was already counted
    _drive(t, [1.0], t0=end, dt_ns=_NS)
    assert t.by_source_m == {Source.VISUAL: pytest.approx(0.5), Source.WHEEL: pytest.approx(0.5)}


def test_d8_reset_zeroes_and_does_not_count_the_gap() -> None:
    t = _tracker()
    end = _drive(t, [0.1 * i for i in range(11)])
    t.reset()
    assert t.total_m == 0.0 and t.basis is Basis.NONE
    _drive(t, [5.0, 5.1, 5.2], t0=end)  # first sample after reset only anchors
    assert t.total_m == pytest.approx(0.2)
    assert t.basis is Basis.VISUAL  # source survives a reset


def test_d9_input_types() -> None:
    t = _tracker()
    with pytest.raises(TypeError):
        t.on_odom(1.5, 0.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        t.on_source("visual")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DistanceTracker({"min_step_m": 0.05})  # type: ignore[arg-type]


def test_d10_product_profile_loads_strictly(tmp_path: Path) -> None:
    p = load_distance_profile(_PRODUCT)
    assert p.min_step_m > 0.0 and p.max_speed_mps > 0.0 and p.publish_rate_hz > 0.0
    bad = tmp_path / "d.yaml"
    bad.write_text("min_step_m: 0.05\nmax_speed_mps: 3\npublish_rate_hz: 5.0\n", encoding="utf-8")
    with pytest.raises(TypeError):  # int, not float
        load_distance_profile(bad)
    bad.write_text("min_step_m: 0.05\nmax_speed_mps: 3.0\npublish_rate_hz: 5.0\nextra: 1.0\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_distance_profile(bad)
