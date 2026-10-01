"""Safety arbiter decisions (pure Python, no ROS). Inputs here are test stimuli, not product data."""

from __future__ import annotations

from pathlib import Path

import pytest

from ugv_safety.arbiter_core import (
    ConfigError,
    Level,
    SafetyArbiter,
    SafetyConfig,
    load_config,
)

S = 1_000_000_000
CONFIG_YAML = Path(__file__).resolve().parents[2] / "config" / "safety" / "safety_timeouts.yaml"


def cfg(**kw) -> SafetyConfig:
    base = dict(
        publish_rate_hz=20.0, perception_s=0.5, localization_s=0.5, nav2_s=0.5, camera_s=1.0, candidate_s=0.5,
        max_linear=0.4, max_angular=0.8, ramp_on_hold=True, decel_linear=1.0, decel_angular=2.0,
    )
    base.update(kw)
    return SafetyConfig(**base)


def healthy(arb: SafetyArbiter, t: int) -> None:
    arb.on_camera(t, t)
    arb.on_perception_degraded(False, t)
    arb.on_pose_valid(True, t)
    arb.on_nav2_heartbeat(True, t)


def test_shipped_config_loads():
    c = load_config(CONFIG_YAML)
    assert (c.perception_s, c.localization_s, c.nav2_s) == (0.5, 0.5, 0.5)  # dev.md watchdog table
    assert c.ramp_on_hold is False  # architecture.md §3.1: zero twist, not a ramp


def test_shipped_config_drops_to_zero_at_once_on_a_hold():
    a = SafetyArbiter(load_config(CONFIG_YAML))
    healthy(a, 0)
    a.on_candidate(0.4, 0.8, 0)
    assert a.step(0).linear == 0.4
    t = int(0.05 * S)
    healthy(a, t)
    a.on_pose_valid(False, t)
    d = a.step(t)
    assert d.level is Level.DEGRADED and (d.linear, d.angular) == (0.0, 0.0)


def test_fresh_arbiter_holds_until_every_source_has_spoken():
    a = SafetyArbiter(cfg())
    a.on_candidate(0.3, 0.0, 0)
    d = a.step(0)
    assert d.level is Level.HEALTH and (d.linear, d.angular) == (0.0, 0.0)
    assert set(d.reasons) == {"camera_stale", "perception_stale", "localization_stale", "nav2_stale"}


def test_healthy_candidate_passes_through():
    a = SafetyArbiter(cfg())
    healthy(a, 0)
    a.on_candidate(0.3, -0.2, 0)
    d = a.step(0)
    assert d.level is Level.NAV2 and (d.linear, d.angular) == (0.3, -0.2)
    assert d.status == "L4 FORWARD"


def test_candidate_is_clamped_to_limits():
    a = SafetyArbiter(cfg())
    healthy(a, 0)
    a.on_candidate(5.0, -9.0, 0)
    d = a.step(0)
    assert (d.linear, d.angular) == (0.4, -0.8)


@pytest.mark.parametrize("lin,ang", [(float("nan"), 0.0), (0.1, float("inf")), (float("-inf"), 0.0)])
def test_non_finite_candidate_is_zero(lin, ang):
    a = SafetyArbiter(cfg())
    healthy(a, 0)
    a.on_candidate(lin, ang, 0)
    d = a.step(0)
    assert (d.linear, d.angular) == (0.0, 0.0) and d.reasons == ("candidate_invalid",)


def test_stale_candidate_is_never_repeated():
    a = SafetyArbiter(cfg())
    healthy(a, 0)
    a.on_candidate(0.3, 0.0, 0)
    t = int(0.6 * S)
    healthy(a, t)
    d = a.step(t)
    assert d.level is Level.NAV2 and (d.linear, d.angular) == (0.0, 0.0) and d.reasons == ("candidate_stale",)


def test_no_candidate_means_idle_zero():
    a = SafetyArbiter(cfg())
    healthy(a, 0)
    d = a.step(0)
    assert (d.linear, d.angular) == (0.0, 0.0) and d.reasons == ("no_candidate",)


def test_estop_zeroes_immediately_and_beats_everything():
    a = SafetyArbiter(cfg())
    healthy(a, 0)
    a.on_candidate(0.4, 0.8, 0)
    assert a.step(0).linear == 0.4
    a.on_estop(True, 0)
    d = a.step(int(0.05 * S))
    assert d.level is Level.ESTOP and (d.linear, d.angular) == (0.0, 0.0)  # no ramp at Level 1


def test_estop_stays_latched_until_explicit_release():
    a = SafetyArbiter(cfg())
    healthy(a, 0)
    a.on_candidate(0.3, 0.0, 0)
    a.on_estop(True, 0)
    for t in (0.1, 5.0, 60.0):  # operator link silent for a minute
        ns = int(t * S)
        healthy(a, ns)
        a.on_candidate(0.3, 0.0, ns)
        assert a.step(ns).level is Level.ESTOP
    a.on_estop(False, 61 * S)
    healthy(a, 61 * S)
    a.on_candidate(0.3, 0.0, 61 * S)
    assert a.step(61 * S).level is Level.NAV2


def test_invalid_pose_holds_level3():
    a = SafetyArbiter(cfg(ramp_on_hold=False))
    healthy(a, 0)
    a.on_pose_valid(False, 0)
    a.on_candidate(0.3, 0.0, 0)
    d = a.step(0)
    assert d.level is Level.DEGRADED and d.reasons == ("pose_invalid",) and d.linear == 0.0


def test_degraded_perception_holds_level3():
    a = SafetyArbiter(cfg(ramp_on_hold=False))
    healthy(a, 0)
    a.on_perception_degraded(True, 0)
    a.on_candidate(0.3, 0.0, 0)
    d = a.step(0)
    assert d.level is Level.DEGRADED and d.reasons == ("perception_degraded",) and d.linear == 0.0


@pytest.mark.parametrize(
    "silent,reason",
    [("camera", "camera_stale"), ("perception", "perception_stale"), ("pose", "localization_stale"), ("nav2", "nav2_stale")],
)
def test_each_silent_source_trips_level2(silent, reason):
    a = SafetyArbiter(cfg(ramp_on_hold=False))
    healthy(a, 0)
    t = int(1.5 * S)  # past every timeout
    if silent != "camera":
        a.on_camera(t, t)
    if silent != "perception":
        a.on_perception_degraded(False, t)
    if silent != "pose":
        a.on_pose_valid(True, t)
    if silent != "nav2":
        a.on_nav2_heartbeat(True, t)
    a.on_candidate(0.3, 0.0, t)
    d = a.step(t)
    assert d.level is Level.HEALTH and d.reasons == (reason,) and d.linear == 0.0


def test_nav2_reporting_down_trips_level2():
    a = SafetyArbiter(cfg(ramp_on_hold=False))
    healthy(a, 0)
    a.on_nav2_heartbeat(False, 0)
    a.on_candidate(0.3, 0.0, 0)
    d = a.step(0)
    assert d.level is Level.HEALTH and d.reasons == ("nav2_down",)


def test_latched_camera_info_with_an_old_stamp_is_not_a_live_camera():
    a = SafetyArbiter(cfg())
    healthy(a, 100 * S)
    a.on_camera(0, 100 * S)  # arrives now, but stamped 100 s ago (a latched sample)
    a.on_candidate(0.3, 0.0, 100 * S)
    assert a.step(100 * S).reasons == ("camera_stale",)


def test_priority_estop_over_health_over_degraded():
    a = SafetyArbiter(cfg(ramp_on_hold=False))
    a.on_pose_valid(False, 0)  # degraded + everything else unspoken (health)
    assert a.step(0).level is Level.HEALTH
    a.on_estop(True, 0)
    assert a.step(0).level is Level.ESTOP
    a.on_estop(False, 0)
    healthy(a, 0)
    a.on_pose_valid(False, 0)
    assert a.step(0).level is Level.DEGRADED


def test_hold_ramps_down_instead_of_stepping():
    a = SafetyArbiter(cfg(decel_linear=1.0, decel_angular=2.0))
    healthy(a, 0)
    a.on_candidate(0.4, 0.8, 0)
    assert a.step(0).linear == 0.4
    t = int(0.1 * S)
    healthy(a, t)
    a.on_pose_valid(False, t)
    d = a.step(t)
    assert d.level is Level.DEGRADED
    assert d.linear == pytest.approx(0.3) and d.angular == pytest.approx(0.6)
    seen = [d.linear]
    for i in range(2, 8):
        t = int(i * 0.1 * S)
        healthy(a, t)
        a.on_pose_valid(False, t)
        seen.append(a.step(t).linear)
    assert seen == sorted(seen, reverse=True) and seen[-1] == 0.0


def test_ramp_never_overshoots_through_zero_to_the_other_sign():
    a = SafetyArbiter(cfg(decel_linear=1.0))
    healthy(a, 0)
    a.on_candidate(-0.05, 0.0, 0)
    a.step(0)
    t = int(0.5 * S)  # one big step: would be -0.05 + 0.5 without the floor at zero
    healthy(a, t)
    a.on_pose_valid(False, t)
    assert a.step(t).linear == 0.0


def test_output_after_hold_follows_candidate_again():
    a = SafetyArbiter(cfg())
    healthy(a, 0)
    a.on_pose_valid(False, 0)
    a.step(0)
    t = int(0.2 * S)
    healthy(a, t)
    a.on_candidate(0.25, 0.0, t)
    assert a.step(t).linear == 0.25


def test_config_rejects_nonsense():
    with pytest.raises(ConfigError):
        cfg(nav2_s=0.0)
    with pytest.raises(ConfigError):
        cfg(max_linear=float("nan"))
    with pytest.raises(ConfigError):
        cfg(ramp_on_hold="yes")


def test_config_rejects_unknown_and_missing_keys(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text("publish_rate_hz: 20\ntimeouts: {perception_s: 0.5}\nlimits: {max_linear: 0.4}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
