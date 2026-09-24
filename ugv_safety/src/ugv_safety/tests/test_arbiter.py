from ugv_safety.arbiter.decide import decide
from ugv_safety.arbiter.types import ArbiterInputs, Twist2D

MOVING = Twist2D(0.5, 0.1)


def _inputs(**overrides: object) -> ArbiterInputs:
    base = dict(
        estop=False,
        watchdog_tripped=False,
        watchdog_reasons=(),
        perception_degraded=False,
        pose_valid=True,
        candidate=MOVING,
    )
    base.update(overrides)
    return ArbiterInputs(**base)


def test_all_clear_forwards_candidate() -> None:
    d = decide(_inputs())
    assert d.level == 4
    assert not d.hold
    assert d.output == MOVING


def test_estop_beats_everything() -> None:
    d = decide(_inputs(estop=True, watchdog_tripped=True, perception_degraded=True, pose_valid=False))
    assert d.level == 1
    assert d.hold
    assert d.output == Twist2D(0.0, 0.0)


def test_watchdog_beats_degraded_and_nav2() -> None:
    d = decide(_inputs(watchdog_tripped=True, watchdog_reasons=("camera:stale(1.0s>0.5s)",)))
    assert d.level == 2
    assert d.hold
    assert "camera" in d.reason


def test_perception_degraded_holds() -> None:
    d = decide(_inputs(perception_degraded=True))
    assert d.level == 3
    assert d.hold
    assert d.reason == "perception_degraded"


def test_invalid_pose_holds() -> None:
    d = decide(_inputs(pose_valid=False))
    assert d.level == 3
    assert d.hold
    assert d.reason == "pose_invalid"


def test_degraded_beats_stale_pose_reason_is_perception() -> None:
    d = decide(_inputs(perception_degraded=True, pose_valid=False))
    assert d.reason == "perception_degraded"


def test_hold_always_outputs_zero_twist() -> None:
    for kwargs in (
        {"estop": True},
        {"watchdog_tripped": True, "watchdog_reasons": ("x",)},
        {"perception_degraded": True},
        {"pose_valid": False},
    ):
        d = decide(_inputs(**kwargs))
        assert d.hold
        assert d.output == Twist2D(0.0, 0.0)
