"""Goal registry: finite input, terminal states are final, preemption, bounded history."""

from __future__ import annotations

import math

import pytest

from ugv_api.goals import GoalRegistry, GoalState, state_from_status, yaw_to_quaternion


def test_create_rejects_non_finite():
    reg = GoalRegistry()
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            reg.create(bad, 0.0, 0.0, "map", 0)


def test_terminal_state_is_final():
    reg = GoalRegistry()
    g = reg.create(1.0, 2.0, 0.0, "map", 0)
    assert reg.active().id == g.id
    reg.update(g.id, state=GoalState.SUCCEEDED)
    reg.update(g.id, state=GoalState.EXECUTING)
    assert reg.get(g.id).state is GoalState.SUCCEEDED
    assert reg.active() is None


def test_preempt_cancels_older_live_goals():
    reg = GoalRegistry()
    a = reg.create(1.0, 0.0, 0.0, "map", 0)
    b = reg.create(2.0, 0.0, 0.0, "map", 1)
    reg.preempt_active(except_id=b.id)
    assert reg.get(a.id).state is GoalState.CANCELED
    assert reg.active().id == b.id


def test_history_is_bounded():
    reg = GoalRegistry(keep=2)
    first = reg.create(0.0, 0.0, 0.0, "map", 0)
    reg.create(0.0, 0.0, 0.0, "map", 1)
    reg.create(0.0, 0.0, 0.0, "map", 2)
    assert reg.get(first.id) is None


def test_status_codes_and_quaternion():
    assert state_from_status(4) is GoalState.SUCCEEDED
    assert state_from_status(5) is GoalState.CANCELED
    assert state_from_status(6) is GoalState.ABORTED
    assert state_from_status(0) is GoalState.FAILED  # UNKNOWN
    x, y, z, w = yaw_to_quaternion(math.pi / 2)
    assert (x, y) == (0.0, 0.0) and z == pytest.approx(math.sqrt(0.5)) and w == pytest.approx(math.sqrt(0.5))
