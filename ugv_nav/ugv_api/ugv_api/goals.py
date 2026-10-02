"""NavigateToPose goal records exposed as /api/v1/navigation/goals. Pure Python.

The registry only tracks what the operator asked for and what Nav2 answered; Nav2 (Dev 4) owns the
goal itself. One goal runs at a time: Nav2's bt_navigator preempts the previous goal on a new one.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum


class GoalState(str, Enum):
    PENDING = "pending"  # sent, Nav2 has not answered yet
    REJECTED = "rejected"  # Nav2 refused it
    EXECUTING = "executing"
    CANCELING = "canceling"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"
    CANCELED = "canceled"
    FAILED = "failed"  # the gateway could not deliver it (e.g. the action server vanished)


TERMINAL = frozenset(
    {GoalState.REJECTED, GoalState.SUCCEEDED, GoalState.ABORTED, GoalState.CANCELED, GoalState.FAILED}
)

# action_msgs/msg/GoalStatus codes -> state
_STATUS = {
    1: GoalState.PENDING,  # ACCEPTED, not executing yet
    2: GoalState.EXECUTING,
    3: GoalState.CANCELING,
    4: GoalState.SUCCEEDED,
    5: GoalState.CANCELED,
    6: GoalState.ABORTED,
}


def state_from_status(code: int) -> GoalState:
    return _STATUS.get(int(code), GoalState.FAILED)


@dataclass(frozen=True)
class GoalRecord:
    id: str
    frame_id: str
    x: float
    y: float
    yaw: float
    created_ns: int
    state: GoalState = GoalState.PENDING
    distance_remaining: float | None = None
    recoveries: int | None = None
    error_code: int | None = None
    error_message: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) for a rotation of `yaw` radians about +z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class GoalRegistry:
    def __init__(self, keep: int = 50) -> None:
        if keep < 1:
            raise ValueError("keep must be >= 1")
        self._keep = keep
        self._lock = threading.Lock()
        self._goals: OrderedDict[str, GoalRecord] = OrderedDict()

    def create(self, x: float, y: float, yaw: float, frame_id: str, now_ns: int) -> GoalRecord:
        for name, v in (("x", x), ("y", y), ("yaw", yaw)):
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
        rec = GoalRecord(uuid.uuid4().hex, frame_id, float(x), float(y), float(yaw), now_ns)
        with self._lock:
            self._goals[rec.id] = rec
            while len(self._goals) > self._keep:
                self._goals.popitem(last=False)
        return rec

    def update(self, goal_id: str, **fields) -> GoalRecord | None:
        """Apply fields to a goal. A terminal state is final: later updates are ignored."""
        with self._lock:
            rec = self._goals.get(goal_id)
            if rec is None or rec.terminal:
                return rec
            rec = replace(rec, **fields)
            self._goals[goal_id] = rec
            return rec

    def get(self, goal_id: str) -> GoalRecord | None:
        with self._lock:
            return self._goals.get(goal_id)

    def active(self) -> GoalRecord | None:
        """Newest goal that has not reached a terminal state."""
        with self._lock:
            for rec in reversed(self._goals.values()):
                if not rec.terminal:
                    return rec
            return None

    def preempt_active(self, except_id: str) -> None:
        """Nav2 preempts the running goal on a new one; mark older live goals canceled."""
        with self._lock:
            for gid, rec in list(self._goals.items()):
                if gid != except_id and not rec.terminal:
                    self._goals[gid] = replace(rec, state=GoalState.CANCELED, error_message="preempted by a newer goal")
