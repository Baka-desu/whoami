"""Pure data types for the 4-tier command priority arbiter.

Authority: architecture.md §3.1, dev.md Dev 5 task 1. No ROS imports here —
the arbiter kernel is testable without rclpy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Twist2D:
    """Minimal 2D twist: differential-drive only needs linear.x and angular.z."""

    linear_x: float
    angular_z: float


ZERO_TWIST = Twist2D(0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ArbiterInputs:
    """One arbitration tick's worth of tier signals.

    watchdog_tripped/reasons come from WatchdogMonitor.evaluate() (Level 2:
    camera / perception mask / pose-TF / Nav2 heartbeat silence or staleness).
    perception_degraded and pose_valid are the honest self-reported flags from
    Dev 1 and Dev 2 (Level 3). candidate is Dev 4's Nav2 output (Level 4).
    """

    estop: bool
    watchdog_tripped: bool
    watchdog_reasons: tuple[str, ...]
    perception_degraded: bool
    pose_valid: bool
    candidate: Twist2D


@dataclass(frozen=True, slots=True)
class ArbiterDecision:
    """Result of one arbitration tick.

    level is the precedence tier that fired (1 = e-stop .. 4 = Nav2 passthrough).
    hold=True means the target is zero twist; the caller (node) is responsible
    for ramping the *published* twist toward that target rather than jumping.
    """

    level: int
    hold: bool
    reason: str
    output: Twist2D
