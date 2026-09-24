"""4-tier command priority arbiter kernel (architecture.md §3.1).

Precedence, highest wins:
  1. E-stop / operator kill          -> zero twist
  2. System-health watchdog trip     -> zero twist
  3. Perception-degraded / invalid pose -> zero twist (hold)
  4. Nav2 candidate                  -> forwarded

This function is pure: same inputs -> same decision, no clock reads, no I/O.
"""

from __future__ import annotations

from ugv_safety.arbiter.types import ArbiterDecision, ArbiterInputs, ZERO_TWIST


def decide(inputs: ArbiterInputs) -> ArbiterDecision:
    if inputs.estop:
        return ArbiterDecision(level=1, hold=True, reason="estop", output=ZERO_TWIST)

    if inputs.watchdog_tripped:
        reason = "watchdog:" + ",".join(inputs.watchdog_reasons)
        return ArbiterDecision(level=2, hold=True, reason=reason, output=ZERO_TWIST)

    if inputs.perception_degraded or not inputs.pose_valid:
        reason = "perception_degraded" if inputs.perception_degraded else "pose_invalid"
        return ArbiterDecision(level=3, hold=True, reason=reason, output=ZERO_TWIST)

    return ArbiterDecision(level=4, hold=False, reason="nav2", output=inputs.candidate)
