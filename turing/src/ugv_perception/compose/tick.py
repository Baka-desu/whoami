"""Port composition tick. Evaluate before infer. No rclpy. No engine imports."""

from __future__ import annotations

from dataclasses import dataclass

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import AdapterError
from ugv_perception.confidence.apply import apply as gate_apply
from ugv_perception.confidence.profile import GateProfile
from ugv_perception.freshness.evaluate import decide_publish, evaluate
from ugv_perception.freshness.profile import FreshnessProfile, PublishDecision
from ugv_perception.port.mask import CanonicalMask, make_mask
from ugv_perception.remap.apply import apply as remap_apply
from ugv_perception.remap.table import RemapTable


@dataclass(frozen=True, slots=True)
class ComposeOut:
    decision: PublishDecision
    mask: CanonicalMask | None


def compose_tick(
    *,
    frame: ImageFrame | None,
    now_ns: int,
    adapter: object,
    remap_table: RemapTable,
    gate_profile: GateProfile,
    freshness_profile: FreshnessProfile,
    now_ns_after: int | None = None,
) -> ComposeOut:
    if frame is None:
        return ComposeOut(
            decision=PublishDecision(valid=False, degraded=True, publish_mask=False),
            mask=None,
        )

    result = evaluate(freshness_profile, frame.stamp_ns, now_ns)
    if result.time_degraded:
        decision = decide_publish(result, False, False)
        return ComposeOut(decision=decision, mask=None)

    adapter_error = False
    collapse = False
    classes = None
    conf = None
    raw = None
    try:
        raw = adapter.infer(frame)
    except AdapterError:
        adapter_error = True
        raw = None
    except Exception:
        adapter_error = True
        raw = None

    if raw is not None:
        if raw.stamp_ns != frame.stamp_ns or raw.frame_id != frame.frame_id:
            adapter_error = True
            raw = None

    if raw is not None:
        classes, scores = remap_apply(
            remap_table,
            adapter_id=raw.adapter_id,
            label_ids=raw.label_ids,
            id_to_name=raw.id_to_name,
            raw_scores=raw.raw_scores,
        )
        classes, conf, _frac, collapse = gate_apply(
            gate_profile,
            adapter_id=raw.adapter_id,
            classes=classes,
            raw_scores=scores,
            runner_up=None,
        )

    now_after = now_ns if now_ns_after is None else now_ns_after
    result = evaluate(freshness_profile, frame.stamp_ns, now_after)
    decision = decide_publish(result, collapse, adapter_error)

    if not decision.publish_mask:
        return ComposeOut(decision=decision, mask=None)

    mask = make_mask(
        stamp_ns=frame.stamp_ns,
        frame_id=frame.frame_id,
        classes=classes,
        confidence=conf,
        producer_ok=decision.valid,
        age_s=result.age_s,
        scale=1.0,
        source_hw=(int(frame.rgb.shape[0]), int(frame.rgb.shape[1])),
    )
    return ComposeOut(decision=decision, mask=mask)
