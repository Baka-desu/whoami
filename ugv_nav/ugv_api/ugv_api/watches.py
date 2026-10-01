"""architecture.md §12 timeout table, evaluated for display and for the goal gate. Pure Python.

This is the operator's view, not the safety authority: the Dev 5 arbiter (ugv_safety, not implemented)
is what zeros /cmd_vel. Every watch fails closed: an input never received, or older than its limit,
is a fail, exactly like an explicit bad value.
"""

from __future__ import annotations

from dataclasses import dataclass

from ugv_api import state as k
from ugv_api.state import Sample, StateStore

# §12 rows, in table order.
WATCH_NAMES = ("camera", "perception", "localization", "tf", "nav2", "e_stop")


@dataclass(frozen=True)
class Timeouts:
    """Display limits in seconds. Enforcement limits belong to Dev 5's safety_timeouts.yaml."""

    camera: float = 0.5
    perception: float = 0.5
    localization: float = 0.5
    tf: float = 0.5
    nav2: float = 0.5

    def __post_init__(self) -> None:
        for name in ("camera", "perception", "localization", "tf", "nav2"):
            if not getattr(self, name) > 0:
                raise ValueError(f"timeout {name} must be > 0")


@dataclass(frozen=True)
class Watch:
    name: str
    ok: bool
    reason: str  # "ok" or why it trips
    age_s: float | None  # age of the input that decided it, None if never received


def _stale(sample: Sample, now_ns: int, limit: float, *, use_stamp: bool) -> str | None:
    age = sample.age_s(now_ns, use_stamp=use_stamp)
    if age > limit:
        return f"stale ({age:.2f} s > {limit:.2f} s)"
    if age < -limit:
        return f"stamp in the future ({-age:.2f} s)"
    return None


def _flag(
    store: StateStore,
    now_ns: int,
    name: str,
    key: str,
    topic: str,
    limit: float,
    *,
    good: bool,
    detail_key: str | None = None,
) -> Watch:
    """Watch for a Bool heartbeat topic: fresh and equal to `good`."""
    s = store.get(key)
    if s is None:
        return Watch(name, False, f"no {topic}", None)
    age = s.age_s(now_ns)
    why = _stale(s, now_ns, limit, use_stamp=False)
    if why is not None:
        return Watch(name, False, f"{topic} {why}", age)
    if bool(s.value) != good:
        detail = store.get(detail_key) if detail_key else None
        text = f"{topic}={str(bool(s.value)).lower()}"
        if detail is not None and str(detail.value):
            text += f" ({detail.value})"
        return Watch(name, False, text, age)
    return Watch(name, True, "ok", age)


def evaluate(store: StateStore, now_ns: int, timeouts: Timeouts, *, estop_asserted: bool) -> list[Watch]:
    """The six §12 watches. `estop_asserted` is the gateway's own assertion."""
    out: list[Watch] = []

    cam = store.get(k.CAMERA_INFO)
    if cam is None:
        out.append(Watch("camera", False, "no /camera/camera_info", None))
    else:
        why = _stale(cam, now_ns, timeouts.camera, use_stamp=True)
        age = cam.age_s(now_ns, use_stamp=True)
        out.append(Watch("camera", why is None, f"camera {why}" if why else "ok", age))

    per = _flag(
        store, now_ns, "perception", k.PERCEPTION_DEGRADED, "/ugv/perception_degraded",
        timeouts.perception, good=False,
    )
    if per.ok:
        mask = store.get(k.MASK)
        if mask is None:
            per = Watch("perception", False, "no /segmentation/mask", None)
        else:
            why = _stale(mask, now_ns, timeouts.perception, use_stamp=True)
            if why is not None:
                per = Watch("perception", False, f"mask {why}", mask.age_s(now_ns, use_stamp=True))
    out.append(per)

    out.append(
        _flag(
            store, now_ns, "localization", k.POSE_VALID, "/ugv/pose_valid", timeouts.localization,
            good=True, detail_key=k.LOCALIZATION_STATUS,
        )
    )

    tf = store.get(k.TF_MAP_BASE)
    if tf is None:
        out.append(Watch("tf", False, "map->base_link missing", None))
    else:
        why = _stale(tf, now_ns, timeouts.tf, use_stamp=True)
        out.append(Watch("tf", why is None, f"map->base_link {why}" if why else "ok", tf.age_s(now_ns, use_stamp=True)))

    out.append(
        _flag(
            store, now_ns, "nav2", k.NAV2_HEARTBEAT, "/ugv/nav2_heartbeat", timeouts.nav2,
            good=True, detail_key=k.NAV2_STATUS,
        )
    )

    seen = store.get(k.E_STOP)
    if estop_asserted:
        out.append(Watch("e_stop", False, "asserted by operator gateway", None))
    elif seen is not None and bool(seen.value):
        out.append(Watch("e_stop", False, "asserted on /ugv/e_stop", seen.age_s(now_ns)))
    else:
        out.append(Watch("e_stop", True, "ok", seen.age_s(now_ns) if seen else None))
    return out


def gate_reasons(watches: list[Watch]) -> list[str]:
    """Why a new goal must not be sent: any tripped §12 watch (an empty list means the gate is open)."""
    return [f"{w.name}: {w.reason}" for w in watches if not w.ok]
