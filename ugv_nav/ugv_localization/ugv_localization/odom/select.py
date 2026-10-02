"""Odometry source selector → the one continuous odom->base_link edge (mindmap D6, D5b).

Sources, each already passed through its own OdomGate:
  * wheel  — Dev 5 /wheel/odom (metric, smooth, high rate)
  * visual — RTAB-Map rgbd_odometry on RGB + DA3 depth (rate = depth rate)

Policy (launch arg odom_source):
  * wheel / visual — only that source; the other is ignored.
  * auto — wheel while it is alive; visual when wheel is silent for wheel_timeout_s.
    Back to wheel after switch_back_hold_s of healthy wheel (hysteresis), or immediately if
    visual is dead too.

Continuity: output = offset_s · raw_s. On every switch the new source is re-anchored so the
first output equals the last one — the odom frame never jumps; drift lands in map->odom where
RTAB-Map corrects it. Motion during the handover gap (≤ wheel_timeout_s) is absorbed there too.

rgbd_odometry marks lost frames (publish_null_when_lost, and the first frame after an auto reset)
with a huge covariance: those are dropped, never re-anchored on. Stamps are never rewritten; an
output older than the previous output (DA3 latency across a switch) is dropped. No ROS imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ugv_localization.common.checks import NS_PER_S, require_positive_float, require_stamp
from ugv_localization.common.yamlio import load_yaml_profile, require_exact_keys
from ugv_localization.odom.gate import OdomGateProfile, TfEdge, odom_gate_profile_from_mapping

_COV_DIAG = (0, 7, 14, 21, 28, 35)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # x, y, z, w


class Source(str, Enum):
    WHEEL = "wheel"
    VISUAL = "visual"


class SourcePolicy(str, Enum):
    AUTO = "auto"
    WHEEL = "wheel"
    VISUAL = "visual"


def parse_odom_source(text: str) -> SourcePolicy:
    low = str(text).strip().lower()
    for policy in SourcePolicy:
        if policy.value == low:
            return policy
    raise ValueError(f"odom_source must be one of auto|wheel|visual, got {text!r}")


def needs_visual_odometry(policy: SourcePolicy) -> bool:
    """rgbd_odometry must run for auto (fallback) and visual; wheel-only never starts it."""
    return policy is not SourcePolicy.WHEEL


def rtabmap_subscribes_odom_info(policy: SourcePolicy) -> bool:
    """OdomInfo matches /odom stamps only when visual odometry is the sole source."""
    return policy is SourcePolicy.VISUAL


@dataclass(frozen=True, slots=True)
class SelectorProfile:
    wheel_timeout_s: float
    visual_timeout_s: float
    switch_back_hold_s: float
    visual_lost_variance: float


@dataclass(frozen=True, slots=True)
class OdomSelectConfig:
    wheel_gate: OdomGateProfile
    visual_gate: OdomGateProfile
    selector: SelectorProfile


@dataclass(frozen=True, slots=True)
class SelectedOdom:
    source: Source
    stamp_ns: int
    translation: Vec3
    rotation: Quat
    pose_covariance: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SwitchEvent:
    stamp_ns: int
    previous: Source | None
    current: Source
    reason: str  # start | wheel_lost | wheel_recovered | visual_lost


@dataclass(frozen=True, slots=True)
class SelectResult:
    output: SelectedOdom | None
    event: SwitchEvent | None
    dropped: str | None  # inactive_source | visual_lost | stale_stamp


def load_odom_select_config(path: str | Path, overlay_path: str | Path | None = None) -> OdomSelectConfig:
    data = load_yaml_profile(path, overlay_path)
    where = str(path)
    require_exact_keys(data, {"wheel_gate", "visual_gate", "selector"}, where=where)
    sel = data["selector"]
    if not isinstance(sel, dict):
        raise ValueError(f"{where}: selector must be a mapping")
    names = ("wheel_timeout_s", "visual_timeout_s", "switch_back_hold_s", "visual_lost_variance")
    require_exact_keys(sel, set(names), where=f"{where}:selector")
    return OdomSelectConfig(
        wheel_gate=odom_gate_profile_from_mapping(data["wheel_gate"], where=f"{where}:wheel_gate"),
        visual_gate=odom_gate_profile_from_mapping(data["visual_gate"], where=f"{where}:visual_gate"),
        selector=SelectorProfile(
            **{n: require_positive_float(sel[n], name=n, allow_zero=n == "switch_back_hold_s") for n in names}
        ),
    )


# --------------------------------------------------------------------------- SE(3) helpers
def _qmul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _qconj(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def _rotate(q: Quat, v: Vec3) -> Vec3:
    x, y, z, _ = _qmul(_qmul(q, (v[0], v[1], v[2], 0.0)), _qconj(q))
    return (x, y, z)


def _compose(a: tuple[Vec3, Quat], b: tuple[Vec3, Quat]) -> tuple[Vec3, Quat]:
    rt = _rotate(a[1], b[0])
    q = _qmul(a[1], b[1])
    n = math.sqrt(sum(c * c for c in q))
    return ((a[0][0] + rt[0], a[0][1] + rt[1], a[0][2] + rt[2]), tuple(c / n for c in q))  # type: ignore[return-value]


def _inverse(p: tuple[Vec3, Quat]) -> tuple[Vec3, Quat]:
    qi = _qconj(p[1])
    t = _rotate(qi, p[0])
    return ((-t[0], -t[1], -t[2]), qi)


_IDENTITY: tuple[Vec3, Quat] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class OdomSelector:
    def __init__(self, profile: SelectorProfile, policy: SourcePolicy) -> None:
        if type(profile) is not SelectorProfile:
            raise TypeError("profile must be a SelectorProfile")
        if type(policy) is not SourcePolicy:
            raise TypeError("policy must be a SourcePolicy")
        self._p = profile
        self._policy = policy
        self._timeout_ns = {
            Source.WHEEL: int(profile.wheel_timeout_s * NS_PER_S),
            Source.VISUAL: int(profile.visual_timeout_s * NS_PER_S),
        }
        self.reset()

    @property
    def active(self) -> Source | None:
        return self._active

    def reset(self) -> None:
        """Call when the clock jumps backwards (bag loop / sim reset)."""
        self._active: Source | None = None
        self._offset = _IDENTITY
        self._last_out: tuple[Vec3, Quat] | None = None
        self._last_out_ns: int | None = None
        self._last_rx_ns: dict[Source, int] = {}
        self._healthy_since_ns: dict[Source, int] = {}

    def _alive(self, source: Source, now_ns: int) -> bool:
        rx = self._last_rx_ns.get(source)
        return rx is not None and now_ns - rx <= self._timeout_ns[source]

    def _desired(self, now_ns: int) -> Source | None:
        if self._policy is SourcePolicy.WHEEL:
            return Source.WHEEL
        if self._policy is SourcePolicy.VISUAL:
            return Source.VISUAL
        wheel = self._alive(Source.WHEEL, now_ns)
        visual = self._alive(Source.VISUAL, now_ns)
        if self._active is Source.VISUAL:
            held = now_ns - self._healthy_since_ns.get(Source.WHEEL, now_ns) >= int(
                self._p.switch_back_hold_s * NS_PER_S
            )
            return Source.WHEEL if wheel and (held or not visual) else Source.VISUAL
        # The incoming sample has already marked its own source alive, so one of the two is.
        return Source.WHEEL if wheel else Source.VISUAL

    def _switch_reason(self, new: Source, now_ns: int) -> str:
        if self._active is None:
            return "start"
        if new is Source.VISUAL:
            return "wheel_lost"
        return "wheel_recovered" if self._alive(Source.VISUAL, now_ns) else "visual_lost"

    def on_sample(
        self, source: Source, edge: TfEdge, pose_covariance: tuple[float, ...], now_ns: int
    ) -> SelectResult:
        if type(source) is not Source:
            raise TypeError("source must be a Source")
        stamp = require_stamp(edge.stamp_ns, name="edge.stamp_ns")
        now_ns = require_stamp(now_ns, name="now_ns")

        if source is Source.VISUAL and any(
            pose_covariance[i] >= self._p.visual_lost_variance for i in _COV_DIAG
        ):
            return SelectResult(None, None, "visual_lost")

        rx = self._last_rx_ns.get(source)
        if rx is None or now_ns - rx > self._timeout_ns[source]:
            self._healthy_since_ns[source] = now_ns  # new healthy streak after a gap
        self._last_rx_ns[source] = now_ns

        desired = self._desired(now_ns)
        if source is not self._active and source is not desired:
            return SelectResult(None, None, "inactive_source")
        if self._last_out_ns is not None and stamp <= self._last_out_ns:
            return SelectResult(None, None, "stale_stamp")

        raw = (edge.translation, edge.rotation)
        event: SwitchEvent | None = None
        if source is not self._active:
            event = SwitchEvent(stamp, self._active, source, self._switch_reason(source, now_ns))
            self._offset = _IDENTITY if self._last_out is None else _compose(self._last_out, _inverse(raw))
            self._active = source

        out = _compose(self._offset, raw)
        self._last_out = out
        self._last_out_ns = stamp
        return SelectResult(
            SelectedOdom(source, stamp, out[0], out[1], tuple(pose_covariance)), event, None
        )
