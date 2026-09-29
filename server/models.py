"""Loads the real models once and exposes plain run_* functions.

Both segmentation and depth now run through Dev 1's real, tested Perception pipeline (turing/,
installed here as the `ugv_perception` library) - not a reimplementation, and not a substitute
model. `build_live_adapter()` and `build_depth_channel()` are Dev 1's own top-level factories
(turing/src/ugv_perception/backend/rugd_live.py, depth_live.py), which pick Intel OpenVINO GPU,
then NVIDIA CUDA safetensors, then OpenVINO CPU (turing/src/ugv_perception/backend/device.py -
Dev 1's own code, added after this server's first version, which had to stand in a CUDA backend
of its own; that stand-in is gone now that Dev 1's real one exists upstream).

Segmentation: the real RUGD SegFormer-B5 checkpoint (JasonTStanley/RUGD-Segformer on Hugging
Face), through Dev 1's real compose_tick() (remap + confidence gate + freshness - see
run_segmentation()'s docstring for the one adaptation needed for a REST call instead of a ROS tick
loop).

Depth: the real Depth Anything 3 Metric Large checkpoint (depth-anything/DA3METRIC-LARGE), through
Dev 1's real DepthChannel. Getting the model class to build needs only `omegaconf`, `addict` and
`einops` - not the full depth-anything-3 pip package (which requires Python <=3.13 and pulls in a
whole multi-view 3D-reconstruction stack unrelated to this). See server/README.md for the install
step (`pip install --no-deps --ignore-requires-python -e <clone>`).
"""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.backend.depth_live import DepthChannel, build_depth_channel
from ugv_perception.backend.rugd_live import build_live_adapter
from ugv_perception.compose.load import load_compose_configs
from ugv_perception.compose.tick import compose_tick

TURING = Path(__file__).parent.parent / "turing"

_rugd_adapter = None
_remap_table = None
_gate_profile = None
_freshness_profile = None
_depth_channel: DepthChannel | None = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load() -> None:
    """Load both models once. Called at server startup so /health is meaningful and the
    first real request isn't slow."""
    global _rugd_adapter, _remap_table, _gate_profile, _freshness_profile, _depth_channel
    if _rugd_adapter is not None:
        return
    _rugd_adapter = build_live_adapter(TURING)
    _remap_table, _gate_profile, _freshness_profile = load_compose_configs(
        remap_path=TURING / "config" / "ontologies" / "rugd.yaml",
        gates_path=TURING / "config" / "perception" / "rugd.yaml",
        freshness_path=TURING / "config" / "perception" / "port.yaml",
    )
    _depth_channel = build_depth_channel(TURING)
    if _depth_channel is None:
        raise RuntimeError(
            "DA3-Metric-Large weights missing. Fetch them first: "
            "bash turing/scripts/fetch_da3metric_large.sh"
        )


def ready() -> bool:
    return _rugd_adapter is not None and _depth_channel is not None


def run_segmentation(
    image: Image.Image, out_w: int, out_h: int, *, streaming: bool,
) -> tuple[np.ndarray, np.ndarray, bool, bool, float]:
    """Runs the real RUGD adapter through Dev 1's real compose_tick (remap + confidence gate +
    freshness). Returns (classes, confidence, valid, degraded, latency_ms), each at (out_h,out_w).

    Freshness: `compose_tick`'s own before/after check (`now_ns_after`) is built for a continuous
    ROS tick loop, where a backed-up queue makes the *next* frame's `now_ns` itself late - the
    caller can't inject a real post-inference timestamp before calling a synchronous function that
    hasn't run yet, so we don't try. Instead we apply architecture.md's exact rule (age >
    perception_max_age -> degraded) against this call's own real measured latency, for a live
    stream only - a stale/slow mask must not present as current (§8.4/§8.6), and this is the same
    finding already surfaced for depth on this hardware. A single uploaded/captured still isn't
    part of the live product's freshness contract at all (there's no "stale" concept for a one-off
    analysis), so latency alone never degrades it.
    """
    t0 = time.time()
    rgb = np.asarray(image, dtype=np.uint8)
    stamp_ns = time.time_ns()
    frame = ImageFrame(rgb=rgb, stamp_ns=stamp_ns, frame_id="workbench_upload")

    out = compose_tick(
        frame=frame, now_ns=stamp_ns, adapter=_rugd_adapter,
        remap_table=_remap_table, gate_profile=_gate_profile, freshness_profile=_freshness_profile,
    )
    latency_ms = (time.time() - t0) * 1000

    if out.mask is None:
        # §8.6 fail-safe: never fabricate traversable/hazard when the real port had nothing to
        # say (adapter failure, or too few confidently-known pixels). All-unknown is the one
        # always-safe claim (unknown inflates, never free).
        classes = np.zeros((out_h, out_w), dtype=np.uint8)
        confidence = np.zeros((out_h, out_w), dtype=np.float32)
        return classes, confidence, False, True, latency_ms

    stale = streaming and (latency_ms / 1000.0) > _freshness_profile.perception_max_age
    valid = out.decision.valid and not stale
    degraded = out.decision.degraded or stale

    classes = np.array(Image.fromarray(out.mask.classes, mode="L").resize((out_w, out_h), Image.NEAREST), dtype=np.uint8)
    confidence = np.array(
        Image.fromarray(out.mask.confidence, mode="F").resize((out_w, out_h), Image.BILINEAR), dtype=np.float32,
    )
    return classes, confidence, valid, degraded, latency_ms


def run_depth(
    image: Image.Image, out_w: int, out_h: int, k: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Runs Dev 1's real DepthChannel (DA3-Metric-Large). Returns a (out_h, out_w) float32 array
    of metric depth in metres. `k` is the 3x3 camera intrinsics matrix (real or assumed - honesty
    about which is the UI's job, not this server's)."""
    t0 = time.time()
    rgb = np.asarray(image, dtype=np.uint8)
    depth_m, _points = _depth_channel.maps(rgb, k)
    # Sky/holes come back as NaN (no measurable depth). Treat as far, matching the mock's own
    # far-clamp convention, so downstream min/median/max and sorting never see a NaN.
    depth_m = np.nan_to_num(depth_m, nan=30.0, posinf=30.0, neginf=0.0)
    resized = np.array(
        Image.fromarray(depth_m.astype(np.float32), mode="F").resize((out_w, out_h), Image.BILINEAR)
    )
    return resized.astype(np.float32), (time.time() - t0) * 1000
