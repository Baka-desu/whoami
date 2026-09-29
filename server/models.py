"""Loads the real models once and exposes plain run_* functions.

Segmentation: Dev 1's real, tested Perception Port pipeline (turing/, installed here as the
`ugv_perception` library — see server/README.md for why it's imported rather than copied) driving
the real RUGD SegFormer-B5 checkpoint (JasonTStanley/RUGD-Segformer on Hugging Face, fine-tuned
from nvidia/segformer-b5-finetuned-ade-640-640 on the actual RUGD classes). `compose_tick()` is
Dev1's own remap + confidence-gate + freshness pipeline; this file only supplies the one piece
that didn't exist yet for this hardware: a CUDA PyTorch inference backend (Dev1's own backend
seam was stubbed for "later, NVIDIA" — turing/src/ugv_perception/backend/cuda_pytorch.py). That
stub, and its factory rejection, are left untouched: they're Dev1's own file with Dev1's own test
asserting the current "not on this box" behaviour, so the backend is implemented here instead and
wired directly into `RugdSegformerAdapter`.

Depth: depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf. dev.md/architecture.md name
"Depth Anything 3 Metric Large" (turing/'s own T08 targets the same model); that package's own
pyproject.toml caps Python at 3.13 (this machine runs 3.14) and pulls in a full multi-view
3D-reconstruction stack (pycolmap, open3d, gsplat) unrelated to single-frame depth. This is the
direct predecessor, integrates via plain `transformers`, and is specifically the outdoor variant.
"""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, SegformerForSemanticSegmentation

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.rugd import N_CLASSES, RugdSegformerAdapter
from ugv_perception.compose.load import load_compose_configs
from ugv_perception.compose.tick import compose_tick

TURING = Path(__file__).parent.parent / "turing"
RUGD_WEIGHTS = TURING / "weights" / "rugd-segformer"
RUGD_INPUT_HW = (640, 640)  # matches turing/config/adapters/rugd.yaml
RUGD_MEAN = (0.485, 0.456, 0.406)
RUGD_STD = (0.229, 0.224, 0.225)

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_rugd_model = None
_rugd_adapter: RugdSegformerAdapter | None = None
_remap_table = None
_gate_profile = None
_freshness_profile = None
_depth_proc = None
_depth_model = None


class _TorchRugdBackend:
    """Satisfies RugdSegformerAdapter's informal backend contract: .run(blob) -> raw logits.
    Not turing/src/ugv_perception/backend/cuda_pytorch.py - see the module docstring for why."""

    def run(self, blob: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            t = torch.from_numpy(blob).to(DEVICE)
            logits = _rugd_model(pixel_values=t).logits
        return logits.cpu().numpy()


def load() -> None:
    """Load both models once. Called at server startup so /health is meaningful and the
    first real request isn't slow."""
    global _rugd_model, _rugd_adapter, _remap_table, _gate_profile, _freshness_profile
    global _depth_proc, _depth_model
    if _rugd_model is not None:
        return
    if not (RUGD_WEIGHTS / "model.safetensors").is_file():
        raise RuntimeError(
            f"RUGD weights missing at {RUGD_WEIGHTS}. Fetch them first: "
            f"bash turing/scripts/fetch_rugd_segformer.sh"
        )
    _rugd_model = SegformerForSemanticSegmentation.from_pretrained(str(RUGD_WEIGHTS)).to(DEVICE).eval()
    if _rugd_model.config.num_labels != N_CLASSES:
        raise RuntimeError(f"checkpoint has {_rugd_model.config.num_labels} classes, expected {N_CLASSES}")
    _rugd_adapter = RugdSegformerAdapter(
        backend=_TorchRugdBackend(), input_hw=RUGD_INPUT_HW, mean=RUGD_MEAN, std=RUGD_STD,
    )
    _remap_table, _gate_profile, _freshness_profile = load_compose_configs(
        remap_path=TURING / "config" / "ontologies" / "rugd.yaml",
        gates_path=TURING / "config" / "perception" / "rugd.yaml",
        freshness_path=TURING / "config" / "perception" / "port.yaml",
    )

    _depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    _depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE).eval()


def ready() -> bool:
    return _rugd_model is not None and _depth_model is not None


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


def run_depth(image: Image.Image, out_w: int, out_h: int) -> tuple[np.ndarray, float]:
    """Returns a (out_h, out_w) float32 array of metric depth in metres."""
    t0 = time.time()
    inputs = _depth_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        raw = _depth_model(**inputs).predicted_depth
    full = torch.nn.functional.interpolate(
        raw.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False,
    ).squeeze().cpu().numpy()
    resized = np.array(
        Image.fromarray(full.astype(np.float32), mode="F").resize((out_w, out_h), Image.BILINEAR)
    )
    return resized.astype(np.float32), (time.time() - t0) * 1000
