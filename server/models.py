"""Loads the two real models once and exposes plain run_* functions.

Segmentation: nvidia/segformer-b2-finetuned-ade-512-512 (generic scene segmentation, remapped
into the canonical 3-class Perception Port via ontology/ade20k_remap.yaml). This is a stand-in
adapter: no pretrained RUGD SegFormer checkpoint exists publicly (see the PR discussion) -
training one on the RUGD dataset is dev.md's actual Dev 1 task, not a "connect the API" job.

Depth: depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf. dev.md/architecture.md name
"Depth Anything 3 Metric Large", but that model's repo (ByteDance-Seed/depth-anything-3) caps
Python at 3.13 and pulls in a full multi-view 3D-reconstruction stack (pycolmap, open3d, gsplat)
that has nothing to do with single-frame depth. This is the direct predecessor, integrates via
plain `transformers`, and is specifically the outdoor metric variant.
"""
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import (
    AutoImageProcessor, AutoModelForDepthEstimation,
    SegformerImageProcessor, SegformerForSemanticSegmentation,
)

SEG_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
REMAP_PATH = Path(__file__).parent / "ontology" / "ade20k_remap.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_seg_proc = None
_seg_model = None
_seg_lut = None  # ade20k class id -> canonical {0,1,2}, built from the remap YAML
_depth_proc = None
_depth_model = None


def _build_lut(id2label: dict[int, str]) -> np.ndarray:
    with open(REMAP_PATH, encoding="utf-8") as fh:
        remap = yaml.safe_load(fh)
    name_to_canon = {"traversable": 1, "unknown": 0, "hazard": 2}
    default = name_to_canon[remap["default"]]
    canon_of_name = {}
    for group in ("traversable", "unknown"):
        for name in remap.get(group, []):
            canon_of_name[name] = name_to_canon[group]
    lut = np.full(max(id2label) + 1, default, dtype=np.uint8)
    for idx, name in id2label.items():
        lut[idx] = canon_of_name.get(name.strip(), default)
    return lut


def load() -> None:
    """Load both models once. Called at server startup so /health is meaningful and the
    first real request isn't slow."""
    global _seg_proc, _seg_model, _seg_lut, _depth_proc, _depth_model
    if _seg_model is not None:
        return
    _seg_proc = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
    _seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID).to(DEVICE).eval()
    _seg_lut = _build_lut(_seg_model.config.id2label)
    _depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    _depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE).eval()


def ready() -> bool:
    return _seg_model is not None and _depth_model is not None


def run_segmentation(image: Image.Image, out_w: int, out_h: int) -> tuple[np.ndarray, float]:
    """Returns a (out_h, out_w) uint8 array with values strictly in {0,1,2}."""
    t0 = time.time()
    inputs = _seg_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = _seg_model(**inputs).logits
    raw = logits.argmax(dim=1)[0].cpu().numpy()
    canonical = _seg_lut[raw]
    resized = np.array(
        Image.fromarray(canonical, mode="L").resize((out_w, out_h), Image.NEAREST)
    )
    return resized.astype(np.uint8), (time.time() - t0) * 1000


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
