"""Confirm Dev 1's real compose_tick pipeline runs end-to-end, through the same
build_live_adapter() factory models.py uses (not a hand-rolled backend). Not used by the
running server.

Usage: .venv\\Scripts\\python scripts\\smoke_test_rugd.py <path-to-an-outdoor-photo>
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.backend.rugd_live import build_live_adapter
from ugv_perception.compose.load import load_compose_configs
from ugv_perception.compose.tick import compose_tick

TURING = Path(__file__).resolve().parents[2] / "turing"

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {sys.argv[0]} <path-to-an-outdoor-photo>")

adapter = build_live_adapter(TURING)
remap_table, gate_profile, freshness_profile = load_compose_configs(
    remap_path=TURING / "config" / "ontologies" / "rugd.yaml",
    gates_path=TURING / "config" / "perception" / "rugd.yaml",
    freshness_path=TURING / "config" / "perception" / "port.yaml",
)

img = Image.open(sys.argv[1]).convert("RGB")
rgb = np.asarray(img, dtype=np.uint8)
now_ns = time.time_ns()
frame = ImageFrame(rgb=rgb, stamp_ns=now_ns, frame_id="cam_optical")

t0 = time.time()
out = compose_tick(
    frame=frame, now_ns=now_ns, adapter=adapter,
    remap_table=remap_table, gate_profile=gate_profile, freshness_profile=freshness_profile,
)
print("compose_tick latency ms:", round((time.time() - t0) * 1000, 1))
print("decision:", out.decision)
if out.mask is not None:
    classes = out.mask.classes
    print("classes shape/dtype:", classes.shape, classes.dtype, "unique:", np.unique(classes))
    print("class pct:", [round(100 * (classes == c).sum() / classes.size, 1) for c in (0, 1, 2)])
    print("confidence shape/dtype:", out.mask.confidence.shape, out.mask.confidence.dtype)
    print("valid:", out.mask.valid, "age_s:", out.mask.age_s)
else:
    print("NO MASK (degraded/invalid) — this is real port behaviour, not a crash")
print("OK")
