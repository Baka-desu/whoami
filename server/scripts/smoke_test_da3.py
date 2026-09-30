"""Confirm Dev 1's real build_depth_channel (DA3-Metric-Large) runs end-to-end.
Not used by the running server.

Usage: .venv\\Scripts\\python scripts\\smoke_test_da3.py <path-to-a-photo>
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ugv_perception.backend.depth_live import build_depth_channel

TURING = Path(__file__).resolve().parents[2] / "turing"

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {sys.argv[0]} <path-to-a-photo>")

channel = build_depth_channel(TURING)
print("channel:", channel)
if channel is None:
    raise SystemExit("build_depth_channel returned None - weights not found")

img = Image.open(sys.argv[1]).convert("RGB")
rgb = np.asarray(img, dtype=np.uint8)
h, w = rgb.shape[:2]
# assumed intrinsics, same convention as the UI's assumedIntrinsics()
f = 0.87 * w
k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)

t0 = time.time()
depth_m, points = channel.maps(rgb, k)
print("latency ms:", round((time.time() - t0) * 1000, 1))
print("depth shape:", depth_m.shape, depth_m.dtype)
finite = depth_m[np.isfinite(depth_m)]
print("depth min/median/max (finite only):", finite.min(), np.median(finite), finite.max())
print("nan fraction:", np.isnan(depth_m).mean())
print("points shape:", points.shape)
print("OK")
