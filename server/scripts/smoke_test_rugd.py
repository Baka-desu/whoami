"""Confirm Dev 1's real compose_tick pipeline runs without ROS, using our own CUDA backend
for the real RUGD SegFormer checkpoint. Not used by the running server."""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.rugd import RugdSegformerAdapter, N_CLASSES
from ugv_perception.compose.load import load_compose_configs
from ugv_perception.compose.tick import compose_tick

TURING = Path(r"D:\Codes\whoami\turing")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# --- our real CUDA backend, satisfying RugdSegformerAdapter's `.run(blob) -> logits` need ---
model = SegformerForSemanticSegmentation.from_pretrained(str(TURING / "weights" / "rugd-segformer")).to(device).eval()
assert model.config.num_labels == N_CLASSES, (model.config.num_labels, N_CLASSES)


class TorchBackend:
    def run(self, blob: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            t = torch.from_numpy(blob).to(device)
            out = model(pixel_values=t).logits  # (1, 25, h/4, w/4)
        return out.cpu().numpy()


adapter = RugdSegformerAdapter(backend=TorchBackend(), input_hw=(640, 640), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
remap_table, gate_profile, freshness_profile = load_compose_configs(
    remap_path=TURING / "config" / "ontologies" / "rugd.yaml",
    gates_path=TURING / "config" / "perception" / "rugd.yaml",
    freshness_path=TURING / "config" / "perception" / "port.yaml",
)

img = Image.open(r"C:\Users\workt\AppData\Local\Temp\claude\d--Codes-whoami\4aa9afa9-ef2a-4898-97bc-6d2e955cdc89\scratchpad\real-cats.jpg").convert("RGB")
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
