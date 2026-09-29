"""One-off: confirm both real models load and run on the GPU end-to-end. Not used by the server."""
import time
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor, AutoModelForDepthEstimation,
    SegformerImageProcessor, SegformerForSemanticSegmentation,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

img = Image.open(r"C:\Users\workt\AppData\Local\Temp\claude\d--Codes-whoami\4aa9afa9-ef2a-4898-97bc-6d2e955cdc89\scratchpad\test-photo.png").convert("RGB")
print("image size:", img.size)

print("\n--- segmentation ---")
seg_proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512").to(device).eval()
t0 = time.time()
inputs = seg_proc(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    logits = seg_model(**inputs).logits
pred = logits.argmax(dim=1)[0].cpu().numpy()
print("seg latency ms:", round((time.time() - t0) * 1000, 1))
print("seg output shape:", pred.shape, "unique classes:", np.unique(pred))

print("\n--- depth ---")
depth_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf").to(device).eval()
t0 = time.time()
inputs = depth_proc(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    out = depth_model(**inputs).predicted_depth
depth = torch.nn.functional.interpolate(
    out.unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False,
).squeeze().cpu().numpy()
print("depth latency ms:", round((time.time() - t0) * 1000, 1))
print("depth shape:", depth.shape, "min/median/max:", depth.min(), np.median(depth), depth.max())

print("\nGPU mem allocated MB:", torch.cuda.memory_allocated() / 1e6 if device == "cuda" else "n/a")
print("OK")
