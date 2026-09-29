"""Local REST backend for the WHOAMI perception workbench UI.

Replaces the UI's JS mock (ui/src/analysis/mock.ts) with real model inference: a generic
segmentation model remapped into the canonical 3-class Perception Port, and a real metric depth
model. Read-only: this process never touches ROS 2, never publishes anything, and has no
connection to /cmd_vel or any safety-relevant topic (architecture.md §3.1 is unaffected because
this code has no path into it at all).

Run: .venv\\Scripts\\uvicorn app:app --host 127.0.0.1 --port 8008
"""
import base64
import io
import time

import models
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

OUT_W, OUT_H = 160, 120  # must match ui/src/types.ts GW, GH

app = FastAPI(title="whoami-perception-backend")

# Local dev tool only, never exposed beyond localhost: permissive CORS so the Vite dev server
# (whichever port it lands on) can always reach it.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    models.load()


@app.get("/health")
def health():
    return {"status": "ok" if models.ready() else "loading", "device": models.DEVICE,
             "seg_model": models.SEG_MODEL_ID, "depth_model": models.DEPTH_MODEL_ID}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    t0 = time.time()
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    mask, seg_ms = models.run_segmentation(image, OUT_W, OUT_H)
    depth, depth_ms = models.run_depth(image, OUT_W, OUT_H)

    return {
        "width": OUT_W, "height": OUT_H,
        "mask_b64": base64.b64encode(mask.tobytes()).decode("ascii"),
        "depth_b64": base64.b64encode(depth.tobytes()).decode("ascii"),
        "seg_model": models.SEG_MODEL_ID, "depth_model": models.DEPTH_MODEL_ID,
        "latency_ms": {"seg": round(seg_ms, 1), "depth": round(depth_ms, 1),
                        "total": round((time.time() - t0) * 1000, 1)},
    }
