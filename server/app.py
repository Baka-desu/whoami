"""Local REST backend for the WHOAMI perception workbench UI.

Replaces the UI's JS mock with real model inference: Dev 1's real Perception Port pipeline
(turing/, see models.py's docstring) for both segmentation and depth. Read-only: this process
never touches ROS 2, never publishes anything, and has no connection to /cmd_vel or any
safety-relevant topic (architecture.md §3.1 is unaffected because this code has no path into it
at all).

Run: .venv\\Scripts\\uvicorn app:app --host 127.0.0.1 --port 8008
"""
import io
import base64
import time

import numpy as np
import models
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

OUT_W, OUT_H = 160, 120  # must match ui/src/types.ts GW, GH
SEG_MODEL = "JasonTStanley/RUGD-Segformer (via Dev 1's ugv_perception port)"
DEPTH_MODEL = "depth-anything/DA3METRIC-LARGE (via Dev 1's ugv_perception port)"

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
             "seg_model": SEG_MODEL, "depth_model": DEPTH_MODEL}


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...), streaming: bool = Form(False),
    fx: float = Form(...), fy: float = Form(...), cx: float = Form(...), cy: float = Form(...),
):
    # A plain `def` route: FastAPI runs it in a worker thread instead of the asyncio event loop,
    # so a slow synchronous model call here can't stall concurrent requests (notably /health,
    # which the UI polls every 5s to decide real-vs-mock - an async def route blocking on torch
    # inference for 2-3s made those polls time out mid-analysis and falsely flip to "offline").
    t0 = time.time()
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    mask, confidence, valid, degraded, seg_ms = models.run_segmentation(image, OUT_W, OUT_H, streaming=streaming)
    depth, depth_ms = models.run_depth(image, OUT_W, OUT_H, k)

    return {
        "width": OUT_W, "height": OUT_H,
        "mask_b64": base64.b64encode(mask.tobytes()).decode("ascii"),
        "confidence_b64": base64.b64encode(confidence.tobytes()).decode("ascii"),
        "depth_b64": base64.b64encode(depth.tobytes()).decode("ascii"),
        "valid": valid, "degraded": degraded,
        "seg_model": SEG_MODEL, "depth_model": DEPTH_MODEL,
        "latency_ms": {"seg": round(seg_ms, 1), "depth": round(depth_ms, 1),
                        "total": round((time.time() - t0) * 1000, 1)},
    }
