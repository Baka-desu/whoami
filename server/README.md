# Perception backend (local REST API)

Real segmentation + real metric depth for the [WHOAMI workbench UI](../ui/README.md), run
locally and called over REST instead of the UI's JS mock.

## Segmentation: Dev 1's real pipeline, not a reimplementation

This does **not** duplicate Dev 1's Perception Port logic. `turing/` (Dev 1's `ugv_perception`
package) is installed here as a library (`pip install -e ../turing`) and driven directly:

- **The real RUGD SegFormer checkpoint** — `JasonTStanley/RUGD-Segformer` on Hugging Face,
  fine-tuned from `nvidia/segformer-b5-finetuned-ade-640-640` on the actual RUGD classes. `dev.md`
  says no such checkpoint exists; that's wrong (or at least the first search for one was
  incomplete) — Dev 1 already found it and wired the adapter/remap/config around it
  (`turing/scripts/fetch_rugd_segformer.sh`, `turing/config/adapters/rugd.yaml`,
  `turing/src/ugv_perception/adapter/rugd.py`).
- **Dev 1's real `compose_tick()`** (`turing/src/ugv_perception/compose/tick.py`) does the actual
  remap (`turing/config/ontologies/rugd.yaml`), confidence gating
  (`turing/config/perception/rugd.yaml`), and freshness evaluation
  (`turing/config/perception/port.yaml`, `perception_max_age: 0.50` — same 500ms budget the UI
  already used). This file supplies exactly one missing piece: a CUDA PyTorch inference backend.
  Dev 1's own backend seam for that (`turing/src/ugv_perception/backend/cuda_pytorch.py`) is an
  intentional stub — their dev box is an Intel Arc GPU, and NVIDIA/CUDA was explicitly marked
  "later" (`turing/README.md`: *"Later NVIDIA, less VRAM (CUDA + PyTorch)"*), with a test
  (`test_b13_cuda_pytorch_factory_raises_on_this_box`) asserting the current stub behaviour. That
  file and `backend/factory.py` are left untouched — they're Dev 1's own, already-tested code.
  Instead, `models.py` here builds a `RugdSegformerAdapter` (imported from `turing/`) directly with
  its own small `_TorchRugdBackend`, without going through `factory.build_backend()` at all.
- **Freshness, adapted for a REST call instead of a ROS tick loop.** `compose_tick`'s own
  before/after check (`now_ns_after`) is built for a continuous ROS loop, where a backed-up queue
  makes the *next* frame's `now_ns` itself late — a single synchronous REST call can't inject a
  real post-inference timestamp before calling a function that hasn't run yet. Instead, this file
  applies architecture.md's exact rule (age > `perception_max_age` → degraded) against the call's
  own real measured latency, for a live stream only. A single uploaded/captured still isn't part of
  the live product's freshness contract at all, so latency alone never degrades it — see the
  docstring on `run_segmentation()` in `models.py`.

## Depth: not the literal model named, and here's why

`dev.md`/`architecture.md` name **Depth Anything 3 Metric Large**. Dev 1's own `turing/` targets
the same model (`turing/scripts/fetch_da3metric_large.sh`, `turing/scripts/export_da3metric_openvino.py`)
but it isn't wired live either (`turing/README.md`: *"T08 ... needs weights"*, not marked shipped).
Its own repo (`ByteDance-Seed/depth-anything-3`) caps Python at 3.13 (this machine runs 3.14) and
pulls in a full multi-view 3D-reconstruction stack (`pycolmap`, `open3d`, `gsplat`) unrelated to
single-frame depth. This server uses `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`
instead — the direct predecessor, integrates via plain `transformers`, real metric depth,
specifically the outdoor variant. Depth is an optional geometry side-channel per architecture.md
§9, not part of the mandatory Perception Port, so this substitution doesn't affect port conformance.

## What this is not

Read-only and has nothing to do with ROS 2. It never publishes anything, never touches
`/cmd_vel`, and has no path into the safety authority (architecture.md §3.1). It exists purely so
the UI's upload/camera "still" and "live" sources can show a real analysis instead of the mock.

ROS 2 itself is **not connected** — there's no WSL/Linux or ROS 2 install on this machine. The
UI's ROS 2 rosbridge source (`ui/src/source/rosbridge.ts`) is unchanged and still untested against
a live rosbridge server.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e ../turing

bash ../turing/scripts/fetch_rugd_segformer.sh
# or, without bash: python -c "from huggingface_hub import snapshot_download as s; \
#   s(repo_id='JasonTStanley/RUGD-Segformer', local_dir='../turing/weights/rugd-segformer')"
```

Pick the CUDA index that matches your driver (`nvidia-smi`); `cu126` is what this machine used
with driver CUDA 13.3 — newer drivers are backward compatible with older CUDA runtime builds. A
plain `pip install torch` on Windows gives you a CPU-only build. The RUGD checkpoint is ~330MB and
is never committed (`turing/weights/rugd-segformer/` is gitignored, same as every other adapter's
weights).

## Run

```
.venv\Scripts\uvicorn app:app --host 127.0.0.1 --port 8008
```

`/analyze` is a plain (non-async) route so FastAPI runs it in a worker thread: the segmentation +
depth models block for seconds, and running that inside an `async def` handler stalls the whole
event loop, which made concurrent `/health` polls from the UI time out mid-analysis and falsely
report the backend offline. Keep new routes that do real inference as sync `def` for the same
reason.

The UI (`ui/`) polls `http://127.0.0.1:8008/health` every 5 seconds and automatically switches
between "REAL ANALYSIS" and "MOCK ANALYSIS" — no manual toggle, no restart needed on either side.

## Performance reality (measured on an RTX 3050 Laptop, 4GB VRAM)

- Segmentation (SegFormer-B5, real): ~1.7-3s. Depth: ~0.6-1.3s. **Total ~2.3-4.3s per frame.**
- This backend cannot meet the 500ms live budget on this hardware, full stop. For a **live**
  source (camera live-detect, ROS 2), the UI honestly shows `STALE` / `perception_degraded: true`
  — that's the real Perception Port telling the truth about real latency, not a bug. A single
  upload or "take photo" still isn't judged against that live-stream budget.
- GPU memory: ~1.5-2GB with both models loaded, well within the 4GB budget — latency, not memory,
  is what's blocking the live budget here.

## Files

- `app.py` — the FastAPI app (`/health`, `POST /analyze`).
- `models.py` — loads the real RUGD adapter (via Dev 1's `ugv_perception`) and the depth model
  once at startup, runs inference each request.
- `scripts/smoke_test_rugd.py` — one-off script used while building this: runs Dev 1's real
  `compose_tick` pipeline end-to-end outside the server, to confirm it works without ROS before
  wiring it into `app.py`. Not used by the running server.
