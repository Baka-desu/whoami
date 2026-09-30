# Perception backend (local REST API)

Real segmentation + real metric depth for the [WHOAMI workbench UI](../ui/README.md), run
locally and called over REST instead of the UI's JS mock. Both models are the ones `dev.md` and
`architecture.md` actually name, run through Dev 1's real, tested pipeline (`turing/`) — nothing
here is a reimplementation or a stand-in model anymore.

## Segmentation: Dev 1's real pipeline

`turing/` (Dev 1's `ugv_perception` package) is installed here as a library
(`pip install -e ../turing`) and driven directly, via Dev 1's own top-level factory:

- **The real RUGD SegFormer checkpoint** — `JasonTStanley/RUGD-Segformer` on Hugging Face,
  fine-tuned from `nvidia/segformer-b5-finetuned-ade-640-640` on the actual RUGD classes.
- **`build_live_adapter()`** (`turing/src/ugv_perception/backend/rugd_live.py`) picks Intel
  OpenVINO GPU, then NVIDIA CUDA safetensors, then OpenVINO CPU
  (`turing/src/ugv_perception/backend/device.py`) — Dev 1's own code, which now includes the CUDA
  backend this server initially had to stand in a version of itself (their dev box is Intel Arc;
  NVIDIA/CUDA was marked "later" until it shipped upstream). `models.py` here just calls
  `build_live_adapter(TURING)` — no backend code of its own anymore.
- **Dev 1's real `compose_tick()`** (`turing/src/ugv_perception/compose/tick.py`) does the actual
  remap (`turing/config/ontologies/rugd.yaml`), confidence gating
  (`turing/config/perception/rugd.yaml`), and freshness evaluation
  (`turing/config/perception/port.yaml`, `perception_max_age: 0.50` — same 500ms budget the UI
  already used).
- **Freshness, adapted for a REST call instead of a ROS tick loop.** `compose_tick`'s own
  before/after check (`now_ns_after`) is built for a continuous ROS loop, where a backed-up queue
  makes the *next* frame's `now_ns` itself late — a single synchronous REST call can't inject a
  real post-inference timestamp before calling a function that hasn't run yet. Instead, this file
  applies architecture.md's exact rule (age > `perception_max_age` → degraded) against the call's
  own real measured latency, for a live stream only. A single uploaded/captured still isn't part of
  the live product's freshness contract at all, so latency alone never degrades it — see the
  docstring on `run_segmentation()` in `models.py`.

## Depth: also Dev 1's real pipeline

`dev.md`/`architecture.md` name **Depth Anything 3 Metric Large**, and that's what runs here:
`depth-anything/DA3METRIC-LARGE`, through Dev 1's own `DepthChannel`
(`turing/src/ugv_perception/backend/depth_live.py`), same `device.py` backend picker as
segmentation.

Getting there took ruling out the obvious path first: `ByteDance-Seed/depth-anything-3`'s own
`pip install -e .` caps Python at 3.13 (this machine runs 3.14) and declares a full multi-view
3D-reconstruction stack (`pycolmap`, `open3d`, `gsplat`, `evo`, `moviepy`) that has nothing to do
with single-frame depth. But building just the metric model class
(`depth_anything_3.cfg.create_object(load_config(MODEL_REGISTRY["da3metric-large"]))`) only
actually imports three external packages transitively — `omegaconf`, `addict`, `einops` — none of
the heavy ones. See **Setup** below for the exact install step
(`pip install --no-deps --ignore-requires-python`).

## What this is not

Read-only and has nothing to do with ROS 2. It never publishes anything, never touches
`/cmd_vel`, and has no path into the safety authority (architecture.md §3.1). It exists purely so
the UI's upload/camera "still" and "live" sources can show a real analysis instead of the mock.

ROS 2 itself is **not connected** — there's no WSL/Linux or ROS 2 install on this machine. The
UI's ROS 2 rosbridge source (`ui/src/source/rosbridge.ts`) is unchanged and still untested against
a live rosbridge server.

## Setup

Built and run on this machine with **Python 3.14** on Windows. `depth-anything-3`'s own package
declares `requires-python <=3.13`, but nothing this server actually imports from it needs anything
3.14 removed (see the `--ignore-requires-python` note below) — 3.14 is not a requirement, just what
was tested; 3.10-3.13 should work unmodified since they satisfy that cap without the flag.

Windows:
```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e ../turing

bash ../turing/scripts/fetch_rugd_segformer.sh
bash ../turing/scripts/fetch_da3metric_large.sh

git clone https://github.com/ByteDance-Seed/depth-anything-3 <somewhere>
.venv\Scripts\pip install --no-deps --ignore-requires-python -e <somewhere>
```

Linux / WSL (the eventual ROS 2 Lyrical host — untested here, no Linux box on this machine):
```
python -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../turing

bash ../turing/scripts/fetch_rugd_segformer.sh
bash ../turing/scripts/fetch_da3metric_large.sh

git clone https://github.com/ByteDance-Seed/depth-anything-3 <somewhere>
.venv/bin/pip install --no-deps --ignore-requires-python -e <somewhere>
```

Pick the CUDA index that matches your driver (`nvidia-smi`); `cu126` is what this machine used
with driver CUDA 13.3 — newer drivers are backward compatible with older CUDA runtime builds. A
plain `pip install torch` gives you a CPU-only build on either platform.

`--ignore-requires-python` skips depth-anything-3's 3.13 cap (nothing actually used here needs
anything Python 3.14 removed); `--no-deps` skips its heavy requirements.txt, since
`requirements.txt` here already installs the three packages (`omegaconf`, `addict`, `einops`) that
building the metric model class actually needs.

Neither checkpoint is ever committed — `turing/weights/rugd-segformer/` and
`turing/weights/da3metric-large/` are both gitignored, same as every other adapter's weights.

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

- Segmentation (SegFormer-B5, real): ~1.7-3s. Depth (DA3-Metric-Large, real): ~1.9-2.6s. **Total
  ~3.5-5s per frame.**
- This backend cannot meet the 500ms live budget on this hardware, full stop. For a **live**
  source (camera live-detect, ROS 2), the UI honestly shows `STALE` / `perception_degraded: true`
  — that's the real Perception Port telling the truth about real latency, not a bug. A single
  upload or "take photo" still isn't judged against that live-stream budget.
- GPU memory: comfortably within the 4GB budget with both models loaded — latency, not memory, is
  what's blocking the live budget here.

## Files

- `app.py` — the FastAPI app (`/health`, `POST /analyze`).
- `models.py` — loads Dev 1's real RUGD adapter and DepthChannel once at startup (via
  `ugv_perception.backend.rugd_live`/`depth_live`), runs inference each request.
- `scripts/smoke_test_rugd.py`, `scripts/smoke_test_da3.py` — one-off scripts used while building
  this: run Dev 1's real pipelines end-to-end outside the server, to confirm each works before
  wiring it into `app.py`. Not used by the running server.
