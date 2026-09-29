# Perception backend (local REST API)

Real segmentation + real metric depth for the [WHOAMI workbench UI](../ui/README.md), run
locally and called over REST instead of the UI's JS mock.

## What this actually is (read before assuming it matches dev.md 1:1)

`dev.md`/`architecture.md` name **RUGD SegFormer** and **Depth Anything 3 Metric Large**. Neither
is usable as "connect the API":

- **No pretrained RUGD SegFormer checkpoint exists anywhere.** RUGD is only a dataset; getting a
  real one means fine-tuning `nvidia/mit-b2` on it yourself — that's dev.md's actual 7-hour Dev 1
  task, not something to wire up in an afternoon.
- **Depth Anything 3's own repo caps Python at 3.13** (this machine has 3.14) **and pulls in a
  full multi-view 3D-reconstruction stack** (`pycolmap`, `open3d`, `gsplat`) that has nothing to
  do with single-frame depth — it's the wrong tool even where it does install.

So this server uses the closest thing that's real and actually runs:

| Task | What dev.md names | What's actually running |
|---|---|---|
| Segmentation | RUGD SegFormer (outdoor-specific) | `nvidia/segformer-b2-finetuned-ade-512-512` (generic 150-class scene segmentation), remapped into the canonical 3-class port via `ontology/ade20k_remap.yaml` — the same "adapter + remap" pattern architecture.md §8.2 describes, just with a stand-in adapter |
| Depth | Depth Anything 3 Metric Large | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` — the direct predecessor, real metric depth, specifically the outdoor variant |

Both give real per-pixel CV output today. Neither is RUGD-specific or the newest release. Training
a real RUGD SegFormer is a separate, much larger task.

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
```

Pick the CUDA index that matches your driver (`nvidia-smi`); `cu126` is what this machine used
with driver CUDA 13.3 — newer drivers are backward compatible with older CUDA runtime builds. A
plain `pip install torch` on Windows gives you a CPU-only build.

## Run

```
.venv\Scripts\uvicorn app:app --host 127.0.0.1 --port 8008
```

First request after startup downloads both models from Hugging Face (a few GB total, cached
under `~/.cache/huggingface` after that). The UI (`ui/`) polls `http://127.0.0.1:8008/health`
every 5 seconds and automatically switches between "REAL ANALYSIS" and "MOCK ANALYSIS" — no
manual toggle, no restart needed on either side if you start/stop this server while the UI is
open.

## Performance reality (measured on an RTX 3050 Laptop, 4GB VRAM)

- Segmentation: ~300-700ms. Depth: ~1.2-1.8s. **Total ~1.5-2.5s per frame.**
- `architecture.md`'s `perception_max_age` is 500ms. This backend cannot meet that on this
  hardware, full stop — the UI will honestly show `STALE` for **live** sources (camera live-detect,
  ROS 2) once connected to this backend, because it's telling the truth about the latency budget.
  A single upload or "take photo" still isn't judged against that live-stream budget (that
  distinction is what `ui/src/analysis/perception.ts` enforces).
- Faster paths exist if this matters later: `segformer-b0` instead of `b2`, a smaller depth model,
  batching, or actually running the RUGD/DA3 models mentioned above on better hardware — none of
  that is done here.

## Files

- `app.py` — the FastAPI app (`/health`, `POST /analyze`).
- `models.py` — loads both models once at startup, runs inference, applies the remap.
- `ontology/ade20k_remap.yaml` — the mandatory adapter remap (architecture.md §8.2): every one of
  the segmentation model's 150 raw labels maps to exactly one of `{0 unknown, 1 traversable,
  2 hazard}`. Read the comment at the top of the file for the rule used.
- `scripts/` — one-off scripts used while building this (inspecting labels, a smoke test). Not
  used by the running server.
