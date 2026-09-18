# Hardware and runtime (Dev 1)

**This file owns only where tensors run** (device, engine, VRAM).  
**It does not own the product architecture.** [`architecture.md`](../architecture.md) still owns the Perception Port, adapters-vs-brain, safety, and `/cmd_vel`. T12 is the only code switch. T07 and the port never import a vendor runtime.

If this file and `architecture.md` disagree on the port or topics, `architecture.md` wins. If they disagree on GPU/engine, this file wins.

## Two different Intel stacks (do not mix)

`OpenVINO` and `XPU` are **not** the same layer. Do not write “OpenVINO + XPU” as one required stack.

```
Production on this Arc (v1):

  YOLOE weights → OpenVINO 2026.4.0 → device="GPU" → Arc B580

Separate optional path (not the product runtime):

  YOLOE / PyTorch → PyTorch XPU → Arc B580
```

| Path | What it is | v1 product? |
|---|---|---|
| **OpenVINO GPU** | OpenVINO runtime, Intel GPU plugin, `device="GPU"` | **Yes — this desktop** |
| **PyTorch XPU** | Intel PyTorch `xpu` device | No. Export/debug only if we use it. Never required *with* OpenVINO. |

Someone implementing T12 must pick **one** backend class, not both in one process as the product path.

## Pin (this project)

- **OpenVINO 2026.4.0** (current stable as of 2026-09-16: `pip install openvino==2026.4.0`).
- Compile / load IR for **GPU**. Do not default to CPU to “make it run.”
- Do not require `intel-extension-for-pytorch` for the product OpenVINO path.

## Machines

| When | GPU | Runtime | Goal |
|---|---|---|---|
| **Now** (this desktop) | Intel Arc **B580** | **OpenVINO 2026.4.0, `device=GPU`** | Efficient Arc inference |
| **Later** (after optimize) | NVIDIA, **less VRAM** than B580 | **CUDA + PyTorch** | Must still fit; do not design to B580 headroom |

```
if Intel Arc (this box):  OpenVINO GPU   (backend id: openvino_gpu)
else NVIDIA:              CUDA PyTorch   (backend id: cuda_pytorch)
```

Ultralytics is **not** the product runtime. It may export ONNX/IR **into** OpenVINO. Do not run Ultralytics/CUDA on this box as `live_cam`.

## VRAM budget (design for the later NVIDIA)

B580 is the development card, not the floor. Assume the NVIDIA has **less** VRAM:

- One frame in flight (latest-only queue; T11).
- YOLOE and Depth Anything **sequential on the same frame** if both run.
- Prefer a **small** YOLOE-seg (s/m, not l) unless outdoor quality forces otherwise.
- No extra GPU copies; no keeping RGB + mask + depth + two models resident if it blows the small card.
- INT8 / extra compression is later, not a dummy-mask shortcut.

## Outdoor / camera (now)

- Work is on a **desktop**. It is **night**. Outdoor live training and outdoor live testing are **later, not now**.
- **No camera on this machine.** T02 cannot be product-tested.
- **No training** (already v1 policy). Night + no camera does not change that.
- **No dummy camera** to unblock T02.

**T02 decode kernel** (Image+CameraInfo → `ImageFrame`) can be built **without** a device — Dev 1 consumes data, Dev 5 owns the driver. Live subscribe still waits on Dev 5 topics. T06 GPU `infer` / T07 ROS / T11 stay blocked on a real stream; T10 still uses header/label fixtures.
