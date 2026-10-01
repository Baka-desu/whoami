# User manual — models and download

Git does **not** ship weights. Clone the repo, then download the checkpoints below. Do not commit `.xml`, `.bin`, `.pt`, or `model.safetensors`.

Live outdoor stack (`architecture.md` §6):

| Role | Model | Link | License |
|---|---|---|---|
| Live mask (25 RUGD classes → `{0,1,2}`) | **SegFormer-B5** `JasonTStanley/RUGD-Segformer` | https://huggingface.co/JasonTStanley/RUGD-Segformer | see the card |
| Optional metric depth | **Depth Anything 3 Metric Large** `depth-anything/DA3METRIC-LARGE` | https://huggingface.co/depth-anything/DA3METRIC-LARGE | Apache 2.0 |
| Selectable, **not** live | YOLOE-26s-seg | Ultralytics `yoloe-26s-seg` | see Ultralytics |
| Eval only | Tutorial ONNX | local IR if you already have it | not a HuggingFace pin |

The node picks an engine at startup: **Intel OpenVINO GPU → NVIDIA CUDA → OpenVINO CPU**. Fine-tune, if ever needed, stays on the Arc B580.

All commands from `turing/` with the project venv.

```bash
cd turing
# create/use turing/.venv as you already do
source .venv/bin/activate   # or: .venv/bin/python ...
```

---

## 1. Download the live models (Intel and NVIDIA)

Same HuggingFace files for both.

```bash
cd turing
bash scripts/fetch_rugd_segformer.sh
bash scripts/fetch_da3metric_large.sh
```

That writes:

```
turing/weights/rugd-segformer/model.safetensors
turing/weights/da3metric-large/model.safetensors
```

Manual equivalent:

```bash
python -m pip install 'huggingface_hub>=0.24'
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("JasonTStanley/RUGD-Segformer", local_dir="turing/weights/rugd-segformer")
snapshot_download("depth-anything/DA3METRIC-LARGE", local_dir="turing/weights/da3metric-large")
print("ok")
PY
```

---

## 2. Intel (Arc GPU or CPU-only)

OpenVINO reads **IR** (`*.xml` + `*.bin`), not safetensors. Export once after download.

Needs: OpenVINO 2026.4.0, `transformers`, `depth-anything-3`, `safetensors`, `torch` (export only).

```bash
cd turing
.venv/bin/python scripts/export_rugd_segformer_openvino.py
.venv/bin/python scripts/export_da3metric_openvino.py --height 336 --width 504
```

`--height 336 --width 504` matches the landscape IR this product uses (sides multiples of 14, long side 504). A different camera aspect is another export of the **same** checkpoint, same script, different `--height` / `--width`.

After export:

```
turing/weights/rugd-segformer.xml
turing/weights/rugd-segformer.bin
turing/weights/da3metric-large.xml
turing/weights/da3metric-large.bin
```

If OpenVINO reports `GPU` / `GPU.0` (Arc), the node compiles there. If there is no Intel GPU, the same IR compiles on **CPU**. No extra device flag.

```bash
pip install openvino==2026.4.0
```

---

## 3. NVIDIA (CUDA)

Skip the OpenVINO export. The live node loads the **safetensors folders** on CUDA.

Needs: NVIDIA driver, **PyTorch with CUDA**, `transformers`, `depth-anything-3`, `safetensors`. No OpenVINO.

```
turing/weights/rugd-segformer/model.safetensors
turing/weights/da3metric-large/model.safetensors
```

If Intel GPU is missing and `torch.cuda.is_available()` is true, the picker uses CUDA. An `.xml` does not run on CUDA; safetensors do not load in OpenVINO.

YOLOE-on-CUDA is **not** wired. Live path is RUGD + DA3.

---

## 4. Mixed boxes

| Machine | Use these files | Engine |
|---|---|---|
| Intel Arc GPU | `*.xml` + `*.bin` | OpenVINO `GPU` |
| NVIDIA GPU, no Intel GPU | safetensors folders | CUDA PyTorch |
| Intel CPU only | `*.xml` + `*.bin` | OpenVINO CPU |
| NVIDIA GPU + Intel CPU, safetensors present | safetensors | CUDA |
| NVIDIA GPU + Intel CPU, IR only | `*.xml` + `*.bin` | OpenVINO (CPU, or NVIDIA plugin if present). CUDA still needs the HuggingFace folders. |
| Intel GPU + NVIDIA GPU | `*.xml` + `*.bin` | Intel OpenVINO GPU |

---

## 5. Optional YOLOE (not live)

Selectable with `adapter:=yoloe`. Default live adapter stays `rugd`.

Intel: put `yoloe-26s-seg.pt` in `turing/weights/`, export to OpenVINO IR `turing/weights/yoloe-26s-seg.xml` (Ultralytics `yolo export ... format=openvino`). The factory GPU path loads that IR.

NVIDIA: YOLOE CUDA is unwired. Do not expect `adapter:=yoloe` on CUDA yet.

---

## 6. What git will never contain

`turing/.gitignore` drops `weights/*.xml`, `weights/*.bin`, `weights/*.pt`, `weights/*.safetensors`, and the HuggingFace folders. After download/export, `git status` stays clean. That is expected.

Tutorial ONNX (`onnx-tutorial.xml`) is eval-only (T09). There is no fetch script for it.
