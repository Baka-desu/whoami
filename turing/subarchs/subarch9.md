# Sub-architecture 9 — Tutorial ONNX eval scaffold

**Task:** [T09](../tasks/T09-onnx-eval-scaffold.md)  
**Depends on (code):** shipped T01–T07, T10, T12 tensor backend; same `infer() → RawSemOutput` seam as T06 / RUGD.  
**Does not change:** live default (`rugd`), `{0,1,2}` topics, τ in `rugd.yaml` / `yoloe.yaml`, T08 depth, `/cmd_vel`.  
**Status:** **implementation started.** No tutorial IR on disk; GPU path refuses `adapter:=onnx` rather than inventing weights. Decode + YAML + default isolation are in. Live default stays RUGD.  
**Authority:** [`architecture.md`](../../architecture.md) §4 (tutorial = eval), §6 adapter scaffold, §8.2 remap YAML, kill list tutorial-as-only-ontology; [`CONFLICTS.md`](../CONFLICTS.md) T09 is not `live_cam`.  
**Not authority:** `dev.md` hours; making ONNX work outdoors; replacing RUGD or YOLOE; inventing a toy ONNX if the tutorial artifact is gone.

This file wins over the T09 checklist. `architecture.md` wins over this file.

If the tutorial ONNX file is missing or useless, **drop T09** in this file rather than invent weights. The adapter seam is already proven by RUGD + YOLOE. T09 exists to prove **swap by config**: `adapter:=onnx` plus its own remap and gate YAML, same port topics.

---

## Pin

| | |
|---|---|
| Role | Eval / bring-up adapter. Not the outdoor product. |
| Launch | `adapter:=onnx` on the existing T07 node |
| Default | `adapter:=rugd`. `live_cam` / `main()` stay RUGD |
| Also legal | `adapter:=yoloe` (selectable, not live) |
| Ontology | `config/ontologies/onnx.yaml` — names from the pinned artifact only. Architecture sidewalk/grass/background is a template |
| Gates | `config/perception/onnx.yaml` — **own τ**. Do not copy `rugd.yaml` or `yoloe.yaml` as if calibrated |
| Weights | Pin at implement time from the tutorial artifact on disk. Not in git. Not a download invented here |
| Runtime | OpenVINO `device=GPU`, no CPU compile. Dense **logits** decoded in the adapter to `[0,1]` scores. Not YOLO-seg / NMS |
| Frames | Real `Image`+`CameraInfo`. No Cityscapes dummy RGB |

Eval ontology is not outdoor truth. `sidewalk→1` is a scaffold mapping. Do not treat it as a trail.

---

## Files

### Authority (read; do not fork)

| File | Why T09 cares |
|---|---|
| [`architecture.md`](../../architecture.md) §8.2 | ONNX sidewalk\|grass\|background → `{0,1,2}` |
| [`subarch3.md`](subarch3.md) | Python `int` keys on `id_to_name`; unmapped name → 0 |
| [`subarch4.md`](subarch4.md) | Scores already in `[0,1]` before τ |
| [`subarch6.md`](subarch6.md) | Same `RawSemOutput` / raise-on-fail contract |
| [`subarch7.md`](subarch7.md) | `adapter:=onnx` was illegal until this task |
| [`subarch12.md`](subarch12.md) | GPU only; T07 does not import OpenVINO |

### Existing code T09 must not rewrite

| File | Binding |
|---|---|
| `turing/src/ugv_perception/compose/tick.py` | Same tick. No `if adapter_id == "onnx"` |
| `turing/src/ugv_perception/remap/apply.py` | T07 remaps. T09 does not |
| `turing/src/ugv_perception/node/wire.py` | Topic names stay `/segmentation/mask`, confidence, port_meta, degraded |
| `turing/config/perception/rugd.yaml` | Product τ. T09 does not edit it |
| `turing/src/ugv_perception/backend/rugd_live.py` | `main()` still builds RUGD |

### Code later (only after this subarch is frozen)

| File | Role |
|---|---|
| `turing/config/ontologies/onnx.yaml` | `adapter_id: onnx`. sidewalk / grass / background → `{0,1,2}` |
| `turing/config/perception/onnx.yaml` | Own τ, `adapter_id: onnx` |
| `turing/config/adapters/onnx.yaml` | Local IR path, `backend: openvino_gpu` |
| `turing/src/ugv_perception/adapter/onnx_tutorial.py` | `OnnxTutorialAdapter.infer` → `RawSemOutput` |
| `turing/src/ugv_perception/tests/test_onnx_scaffold.py` | Swap, missing YAML, default still rugd |

Do **not** add `DummySource`.  
Do **not** publish `/cmd_vel`.  
Do **not** train.  
Do **not** make `main()` construct this adapter.  
Do **not** share τ with RUGD or YOLOE.

---

## When coding (after this file is frozen)

1. Freeze this file. If the tutorial IR is not on disk, write **dropped** in Status and stop. Do not invent an ONNX.
2. Add `onnx` to the node’s allowed `adapter` names. Default stays `rugd`. Constructing the default node **must not import or compile** the ONNX adapter. Load ONNX only when `adapter:=onnx`.
3. `adapter:=onnx` loads `ontologies/onnx.yaml` and `perception/onnx.yaml` the same way `rugd` and `yoloe` load theirs.
4. Inspect the pinned tutorial ONNX. Record input shape, layout, RGB vs BGR, mean/std, output tensor shape, class axis, and native resolution in `config/adapters/onnx.yaml`. Those values are constants. **No runtime guessing.**
5. `infer()` always returns `RawSemOutput` at the camera frame `H×W`. Decode: resize logits to that `H×W`, softmax over the class axis, `label_ids = argmax`, `raw_scores = max probability` in `[0,1]`. T03 and T04 never see raw logits.
6. Adapter raises on backend failure. T07 sets `adapter_error`. No all-traversable mask.
7. Names may live in the tutorial’s labels file, not inside the ONNX. Source of truth: model class id → adapter-owned `id_to_name` (Python `int` keys) → `onnx.yaml` remap. YAML may only list names that artifact actually provides. **No invented names**, including no fake hazard.
8. Tests use a scripted `RawSemOutput` or a real IR on a real still. No generated Cityscapes frames.
9. Under `adapter:=rugd` and `adapter:=onnx`, topic names and the `{0,1,2}` encoding are the same. Pixel **distributions** need not match. All-0 or all-1 is a legal ONNX eval mask.
10. `adapter:=onnx` with a missing IR or missing `onnx.yaml` **fails construction**. It does not fall back to RUGD. Default `adapter:=rugd` starts without touching ONNX.

---

## 1. What T09 is

```
adapter:=rugd   → RUGD SegFormer   → same port topics
adapter:=yoloe  → YOLOE            → same port topics
adapter:=onnx   → tutorial ONNX    → same port topics   (eval only)

main() / live_cam  → rugd
```

T09 is a third **selectable adapter implementation**. It is not a second node. It is not T08. Depth stays a side-channel.

Architecture DoD: swap is remap YAML + confidence profile + the `adapter` parameter. No new publishers.

---

## 2. Ontology (eval)

The IR’s class list is authoritative.

```
model class id  (from the IR)
        ↓
adapter-owned id_to_name  (Python int keys; names from the tutorial labels/config)
        ↓
onnx.yaml  (only those names)
        ↓
{0,1,2}
```

No YAML row may invent an IR class. Architecture’s sidewalk example is a **template** until the artifact is inspected:

| Name (if present on the IR) | Port |
|---|---|
| `background` | 0 |
| `sidewalk` | 1 |
| `grass` | 1 |
| default / unmapped | 0 |

Class 2 is used only if the IR actually has a hazard name. An all-0 or all-1 mask is a legal eval result. T09 does not manufacture a hazard class to fill the port.

---

## 3. Runtime

```
Tutorial ONNX  (signature recorded at implement time, not guessed)
   ↓
OpenVINO GPU raw tensor
   ↓
OnnxTutorialAdapter preprocess + decode
   ├─ resize logits to camera H×W
   ├─ softmax over class axis
   ├─ argmax → label_ids int32 H×W
   └─ max probability → raw_scores float32 H×W in [0,1]
   ↓
RawSemOutput
   - original stamp_ns and optical frame_id
   - id_to_name: exact Python-int LUT
   ↓
T03 onnx.yaml → T04 onnx.yaml → T05 → T07 → T10
   ↓
same port topics, pixels in {0,1,2}
```

No YOLO decode, no NMS, no `if adapter_id` in `compose_tick`.

If the IR input size is static, compile once at that size. Do not pad a trail still into a Cityscapes square to “make it work outdoors.” T09 is not outdoor competence.

---

## 4. Tests

- `adapter:=onnx` without `onnx.yaml` or without the pinned IR **fails**. No silent fallback to RUGD.
- Default `adapter:=rugd` / `main()` starts without the ONNX IR, without importing `onnx_tutorial`, and without compiling that IR.
- Topic names and canonical encoding are the same for `rugd` and `onnx`. Every published pixel is in `{0,1,2}`. Class histograms may differ; a scaffold with no hazard class may publish only 0 and 1.
- Scripted logits such as `[-4.2, 0.7, 3.8]` become scores in `[0,1]` after softmax; T04 sees those scores, not the logits.

---

## 5. Drop rule

If there is no tutorial ONNX on disk when this subarch is frozen, Status becomes **dropped**. The seam is already proven. Do not train a stand-in. Do not convert a random Cityscapes checkpoint “to have T09.”
