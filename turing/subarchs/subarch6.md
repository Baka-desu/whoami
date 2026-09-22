# Sub-architecture 6 — YOLOE outdoor adapter (T06)

**Task:** [T06](../tasks/T06-yoloe-adapter.md)  
**Depends on (code/import):** T01 (`stamp_ns` Python `int > 0`); T03 contract (prompt names ⊂ ontology keys; `id_to_name` keys Python `int`); T12 `InferenceBackend` for `infer()`. T06 does **not** import remap apply, gates, freshness, or T07.  
**Does not import / need at build for `pack()`:** T02, T04, T05, T07, camera.  
**Blocked for product `infer()` proof:** a real Image+CameraInfo stream (Dev 5). **YOLOE-26s** OpenVINO IR is on disk (`weights/yoloe-26s-seg.xml`). T02 decode/subscribe **shipped**; no dummy RGB as outdoor.  
**Runtime contract (not a T06→T07 import):** T06 **raises** on failure. T07 catches and sets `adapter_error=True` for T05. T06 does not set `degraded`.  
**Authority:** [`architecture.md`](../../architecture.md) §3, §6 (YOLOE is the outdoor **source**, not the brain), §8.2, §16 (YOLOE-as-brain; adapter publish without remap)  
**Not authority:** `dev.md` hours; Ultralytics as `live_cam` runtime; PyTorch XPU; `interfaces.md` “raw scores not [0,1] yet” — v1 YOLOE + T04 `identity` requires numeric domain `[0,1]` at this boundary  
**Hardware:** [`HARDWARE.md`](../HARDWARE.md) — OpenVINO **2026.4.0**, `device="GPU"`, Arc B580. Engine implementation is **T12**, not this file.

This file is the architecture of T06. The task file is the build checklist. If they disagree, this file wins, then `architecture.md`. Hardware engine: `HARDWARE.md` then T12.

---

## Files this subarch refers to

### Authority (read; do not fork)

| File | Why T06 cares |
|---|---|
| [`architecture.md`](../../architecture.md) §6, §8.2, §16 | YOLOE = adapter; remap YAML mandatory; not the brain |
| [`HARDWARE.md`](../HARDWARE.md) | OpenVINO GPU vs PyTorch XPU; small model; no dummy camera |
| [`subarch1.md`](subarch1.md) | v1 `scale == 1.0`; `stamp_ns` Python `int > 0` |
| [`subarch3.md`](subarch3.md) | R10 Python `int` keys; R12 unmapped name → 0; T06 converts `np.int64` |
| [`subarch4.md`](subarch4.md) | v1 `identity` needs scores already in `[0,1]` (domain, not calibration) |
| [`subarch5.md`](subarch5.md) | T07 supplies `adapter_error` from T06 raise; T06 does not catch itself into a fake mask |
| [`T12-backend-seam.md`](../tasks/T12-backend-seam.md) | `InferenceBackend`; `openvino_gpu` |

### Existing code T06 binds to (do not modify in T06)

| File | Binding |
|---|---|
| `turing/config/ontologies/yoloe.yaml` | Prompt names **must be a subset** of `map` keys |
| `turing/src/ugv_perception/remap/apply.py` | T07 will call remap; T06 does not |
| `turing/src/ugv_perception/confidence/apply.py` | T07 will gate; T06 does not clip for T04 |
| `turing/src/ugv_perception/freshness/evaluate.py` | T07 sets `adapter_error` if T06 raises |
| `turing/src/ugv_perception/port/` | T06 does not call `make_mask` or publish |

### Code later (T06 implementation — only after this subarch is complete)

| File | Role |
|---|---|
| `turing/src/ugv_perception/adapter/frame.py` | `ImageFrame` DTO (rgb, stamp_ns, frame_id). T02 fills this. Not a camera driver |
| `turing/src/ugv_perception/adapter/output.py` | `RawSemOutput` |
| `turing/src/ugv_perception/adapter/prompts.py` | `load_prompts` |
| `turing/src/ugv_perception/adapter/pack.py` | instances → dense `label_ids` / `raw_scores` |
| `turing/src/ugv_perception/adapter/yoloe.py` | `YoloeAdapter` (`infer` = backend.run + pack) |
| `turing/src/ugv_perception/tests/test_yoloe_pack.py` | Y1–Y14 packing / contract tests (no GPU required) |
| `turing/config/perception/yoloe_prompts.yaml` | prompt list |
| `turing/config/adapters/yoloe.yaml` | weights path, `backend: openvino_gpu` |

Do **not** put `backend:` into `config/perception/yoloe.yaml` (that file is T04 gates).  
Do **not** implement `OpenVinoGpuBackend` in T06 — that is T12.  
Do **not** add `DummySource` or a constant-mask backend.

---

## When coding (after this subarch is complete)

1. Freeze this file first (same rule as T01/T03–T05).  
2. Create **only** the files in “Code later”.  
3. Do **not** edit `port/`, `remap/`, `confidence/`, `freshness/`, T07, or `architecture.md`.  
4. **Order:** `pack()` + prompt load first (no GPU). `infer()` uses T12 + **yoloe-26s-seg** IR (on disk). Product proof needs Dev 5 frames.  
5. Inject the backend: `YoloeAdapter(backend: InferenceBackend, ...)`. T06 `pack` does not import `openvino`.  
6. Convert `id_to_name` keys with `int(k)` **here**. Do not ask T03 to coerce.  
7. On backend failure: **raise**. Do not return all-traversable / all-unknown as a “safe” mask.  
8. No training. No download at field runtime. No PyTorch XPU as the product path.  
9. Pin **YOLOE-26s-seg** IR in `config/adapters/yoloe.yaml`, not in Python. 26m only if 26s is weak.  
10. Tests: packing uses **scripted instances** (contract fixtures). That is not a dummy camera. GPU engine `run` is allowed on the IR. Product `infer` on outdoor RGB stays skipped until Dev 5 frames exist.

---

## 1. Role in the product

YOLOE is the default **outdoor adapter**. It speaks model language (prompt ids, instance masks, scores). The brain never sees that. T03 remaps; T04 gates; T07 publishes.

```
ImageFrame (T02 decode / ROS subscribe)
        │
        ▼
┌────────────── T06 ──────────────┐
│  prompts YAML (⊂ ontology keys) │
│  InferenceBackend.run(rgb)      │  T12: OpenVINO GPU on this box
│  pack instances → RawSemOutput  │
│    stamp/frame copied           │
│    id_to_name Python int keys   │
│    scores numeric [0,1] or raise│
│    unlabeled → name for T03 → 0 │
│  raise on failure               │
└──────────────┬──────────────────┘
               │ RawSemOutput  (not the port)
               ▼
        T03 remap → T04 gates → T05 policy → T07 make_mask
```

T06 must not import Nav2, `/cmd_vel`, or port publishers. If a change needs to know OpenVINO vs CUDA, it belongs in **T12**, not in `pack()`.

---

## 2. Scope (tight)

### T06 owns

| Piece | Why |
|---|---|
| `ImageFrame` DTO fields used by infer | stamp/frame/rgb identity; T02 will source them |
| `RawSemOutput` | adapter language; must not leak past T03 |
| Prompt list load + subset of ontology | §8.2 |
| `pack` (instances → dense maps) | overlapping instances; unlabeled |
| `YoloeAdapter.infer` orchestration | backend + pack; copy stamp/frame |
| `int(k)` on `id_to_name` | T03 R10 |

### T06 does not own

| Piece | Owner |
|---|---|
| Camera driver | Dev 5. T02 fills `ImageFrame` from Image+CameraInfo |
| OpenVINO GPU / CUDA `InferenceBackend` | T12 |
| Remap YAML / LUT | T03 |
| τ / identity / collapse | T04 |
| `degraded` / `adapter_error` flag | T05 policy, T07 sets the bool |
| `make_mask`, topics | T01 / T07 |
| Depth Anything | T08 |
| Tutorial ONNX | T09 |
| Training / fine-tune | not v1 |

---

## 3. Types

```python
UNLABELED_NAME = "unlabeled"   # not a prompt; T03 map miss → default 0
UNLABELED_ID = 0               # reserved. No prompt may ever use this id.

@dataclass(frozen=True, slots=True)
class ImageFrame:
    rgb: NDArray[np.uint8]     # H,W,3
    stamp_ns: int              # Python int > 0; source image time
    frame_id: str              # non-empty optical frame

@dataclass(frozen=True, slots=True)
class RawSemOutput:
    adapter_id: str            # "yoloe"
    label_ids: NDArray[np.int32]
    raw_scores: NDArray[np.float32]  # winning Instance.score copied; domain [0,1]
    id_to_name: dict[int, str]       # type(key) is int
    stamp_ns: int
    frame_id: str
    hw: tuple[int, int]

@dataclass(frozen=True, slots=True)
class Instance:
    prompt_id: int             # 1..N matching prompts. Never 0 (unlabeled is not a prompt)
    score: float               # Python float, finite, in [0,1]; copied into dense raw_scores
    mask: NDArray[np.bool_]    # H,W same as rgb

class InferenceBackend:        # T12; T06 depends on the protocol
    id: str
    def load(self, weights_path: str, **engine_args) -> None: ...
    def run(self, rgb: NDArray[np.uint8]) -> tuple[Instance, ...]: ...
```

v1 `hw == rgb.shape[:2]` (`scale == 1.0`). No downscale in T06.

**Prompts** (`config/perception/yoloe_prompts.yaml`):

```yaml
adapter_id: yoloe
prompts:
  - dirt_path
  - gravel
  - grass
  - sky
  - person
  - vehicle
  - rock
  - water
  - fence
  - tree
  - vegetation
```

Every prompt is a non-empty Python `str`. **Subset** of `config/ontologies/yoloe.yaml` `map` keys. `unlabeled` is **not** in this list. Prompt index `i` in this list is id `i+1`. **Id 0 is never assigned to a prompt.**

**Adapter config** (`config/adapters/yoloe.yaml`):

```yaml
adapter_id: yoloe
backend: openvino_gpu
weights: weights/yoloe-26s-seg.xml
```

Ultralytics may be used **offline** to export IR. It is not `live_cam`.

---

## 4. Pack mechanics

Input: `ImageFrame` + sequence of `Instance` (from T12 `run`).  
Output: `RawSemOutput`.

1. Validate frame: `stamp_ns` Python `int > 0`; `frame_id` non-empty `str`; `rgb` uint8 HWC, H>0, W>0, C=3.  
2. Allocate `label_ids` `int32` filled with `UNLABELED_ID`, `raw_scores` `float32` zeros, shape HW.  
3. Reject any instance with `prompt_id == 0` or `prompt_id` not in `1..N`.  
4. Sort instances by **score descending**. For each, where `mask` and `score` **>** the score already on that pixel, write `prompt_id` and **that same** `Instance.score` into `raw_scores`. Highest score wins.  
5. `id_to_name = {0: "unlabeled"}` plus `{int(i): prompt for i, prompt in enumerate(prompts, start=1)}`. Every key `type is int`. No prompt name is stored under `0`.  
6. `adapter_id == "yoloe"`. `stamp_ns`/`frame_id` **copied** from the frame (not `time.now()`).  
7. Every instance score finite and ∈ `[0,1]`. If outside, **raise** (do not clip).  
8. Do not mutate `rgb` or instance masks.

### `raw_scores` semantics

The name `raw_scores` means **pre-T04** (before identity/sigmoid and τ), not “unnormalized logits.”

- Pack **copies** `Instance.score` onto winning pixels. It does not minmax, sigmoid, average, or invent a new statistic.  
- Unlabeled pixels keep `0.0`. That is the fill value, still in `[0,1]`. It is not a calibrated P(class).  
- T04 `identity` consumes this dense array as-is. Domain `[0,1]` ≠ well-calibrated probability (subarch4).  
- v1 pack does **not** emit a runner-up map. T04 κ is skipped unless T07 later passes one (it will not, from T06 v1).

Empty instance list is legal: all unlabeled, scores 0. That is not a fake “all traversable.” T03 → all 0; T04 may collapse. T07/`adapter_error` is only for **raise**, not for empty detections.

---

## 5. Invariants (fail closed)

Every ID has a test (packing tests do not need a camera).

| ID | Rule |
|---|---|
| Y1 | `adapter_id` is `"yoloe"` (`type is str`) |
| Y2 | `stamp_ns` / `frame_id` identity with `ImageFrame`; `type(stamp_ns) is int`; no `time.now()` |
| Y3 | `label_ids` `int32` 2-D HW = rgb HW; `raw_scores` `float32` same HW |
| Y4 | `id_to_name` keys `type is int`; `{np.int64(1): "person"}` never stored. Prompts occupy ids `1..N` only |
| Y5 | Prompt load: names ⊂ ontology `map` keys; unknown prompt → load ERROR |
| Y6 | `unlabeled` is not in the prompt list. **No prompt may ever receive id 0.** `prompt_id == 0` at pack → raise. `id_to_name[0] == "unlabeled"` only |
| Y7 | Overlap: higher score wins per pixel; dense score is that instance’s score copied |
| Y8 | Uncovered pixels stay unlabeled / score `0.0` |
| Y9 | Instance score outside `[0,1]` or non-finite → raise (no clip, no rescale) |
| Y10 | Packed `raw_scores` are those copied scores (or `0.0` unlabeled), still in `[0,1]`. Pack does not transform them |
| Y11 | Backend / pack failure → **raise**; no all-traversable fallback |
| Y12 | `pack` / `yoloe.py` do not import `remap`, `confidence`, `freshness`, `rclpy`, `openvino` (openvino lives in T12) |
| Y13 | `YoloeAdapter` takes `InferenceBackend`; does not branch `if backend.id == ...` inside `pack` |
| Y14 | No ROS publishers in `adapter/` |

---

## 6. `infer`

```
raw_instances = self.backend.run(frame.rgb)   # T12; may raise
return pack(frame, raw_instances, prompts)
```

If `backend.run` raises, `infer` raises (same exception or a wrapping `AdapterError`). T07 sets `adapter_error=True`. T06 does not call T05.

`infer` product test: **skipped until a Dev 5 Image+CameraInfo stream exists**. IR is on disk. Do not unblock with a synthetic RGB scene claimed as outdoor.

---

## 7. Aptness vs `architecture.md`

| Clause | Fit |
|---|---|
| §6 YOLOE default outdoor adapter | **High** |
| §3 adapters are sources, not brain | **High** — no `/cmd_vel`, no port publish |
| §8.2 remap mandatory | **High** — T06 does not remap; prompts ⊂ YAML |
| §16 YOLOE-as-brain | **High** |
| §8.3 identity `[0,1]` | **High** — scores domain at pack; no minmax |
| `HARDWARE.md` OpenVINO GPU | **High** — engine is T12; T06 injects it |

---

## 8. Tests

**Y1–Y14** in `test_yoloe_pack.py` with scripted `Instance`s (no camera, no OpenVINO required).

GPU engine `run` is tested on the IR. Product `infer` on outdoor RGB stays skipped until Dev 5 frames exist. A `FixtureBackend` that returns scripted instances is allowed **only in tests**, never as `live_cam`.

**Implementation note (not a new architecture ID):** before packing, validate each `Instance.mask` as **exactly** `dtype=bool` (`np.bool_`), 2-D, non-empty, and `shape == rgb.shape[:2]`. Implied by the type/`pack` mechanics; test it anyway. Do not treat uint8 0/1 as a mask.

---

## 9. Done when

- `pack` + prompt load pass Y1–Y14.  
- `YoloeAdapter` is backend-injected.  
- Weights path is config, local, not a runtime download.  
- Product `infer()` on IR + real frames is required to mark T06 *task* done. Pack/Y1–Y14 are shipped.

## 10. Non-goals

T02 driver, T12 OpenVINO impl, remap, τ, topics, Depth Anything, training, PyTorch XPU, Ultralytics-as-runtime.
