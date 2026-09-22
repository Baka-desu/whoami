# Sub-architecture 12 — Inference backend seam (T12)

**Task:** [T12](../tasks/T12-backend-seam.md)  
**Authority for the port:** [`architecture.md`](../../architecture.md) §3, §6 — T12 must not change topics or `{0,1,2}`.  
**Authority for the engine:** [`HARDWARE.md`](../HARDWARE.md) — OpenVINO **2026.4.0**, `device="GPU"`, Arc B580. Not PyTorch XPU.  
**Authority for the Python protocol:** **already-shipped T06** — [`yoloe.py` `InferenceBackend`](../src/ugv_perception/adapter/yoloe.py) and [`Instance`](../src/ugv_perception/adapter/output.py). **This file does not fork a second protocol.** The old T12 checklist (`run() → BackendTensors`, `backend:` in T04’s `perception/yoloe.yaml`) is **dead**.

If this file and T06’s implemented `run() -> tuple[Instance, ...]` disagree, **T06 wins**. If this file and `HARDWARE.md` disagree on GPU, **HARDWARE wins**.

---

## Dependency match (read this first)

```
T06 YoloeAdapter.infer(frame)
      backend.run(rgb)  ──────────────►  tuple[Instance, ...]     # T12 MUST emit this
      pack(frame, instances, prompts) ►  RawSemOutput             # T06; T12 must NOT call pack

T07 compose_tick
      adapter.infer(frame)                                        # SpyAdapter in tests
      does NOT import openvino / torch / backend.factory

T04 identity
      Instance.score already in [0,1]                             # T12 copies or raises; no minmax

T03
      prompt_id 1..N; id 0 unlabeled                              # T12 never emits prompt_id 0
```

| Direction | Allowed? | What |
|---|---|---|
| T12 → `adapter.output.Instance` / `AdapterError` | **yes** | emit what `pack()` already validates |
| T12 → `adapter.yoloe.InferenceBackend` | **yes** (implement the Protocol) | do not copy a second Protocol class |
| T12 → `adapter.pack` / `YoloeAdapter` | **no** | T06 owns pack/infer |
| T12 → `compose/`, `remap/`, `confidence/`, `freshness/`, `port/mask` | **no** | would leak engine into the port |
| `compose/` → `backend/` or `openvino` | **no** | N2 already tested |
| T07 `adapter_node.main` → `backend.factory` | **yes, lazy** | factory must not import OpenVINO at module import time |
| Factory → `OpenVinoGpuBackend` | **lazy import inside the branch** | `import openvino` only when `backend: openvino_gpu` |

`YoloeAdapter` **does not call `backend.load()`**. The factory returns a backend that is **already `load()`’d**. No T06 change.

`run(self, rgb)` takes **only rgb** (T06). Prompt list is bound at **`load()`**, not at `run()`. The **same** `prompts` tuple used for `YoloeAdapter(backend, prompts)` must be passed into `load` / factory. If they differ, `prompt_id` is a lie.

T04 gate YAML (`config/perception/yoloe.yaml`) is **not** the backend file. Backend id lives in `config/adapters/yoloe.yaml` (`backend: openvino_gpu`).

---

## Files this subarch refers to

### Authority / bind (do not fork, do not “fix” the protocol)

| File | Binding |
|---|---|
| `turing/src/ugv_perception/adapter/yoloe.py` | `InferenceBackend`: `id`, `load(weights_path, **engine_args)`, `run(rgb) -> tuple[Instance, ...]` |
| `turing/src/ugv_perception/adapter/output.py` | `Instance(prompt_id: int, score: float, mask: bool HW)`; `AdapterError` |
| `turing/src/ugv_perception/adapter/pack.py` | consumers of T12 output; T12 does not call it |
| `turing/src/ugv_perception/compose/tick.py` | must keep zero `openvino` imports |
| `turing/config/adapters/yoloe.yaml` | `backend`, `weights` (local `.xml`) |
| `turing/config/perception/yoloe_prompts.yaml` | `1..N` ids; same tuple as adapter |
| [`HARDWARE.md`](../HARDWARE.md) | `openvino==2026.4.0`, `device="GPU"`, no XPU product path |
| [`subarch6.md`](subarch6.md) | score domain, id 0, mask bool HW, raise don’t fake |
| [`subarch7.md`](subarch7.md) | compose does not construct engines |

### Code later (only after this subarch is complete)

| File | Role |
|---|---|
| `turing/src/ugv_perception/backend/__init__.py` | export factory only; **no** `import openvino` here |
| `turing/src/ugv_perception/backend/factory.py` | `build_backend(backend_id, weights_path, prompts)`; lazy import |
| `turing/src/ugv_perception/backend/instances.py` | engine arrays → `tuple[Instance, ...]` (CPU, testable) |
| `turing/src/ugv_perception/backend/openvino_gpu.py` | `OpenVinoGpuBackend`; `id = "openvino_gpu"` |
| `turing/src/ugv_perception/backend/cuda_pytorch.py` | `CudaPytorchBackend`; `id = "cuda_pytorch"`; **raise** if selected on this Arc box |
| `turing/src/ugv_perception/tests/test_backend_seam.py` | B1–B14; **no** OpenVINO required |

Do **not** edit `compose/`, `pack.py`, `tick.py`, T04 YAML, or `architecture.md`.  
Do **not** add a live_cam null backend.  
Do **not** implement PyTorch XPU.

---

## When coding (after this subarch is complete)

1. Freeze this file. Match **T06’s Protocol**, not the old T12 checklist.  
2. Implement `instances.py` + factory + seam tests **first** (no GPU).  
3. `OpenVinoGpuBackend` next; GPU/IR tests **skip** if weights/GPU missing (same honesty as T06 `infer`).  
4. `pip` extra: `openvino==2026.4.0` — optional extra, not a hard dep of `compose` tests.  
5. `load(..., device="GPU")`. **Never** fall back to CPU to “make it run.”  
6. Factory: `if backend_id == "openvino_gpu": from .openvino_gpu import ...` inside the function.  
7. Bind `prompts` at load: class index `i` (0-based in the prompt list) → `prompt_id = i + 1`. Never 0.  
8. `Instance.score`: Python `float`, finite, `[0,1]`, **copy** of the engine confidence. No minmax, no clip. Out of range → raise.  
9. `Instance.mask`: `dtype=bool`, 2-D, non-empty, `shape == rgb.shape[:2]`.  
10. Empty detections → `()`. That is not all-traversable.  
11. Engine failure → raise (`AdapterError` or Exception; T06 wraps).  
12. Do not call `pack()` or `make_mask()`.  
13. Do not import `compose`. Confirm `compose/` still has no `openvino` import.

---

## 1. Role

T12 is **how tensors run**. T06 is **what YOLOE means**. T07 is **the port**.

```
rgb uint8 HWC
    → OpenVINO 2026.4.0 device=GPU     # this box
    → instances_from_engine(...)       # CPU
    → tuple[Instance, ...]
    → (T06) pack → RawSemOutput → T03 → T04 → T05 → T07
```

Later NVIDIA: same `Instance` tuple, `CudaPytorchBackend`, less VRAM. Port topics unchanged.

---

## 2. Types (do not redefine)

Use T06’s:

```python
# already in adapter/yoloe.py
class InferenceBackend(Protocol):
    id: str
    def load(self, weights_path: str, **engine_args) -> None: ...
    def run(self, rgb: NDArray[np.uint8]) -> tuple[Instance, ...]: ...
```

`engine_args` **must** include prompts at load:

```python
backend.load(weights_path, prompts=prompts)  # tuple[str, ...], same object/order as YoloeAdapter
```

Factory:

```python
def build_backend(backend_id: str, weights_path: str, prompts: tuple[str, ...]) -> InferenceBackend:
    # lazy-import the impl; call load(); return ready backend
```

`backend_id` is `"openvino_gpu"` | `"cuda_pytorch"`. Anything else → raise.  
`"ultralytics"` / `"openvino_xpu"` / `"xpu"` → raise (not product).

---

## 3. `instances_from_engine` (CPU helper)

Input: `rgb_hw: (H, W)`, prompts `N`, parallel per-instance `class_index` (0-based), `score`, `mask` (bool **or** float in `[0,1]`; may be model resolution, not rgb).  
Output: `tuple[Instance, ...]` with **every** mask `shape == rgb_hw`.

This helper is the **only** place that may resize/threshold. `OpenVinoGpuBackend` and later `CudaPytorchBackend` must call it. They must not pick their own interpolation.

### Resize / threshold (mandatory, single recipe)

Let `mh, mw = mask.shape`, `H, W = rgb_hw`.

1. If `(mh, mw) == (H, W)`: do not resample.  
2. If mask is **bool** (or integer 0/1 already bool): resize with **nearest-neighbor** to `(H, W)`. Stay bool. No blur, no `>= 0.5` after the fact.  
3. If mask is **float** (soft, finite, values in `[0, 1]`): resize with **bilinear** to `(H, W)`, then `bool_mask = resized >= 0.5`. Threshold is **0.5**, not configurable in v1.  
4. Any other dtype / non-finite / float outside `[0, 1]` → raise (no clip).  
5. Scores and class indices are **per-instance scalars**. Do not spatially resample them.  
6. After this, `Instance.mask` is `np.bool_`, 2-D, non-empty, `shape == (H, W)`.

Port `scale` stays **1.0** because T06 `pack` then sees HW = rgb HW. Interpolation is an engine-boundary detail, not a second perception port.

| Rule | Why |
|---|---|
| `prompt_id = int(class_index) + 1` | T06 Y6; never 0 |
| `score` Python `float` via `float(x)` only after checking finite and in `[0,1]` | T04 identity; `float(np.float32)` is a Python float — **allowed here** at the engine boundary so pack’s `type(score) is float` passes |
| mask bool, 2-D, HW = rgb **after** the recipe above | T06 implementation note |
| class_index in `0..N-1` else raise | don’t invent prompts |
| no minmax / sigmoid / clip of **scores** | T04/T06 |

`float(np.float32(0.9))` is the **T12** conversion. T06 pack still requires `type is float`. That is the adapter/engine split: T12 may coerce numpy scalars to Python float; T06 pack does not.

---

## 4. OpenVINO GPU (this desktop)

- Pin **2026.4.0**. Compile/load IR from local `weights/yoloe-26s-seg.xml` (and `.bin`). No runtime download. No Ultralytics in `run()`.  
- `device="GPU"`. If GPU compile/load fails, **raise** — do not retry CPU.  
- One compiled model in memory. One frame in `run()` (T11 latest-only is T07/T11, not a batch here).  
- If IR masks are not rgb HW, pass them through **`instances_from_engine`** (nearest bool / bilinear-then-0.5 float). Do not `cv2.resize` ad hoc in `OpenVinoGpuBackend.run`. Port `scale` stays 1.0.  
- Do not keep RGB + two models resident (Depth is T08, sequential later).

---

## 5. CUDA PyTorch (later NVIDIA)

Class exists so the factory string is real. On **this Arc box**, `build_backend("cuda_pytorch", ...)` **raises** (wrong machine). Do not silently run CPU PyTorch. Implement `run()` later when that GPU exists; same `Instance` contract.

---

## 6. Invariants

| ID | Rule |
|---|---|
| B1 | `OpenVinoGpuBackend.id == "openvino_gpu"`; CUDA id `cuda_pytorch` |
| B2 | Implements T06 `InferenceBackend`; `run` returns `tuple[Instance, ...]` not a private tensor type |
| B3 | Factory unknown / `ultralytics` / `xpu` / `openvino_xpu` raises |
| B4 | Factory lazy-imports OpenVINO; `backend/__init__.py` and `factory.py` module import do not load `openvino` |
| B5 | `compose/` still has no `openvino` / `torch` / `ultralytics` import |
| B6 | `prompt_id` in `1..N`; never 0 |
| B7 | `type(score) is float`, finite, `[0,1]`; no clip/minmax |
| B8 | mask `bool`, 2-D, `shape == rgb.shape[:2]` after the single resize recipe (nearest bool / bilinear+0.5 float) |
| B9 | empty engine output → `()` |
| B10 | engine fail → raise; no all-traversable / all-unknown fake instances |
| B11 | `device` is GPU; no CPU fallback in OpenVINO impl |
| B12 | T12 does not import `compose`, `remap`, `confidence`, `freshness`, `pack` |
| B13 | `build_backend("cuda_pytorch", ...)` raises on this Intel box |
| B14 | Seam tests pass **without** OpenVINO installed |

GPU `run()` on real IR: skip until weights exist (not a dummy RGB outdoor claim).

---

## 7. Aptness

| Source | Fit |
|---|---|
| architecture §6 YOLOE is adapter | **High** — engine behind T06 |
| architecture §8 port unchanged | **High** |
| HARDWARE OpenVINO GPU vs XPU | **High** |
| T06 Protocol / Instance | **High** — this file yields to shipped T06 |
| T07 no engine import | **High** — lazy factory, compose untouched |
| Old T12 `BackendTensors` | **Rejected** — would break `YoloeAdapter.infer` |

---

## 8. Tests (`test_backend_seam.py`)

B1–B14 are **seam/contract tests** (factory, `instances_from_engine`, import graph). They must pass **without** OpenVINO installed, without a GPU, and without IR. They are not proof that Arc inference works.

**Separate** (skip unless local IR + GPU): `OpenVinoGpuBackend.run` on a real compiled model. That skip is not a dummy camera.

| ID | Test |
|---|---|
| B3 | bad backend id raises |
| B4 | `import ugv_perception.backend.factory` does not import `openvino` (sys.modules) |
| B5 | reuse compose import scan |
| B6–B8 | `instances_from_engine` happy path + id 0 rejected + score `1.1` raises + uint8 mask raises + bool mask 2×2 → rgb 4×4 via nearest + float mask bilinear then `>= 0.5` |
| B7 | `np.float32(0.9)` in → Python `float` out |
| B9 | empty → `()` |
| B13 | cuda_pytorch factory raises here |
| B14 | pytest without `openvino` package still collects/passes this file |

Skip (not part of B1–B14): compiled GPU/IR `run()`.

---

## 9. Done when

- **Seam:** factory + `instances_from_engine` + **B1–B14** pass with no OpenVINO, no GPU, no IR.  
- **Engine class:** `OpenVinoGpuBackend` exists and requests `device="GPU"` (no CPU fallback).  
- **Live OpenVINO `run()`:** skipped until `weights/yoloe-26s-seg.xml` exists — that skip does **not** fail B1–B14. Pin is **YOLOE-26s**, not v8s.  
- `YoloeAdapter.infer` + `compose_tick` unchanged and still green.

## 10. Non-goals

T02, T07 ROS, T08, T09, T11, training, Ultralytics runtime, PyTorch XPU, CPU fallback, forking `InferenceBackend`.
