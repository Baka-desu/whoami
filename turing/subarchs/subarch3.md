# Sub-architecture 3 — Remap kernel (T03)

**Task:** [T03](../tasks/T03-remap.md)  
**Depends on (code/import):** T01 port kernel ([subarch1](subarch1.md)) for `{0,1,2}` and `assert_canonical`  
**Does not import / need at build:** T02 (skipped), T06, GPU, camera. T03 tests do not load YOLOE.  
**Runtime contract (not an import):** any adapter that calls `apply` (T06, T09) must present T03’s input types — including Python `int` keys on `id_to_name`. That is an adapter obligation, not a T03→T06 dependency.  
**Authority:** [`architecture.md`](../../architecture.md) §3, §8.1, §8.2, §8.6, §16 (no remap → no publish; unknown ≠ free; no `cautious`)  
**Not authority:** `dev.md` hours; YOLOE prompt lists (T06); τ (T04)  
**Code later:** `ugv_perception/remap/` + ontology YAML. No node, no model, no `CanonicalMask`.

This file is the architecture of T03. The task file is the build checklist. If they disagree, this file wins, then `architecture.md`.

---

## 1. Role in the product

Adapters speak model names (`person`, `dirt_path`, ONNX `sidewalk`). The brain and Dev 3 speak only `{0,1,2}`.

T03 is the **legal boundary** in architecture §8.2: every adapter **must** ship a remap YAML; **no remap → must not publish**. T03 is that YAML as an executable table. It does not run YOLOE, does not gate confidence, does not publish, and does not mint a port mask.

```
T06 RawSemOutput (names + raw scores + ids)
        │
        ▼
┌────────────── T03 ──────────────┐
│  load YAML → RemapTable         │
│  LUT: name → {0,1,2}            │
│  nameless id → ERROR (R11)      │
│  known name, no YAML → 0 (R12)  │
│  scores identity; never mutate  │
└──────────────┬──────────────────┘
               │ uint8 classes (pre-gate)
               ▼
            T04 gates → T05 → T07 make_mask
```

T01 still owns “is this a legal port sample.” T03 owns “did this adapter’s names become `{0,1,2}`.” T07 must not publish T03’s array until T01 `make_mask` says yes.

---

## 2. Scope (tight)

### T03 owns

| Piece | Why |
|---|---|
| Ontology YAML schema | §8.2 each adapter ships remap YAML |
| `RemapTable` (load-time, fail closed) | illegal class IDs must not reach the port |
| `apply(...)` vectorized LUT | name → `{0,1,2}` per pixel |
| v1 `default == 0` | unnamed / unknown name → unknown, never free |
| Output `uint8` classes that satisfy T01 I1–I2 | so T04 never sees a `3` |

### T03 does not own

| Piece | Owner |
|---|---|
| Canonical IDs, `assert_canonical`, `make_mask` | T01 — T03 **calls** them; it does not redefine them |
| `ImageFrame`, camera, calibration | T02 (skipped) |
| YOLOE prompts, backends, OpenVINO/CUDA | T06 / T12 |
| `RawSemOutput` as a whole (stamp, frame_id) | T06. T03 takes a **minimal input** (below) |
| τ / normalize | T04 |
| freshness / `producer_ok` / degraded | T05 / T07 |
| Topics | T07 |
| Depth | T08 |
| ONNX tutorial ontology as product | T09 (own YAML, not `live_cam`) |

Do not return `CanonicalMask`. That would skip T04/T05 and steal T01.

---

## 3. Types

```python
@dataclass(frozen=True, slots=True)
class RemapTable:
    adapter_id: str                 # Python str, non-empty; matches YAML + apply()
    name_to_class: dict[str, int]   # name → 0|1|2; keys Python str; values Python int
    default: int                    # v1: must be 0 (UNKNOWN)

def load_remap(path: str | Path) -> RemapTable: ...

def apply(
    table: RemapTable,
    *,
    adapter_id: str,
    label_ids: NDArray[np.int32],    # H,W
    id_to_name: dict[int, str],      # Python int → Python str (T06 converts)
    raw_scores: NDArray[np.float32], # H,W; not [0,1] yet
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """classes in {0,1,2}; scores is raw_scores (same object). T03 never writes scores."""
```

T06’s `RawSemOutput` is a **superset**. T07 unpacks it into `apply`. T03 does not import stamp/`frame_id` and must not rewrite them.

**Adapter runtime contract (callers of `apply`, not a T03 import of T06):** NumPy pipelines often use `np.int64` keys. T03 still requires `type(key) is int`. **T06** (and T09) must convert `id_to_name` keys with `int(k)` before `apply`. T03 does not import those modules and must not call `int()` on keys to “be nice” — that would hide adapter bugs and accept `True`. T03 can be implemented and tested with table fixtures only.

**YAML** (`config/ontologies/`): keys are **arbitrary adapter vocabulary** (YOLOE prompts, ONNX class names). Values are **only** `{0,1,2}`:

```
dirt_path / gravel / grass  → 1 traversable
person / vehicle / rock     → 2 hazard
vegetation / sky            → 0 unknown
```

**Repository convention (not architecture.md):** files are named `<adapter_id>.yaml` (e.g. `yoloe.yaml` with `adapter_id: yoloe`). §8.2 requires a per-adapter YAML, not that the stem equal the id. `load_remap(path)` still checks stem == `adapter_id` in this repo so two adapters cannot share a misnamed file. That check is convention, not a product-port law.

**YAML example:**

```yaml
adapter_id: yoloe
map:
  dirt_path: 1
  gravel: 1
  grass: 1
  sky: 0
  person: 2
  vehicle: 2
  rock: 2
  water: 2
  fence: 2
  tree: 2
  vegetation: 0
default: 0
```

`map` **names** are unrestricted strings. `map` **values** and `default` are **Python `int` in `{0,1,2}`** after load. `true` / `yes` / `1.0` / `True` are load errors (`type(v) is int`, not `isinstance` — `bool` subclasses `int`).

**v1 product default is `0`.** Architecture: ambiguous and unnamed → `unknown` (inflate). A YAML `default: 1` would make unknown names traversable (kill list: unknown-as-free). `default: 2` would invent hazards. Neither is v1.

Grass `1` vs vegetation `0` is **product YAML**, not code. Change the file, not T03.

T09 gets `config/ontologies/onnx.yaml` later. Missing file for an `adapter_id` → cannot construct / cannot apply.

---

## 4. Invariants (fail closed)

Every ID has a test. Annotations are not enforcement. `type(x) is T`, not `isinstance` where `bool` would sneak in.

### Load (`load_remap`)

| ID | Rule | Architecture |
|---|---|---|
| R1 | File exists; missing YAML raises. No identity remap, no “pass labels through” | §8.2 no remap → no publish |
| R2 | `type(adapter_id) is str` and non-empty; YAML `adapter_id` is the table identity | §8.2 per-adapter file |
| C1 | **Repo convention:** if loaded from a path, filename stem equals `adapter_id`. Not required by §8.2 | this tree only |
| R3 | Every `map` key is a non-empty Python `str` | names, not prompt tensors |
| R4 | Every `map` value and `default`: `type is int` and ∈ `{0,1,2}` | §8.1 |
| R5 | No `3`, no `cautious`, no string `"1"` | §8.1, §14 |
| R6 | v1: `default == 0` | §8.6 unknown ≠ free |
| R7 | Duplicate YAML keys: loader must not silently keep one (fail or round-trip check) | fail closed |

Empty `map:` with `default: 0` is legal (everything unknown). Useless, not illegal.

### Apply

| ID | Rule | Architecture |
|---|---|---|
| R8 | `type(adapter_id) is str` and `adapter_id == table.adapter_id` else raise | wrong ontology is a bug |
| R9 | `label_ids` is `ndarray`, `dtype == int32`, 2-D, non-empty, all values `>= 0` | adapter dense map |
| R10 | `id_to_name`: keys `type is int` (`>= 0`), values non-empty Python `str`. `np.int64` keys **fail**. Callers convert; T03 does not import T06 | LUT is exact; adapter runtime contract |
| R11 | Every distinct id in `label_ids` is a key of `id_to_name` else **raise**. Do not default a nameless id | adapter failed to name the id |
| R12 | Name is in `id_to_name` but **not** in YAML `map` → `default` (0). Only this path defaults | known label, product did not assign it |
| R13 | `raw_scores` is `ndarray` `float32`, **same H×W** as `label_ids`. Not checked for `[0,1]` (T04). Return **the same object**. **T03 never mutates `raw_scores`** (no clip, no fill, no in-place scale) | §8.3 normalize is T04 |
| R14 | Output `classes` is `uint8` 2-D, same HW, every pixel ∈ `{0,1,2}`; run T01 `assert_canonical`. T03 must **not** rewrite `0 → 1` | §8.1, I10 |
| R15 | Apply is a **LUT + gather**, not a Python per-pixel loop | efficiency; no GPU |

Construction:

```
table = load_remap(path)
  R1–R7 fail → raise (no table)

classes, scores = apply(table, adapter_id=..., label_ids=..., id_to_name=..., raw_scores=...)
  R8–R14 fail → raise (no array)
  scores is raw_scores
```

Do not coerce illegal YAML to `0` at load. Coercion of **unmapped names** to `default` at apply is the specified policy, not a type coercion.

### Two different “unknowns” (do not collapse)

```
ID absent from id_to_name
        ↓
       ERROR (R11)
       adapter failed to tell us what the model ID means

Name present in id_to_name, absent from YAML map
        ↓
       0 (R12)
       known model label, but the product does not assign it
```

R11 is an adapter defect. R12 is a product ontology gap (inflate, never free). T06/T07 must not treat them as the same “unknown.”

---

## 5. Apply mechanics (LUT)

At `apply` (not at load — `id_to_name` is per frame / per adapter instance):

1. `max_id = max(id_to_name)` (and `>=` max of `label_ids` after R11).
2. `lut = np.full(max_id + 1, table.default, dtype=np.uint8)`.
3. For each `i, name` in `id_to_name`: `lut[i] = table.name_to_class.get(name, table.default)`.
4. `classes = lut[label_ids]` (numpy gather).
5. `assert_canonical(classes)`.
6. Return `(classes, raw_scores)` — same scores object; **no writes** to `raw_scores` or `label_ids`.

Negative ids never index the LUT (R9). GPU stays off: remap is CPU and cheap next to YOLOE. Hardware.md does not apply.

---

## 6. Module layout (when coded)

```
ugv_perception/
  remap/                    # T03 only
    table.py                # RemapTable
    load.py                 # load_remap
    apply.py                # apply + LUT
  tests/test_remap.py
config/ontologies/
  yoloe.yaml                # product ontology (not code)
```

T03 imports `ugv_perception.port` (`UNKNOWN`, `CANONICAL`, `assert_canonical`).  
`ugv_perception.port` must not import `remap`.

PyYAML `safe_load` only. No GPU libraries.

---

## 7. Efficiency

| Choice | Why |
|---|---|
| LUT + gather | O(ids + pixels) C-level; no Python over H×W |
| CPU numpy | remap is not inference; keep OpenVINO GPU / CUDA for T06 |
| Scores identity + no mutate | extra `float32` copy would double bandwidth; in-place clip would corrupt T04 |
| Load once | `RemapTable` is immutable; T07 holds one per adapter |
| Small dict | dozens of names, not a dataset |

Do not: GPU remap, batched images, embedding prompts in the LUT, calling OpenVINO.

---

## 8. Aptness vs `architecture.md`

| Clause | Fit | Notes |
|---|---|---|
| §3 adapters implement the port; port is canonical | **High** | T03 is the name→canonical step. Models stay T06. |
| §8.1 three IDs, no `cautious` | **High** | R4–R5, R14 + T01 `assert_canonical`. |
| §8.2 remap YAML mandatory; no remap → no publish | **High** | R1. T07 still must refuse to start without a table (T03 cannot publish). |
| §8.3 conf normalize | **N/A (correct miss)** | Scores untouched. T04. |
| §8.6 unknown ≠ free | **High** | v1 `default == 0` (R6). Unmapped name → 0, not 1. |
| §9 geometry | **N/A** | T03 does not touch depth. |
| §16 adapter publish without remap; unknown-as-free | **High** | R1; R6/R12. |
| Kill: tutorial-as-only-ontology | **High** | Example YAML is YOLOE outdoor; T09 is a second file. |
| DoD 3: adapter swap via remap + conf profile only | **Partial (T03 half)** | Swap = different YAML + T06 adapter. Same `apply`. |

Original T03 checklist was already the right task. Gaps this file closes: `default` locked to 0; YAML `bool`/`float` rejected; LUT not a pixel loop; output is not `CanonicalMask`; nameless ids raise; scores are identity not a copy; T03 calls T01 certify.

---

## 9. Tests (tables, not images)

No RGB, no camera, no weights. **Every R1–R15 has a test.**

| ID | Tests |
|---|---|
| R1 | Missing file raises. |
| R2 | Empty YAML `adapter_id` raises. |
| C1 | `yoloe.yaml` with `adapter_id: onnx` raises (repo convention). |
| R3–R5 | `cautious: 1` or `person: 3` or `person: true` or `person: 1.0` fails at load. |
| R6 | `default: 1` fails at load. `default: 0` passes. |
| R8 | Apply with other `adapter_id` raises. `adapter_id=1` TypeError. |
| R9 | `float32` labels raise. 1-D raises. Negative id raises. |
| R10 | `{np.int64(1): "person"}` TypeError (T06 must convert; T03 does not). |
| R11 | Pixel id `9` not in `id_to_name` raises (not silently 0). |
| R12 | Name `mystery` in `id_to_name` but not in map → class 0. `person` → 2. `dirt_path` → 1. |
| R13 | Returned scores `is` input scores; values unchanged after apply (no mutation). Shape mismatch raises. |
| R14 | Output dtype `uint8`; `assert_canonical` passes; all-unknown stays 0. |
| R15 | (optional bench, not required) apply on a large array does not use a Python `for` over pixels — enforced by code review of `apply`, not by a dummy image. |

A second adapter id without a file cannot `load_remap`. No model is loaded.

---

## 10. Done when

- `RemapTable` + `load_remap` + `apply` exist and T01 does not import them.
- R1–R15 tests pass with no camera.
- Product `yoloe.yaml` is a file, not hardcoded if/else on names.
- T07’s composition path is specified to call `apply` then T04 then `make_mask` (not `CanonicalMask(...)`).

## 11. Non-goals

Prompts, backends, τ, freshness, topics, CameraInfo, Depth Anything, training, outdoor night testing.
