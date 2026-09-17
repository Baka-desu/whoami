# Sub-architecture 4 — Confidence gates (T04)

**Task:** [T04](../tasks/T04-confidence.md)  
**Depends on (code/import):** T01 (`UNKNOWN`, `CANONICAL`, `assert_canonical`). Input arrays are T03’s output shape (`uint8` classes + `float32` scores), but T04 does **not** import `remap`.  
**Does not import / need at build:** T02, T05, T06, T07, GPU, camera. Tests are numeric tables.  
**Runtime contract (not an import):** T05/T07 consume `collapse_candidate` when deciding `degraded` / `producer_ok`. T04 never publishes. Adapters (T06) that use `identity` must already emit `[0,1]` — T04 does not clip them into legality.  
**Authority:** [`architecture.md`](../../architecture.md) §8.3, §8.6, §16 (gates in port-normalized `[0,1]`; no shared τ on raw multi-model scores; unknown ≠ free)  
**Not authority:** `dev.md` hours; `HARDWARE.md` (T04 is CPU numpy; no OpenVINO)  
**Code later:** `ugv_perception/confidence/` + `config/perception/<adapter_id>.yaml`. No node, no `CanonicalMask`, no `/ugv/perception_degraded`.

This file is the architecture of T04. The task file is the build checklist. If they disagree, this file wins, then `architecture.md`.

---

## 1. Role in the product

T03 made names into `{0,1,2}`. Those class ids are still only as good as the scores behind them. Architecture §8.3: τ apply in **port-normalized `[0,1]`**, per adapter, never on raw multi-model output.

T04 is that step: **normalize (honestly) → gate → maybe reclass to unknown.** It does not decide freshness, does not mint a port mask, and does not own degraded.

```
T03 classes {0,1,2} + raw_scores
        │
        ▼
┌────────────── T04 ──────────────┐
│  load per-adapter GateProfile   │
│  normalize → conf in [0,1]      │
│    identity: already [0,1] or ERROR (no clip)
│    sigmoid: explicit center/scale
│    minmax: FORBIDDEN (v1)
│  τ_min / τ_trav / τ_haz / κ     │
│  known_fraction after all gates │
│  collapse_candidate (bool)      │
│  never sets degraded            │
└──────────────┬──────────────────┘
               │ classes, conf, known_fraction, collapse_candidate
               ▼
            T05 / T07  →  make_mask(producer_ok=...)
```

Per-frame minmax is not a normalizer for this product. A constant `0.42` map becoming `1.0` is fake confidence and defeats fail-closed gates.

---

## 2. Scope (tight)

### T04 owns

| Piece | Why |
|---|---|
| Per-adapter gate YAML | §8.3 profiles; no shared global τ |
| `GateProfile` load (fail closed) | illegal `minmax`, missing sigmoid params |
| `normalize` + `apply_gates` | port-normalized `[0,1]`, then τ and κ |
| `known_fraction` / `collapse_candidate` | collapse **signal** after all gates |
| Reclass to `0` only | never invent hazard; never `0 → 1` |

### T04 does not own

| Piece | Owner |
|---|---|
| `{0,1,2}` / `assert_canonical` / `make_mask` | T01 — T04 **calls** `assert_canonical` on output classes |
| Remap YAML / LUT | T03 |
| `degraded`, `/ugv/perception_degraded`, `producer_ok` | T05 / T07 |
| Topics, stamps, `frame_id` | T07 |
| YOLOE / OpenVINO GPU / CUDA | T06 / T12 |
| Camera | T02 (skipped) |
| Learned τ | deferred §14 |

Do not return `CanonicalMask`. Do not set `degraded = collapse_candidate`.

---

## 3. Types

```python
@dataclass(frozen=True, slots=True)
class GateProfile:
    adapter_id: str              # Python str, non-empty
    normalizer: str              # "identity" | "sigmoid"  — never "minmax"
    tau_min: float               # Python float, finite, in [0, 1]
    tau_trav: float
    tau_haz: float
    kappa: float                 # in [0, 1]; compared in normalized space
    min_known_fraction: float    # in [0, 1]
    sigmoid_center: float | None # required iff normalizer == "sigmoid"
    sigmoid_scale: float | None  # required iff sigmoid; finite and > 0

def load_gates(path: str | Path) -> GateProfile: ...

def apply(
    profile: GateProfile,
    *,
    adapter_id: str,
    classes: NDArray[np.uint8],       # H,W  {0,1,2} from T03
    raw_scores: NDArray[np.float32],  # H,W  not yet port-normalized unless identity
    runner_up: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], float, bool]:
    """(classes_out, conf, known_fraction, collapse_candidate).
    T04 never mutates classes or raw_scores in place.
    T04 never sets degraded."""
```

**YAML** (`config/perception/<adapter_id>.yaml`):

```yaml
adapter_id: yoloe
normalizer: identity
tau_min: 0.25
tau_trav: 0.50
tau_haz: 0.35
kappa: 0.10
min_known_fraction: 0.05
```

v1 YOLOE: `identity` — YOLOE-seg scores are in the **numeric domain** `[0,1]`, which is what the τ contract needs. That is **not** a claim they are well calibrated.

**Do not architect the numeric values.** Architecture / this subarch own: the **fields exist**, they are Python floats in `[0, 1]`, gate **order**, strict `<`, κ only on survivors, collapse formula. They do **not** own `0.25` / `0.50` / `0.35` / `0.10` / `0.05`. Those figures in the YAML block above are a **starting example** for `config/perception/yoloe.yaml`, not G-invariants and not `architecture.md`. Do not hardcode them in T04. Validate on real outdoor frames later (not now: night, no camera). Until then they are YAML, not learned truth.  
If `normalizer: sigmoid`, **both** `sigmoid_center` and `sigmoid_scale` must be present at load (Python float, finite; `scale > 0`). No apply-time defaults.

**Repo convention (not architecture.md):** filename stem equals `adapter_id` (same idea as T03 C1). `load_gates` still checks it in this tree.

---

## 4. Normalizers

| Name | Rule | v1 |
|---|---|---|
| `identity` | `conf` is `raw_scores` **if and only if** every value is finite and ∈ `[0,1]`. Else **ERROR**. **No clip.** `17.3` must not become `1.0`. Domain `[0,1]` ≠ calibrated probability. Same-object return is allowed because T04 does not write `conf` in this mode. | **Default for YOLOE** |
| `sigmoid` | `conf = 1 / (1 + exp(-(raw - center) / scale))` — new array, already in `(0,1)`. `center`/`scale` from YAML only. | Allowed for logit adapters |
| `minmax` | per-frame `(x-min)/(max-min)` | **Forbidden.** Load **ERROR**. Constant `0.42` → `1.0` is not confidence. Relative ranks are not absolute τ. |

Do not add a fourth name. Unknown `normalizer` string → load ERROR.

---

## 5. Invariants (fail closed)

Every ID has a test. `type(x) is T`, not `isinstance` where `bool` would sneak in.

### Load (`load_gates`)

| ID | Rule | Architecture |
|---|---|---|
| G1 | File exists; missing YAML raises | per-adapter profile §8.3 |
| G2 | `type(adapter_id) is str`, non-empty | §8.3 |
| C1 | Repo convention: path stem == `adapter_id` | this tree |
| G3 | `normalizer` is `"identity"` or `"sigmoid"`. `"minmax"` → ERROR | fail closed; not §8.3 minmax |
| G4 | `tau_min`, `tau_trav`, `tau_haz`, `kappa`, `min_known_fraction`: `type is float`, finite, ∈ `[0, 1]` | §8.3 |
| G5 | If `sigmoid`: `sigmoid_center` and `sigmoid_scale` present, `type is float`, finite, `scale > 0`. If `identity`: those fields must be absent or ignored-as-unused at load — **prefer absent**; extra sigmoid fields with identity is a load ERROR (wrong profile) | no hidden params |
| G6 | `bool` / `int` `1` for a τ is TypeError (`True` is not a float) | same strictness as T01/T03 |

### Apply

| ID | Rule | Architecture |
|---|---|---|
| G7 | `type(adapter_id) is str` and equals `profile.adapter_id` | wrong profile is a bug |
| G8 | `classes` pass T01 `assert_canonical` **on input** (uint8, `{0,1,2}`, 2-D, non-empty) | T03 already certified; T04 re-checks |
| G9 | `raw_scores` is `float32`, same HW as `classes` | §8.3 |
| G10 | `identity`: any `nan` / `inf` / `<0` / `>1` → ERROR (no clip). Constant `0.42` stays `0.42`. Domain check only, not calibration | §8.3 numeric `[0,1]` |
| G11 | `sigmoid`: write a **new** `conf` array; do not write `raw_scores` | T03 identity of scores |
| G12 | Gate order after normalize: `tau_min`, then `tau_trav` on class 1, then `tau_haz` on class 2, then `kappa`. Comparisons are **strict `<`**. Reclass **to `0` only** | §8.6 |
| G13 | Never `0 → 1` or `0 → 2`. Never class `1` because a gate failed. Weak hazard → `0`, not traversable | unknown ≠ free |
| G14 | `kappa`: if `runner_up is None`, skip. If present: same HW, `float32`, **normalize with the same profile**. Apply only to pixels that **already survived** τ: `(classes_out != 0) & ((conf - conf2) < kappa)` → class `0`. Meaning: among survivors, reject a too-small normalized top-1 margin. `0 → 0` is harmless but not what κ is for. Do not fake a runner-up. Do not compare raw scores | §8.3 same `[0,1]` space |
| G15 | After **all** gates including kappa: `known = classes_out != 0`; `known_fraction = count(known) / N` (`float`); `collapse_candidate = known_fraction < min_known_fraction` (strict `<`) | collapse ≠ all-unknown-valid |
| G16 | Return `(classes_out, conf, known_fraction, collapse_candidate)`. `collapse_candidate` is `type is bool`. **No `degraded` field.** `classes_out is not classes` (copy). `raw_scores` unchanged | T05/T07 own degraded |
| G17 | Output `classes_out` passes `assert_canonical`. `conf` is `float32`, same HW, finite, ∈ `[0,1]` | T01 I1–I3 |

Do not coerce illegal identity scores to `[0,1]`. Do not implement minmax “just for tests.”

---

## 6. Apply sequence

```
assert adapter_id / shapes                    # G7–G9
conf = normalize(profile, raw_scores)         # G10–G11  (identity: conf is raw_scores)
classes_out = classes.copy()                  # never write the T03 array
# gates on classes_out, reading conf
classes_out[conf < tau_min] = 0
classes_out[(classes_out == 1) & (conf < tau_trav)] = 0
classes_out[(classes_out == 2) & (conf < tau_haz)] = 0
if runner_up is not None:
    conf2 = normalize(profile, runner_up)     # same normalizer
    classes_out[(classes_out != 0) & ((conf - conf2) < kappa)] = 0
known_fraction = (classes_out != 0).mean()    # after kappa
collapse_candidate = known_fraction < min_known_fraction
assert_canonical(classes_out)
return classes_out, conf, known_fraction, collapse_candidate
```

Numpy boolean masks, not Python per-pixel loops. CPU only. `HARDWARE.md` does not apply.

`known_fraction.mean()` on uint8/bool is a numpy float; return Python `float(...)`. `collapse_candidate` is `type is bool` (`bool(numpy_scalar)` is OK **after** the comparison iff we then check `type is bool` — construct with `bool(...)` only for the returned flag, not for τ).

---

## 7. Collapse vs unknown vs degraded

```
all pixels unknown, known_fraction == 0, but producer still honest
        → collapse_candidate True if 0 < min_known_fraction
        → T07 may set degraded
        → that is NOT “unknown = free”

all-unknown + valid=true (no collapse, e.g. min_known_fraction == 0)
        → legal. Costmaps inflate. Dev 5 does not have to hold.

T04 collapse_candidate
        → T05/T07 may OR it into degraded
        → T04 itself has no degraded bit and no publisher
```

Implementors of T04 must not write `degraded = known_fraction < min_known_fraction`.

---

## 8. Module layout (when coded)

```
ugv_perception/
  confidence/                 # T04 only
    profile.py                # GateProfile
    load.py                   # load_gates
    apply.py                  # normalize + gates
  tests/test_confidence.py
config/perception/
  yoloe.yaml                  # product profile (not code)
```

T04 imports `ugv_perception.port` only.  
`ugv_perception.port` and `ugv_perception.remap` must not import `confidence`.

---

## 9. Efficiency

| Choice | Why |
|---|---|
| Vectorized numpy gates | cheap vs YOLOE; no GPU |
| identity conf = same object | no copy when adapter already `[0,1]` |
| classes copied once | T03 array stays immutable |
| One profile load at node start | T07 holds `GateProfile` |

Do not: OpenVINO for τ, per-frame minmax “to use the GPU range,” clipping identity.

---

## 10. Aptness vs `architecture.md`

| Clause | Fit | Notes |
|---|---|---|
| §8.3 gates in port-normalized `[0,1]`; per-adapter; no shared raw τ | **High** | identity/sigmoid only; κ in normalized space |
| §8.6 unknown ≠ free; gate fail → degraded **system** | **High** | reclass to 0; collapse is a **candidate** for T05/T07, not a T04 topic |
| §16 shared τ on raw multi-model conf | **High** | per-adapter YAML; no minmax fake-1.0 |
| Kill: unknown-as-free | **High** | G13; default unknown on fail |
| Learned calibration §14 | **N/A (correct miss)** | YAML τ only |

Original T04 checklist allowed `minmax` and clipped `identity`. That contradicted fail-closed. This file forbids both.

---

## 11. Tests (numeric, no images)

**Every G1–G17 has a test.**

| ID | Tests |
|---|---|
| G1 | Missing file raises. |
| G3 | `normalizer: minmax` load ERROR. |
| G4–G6 | `tau_min: true` TypeError; `tau_min: 1` TypeError; `tau_min: 1.5` ValueError. |
| G5 | sigmoid without center/scale load ERROR; `sigmoid_scale: 0.0` ERROR. |
| G7 | apply with other `adapter_id` raises. |
| G10 | identity `1.0001` / `-0.1` / `nan` ERROR; constant `0.42` stays `0.42`. |
| G12 | `0.1` class 1 → 0; `0.9` class 1 `tau_trav=0.5` stays 1; `0.4` class 2 `tau_haz=0.35` stays 2; `0.2` class 2 → 0. |
| G13 | all-0 input stays 0. |
| G14 | runner-up normalized; no runner-up → skip; raw-scale gap must not be used. |
| G15 | `known_fraction` after kappa; `!= 0` over all pixels. |
| G16 | return has no degraded; `collapse_candidate is True` does not set one; `raw_scores` unchanged; `classes_out is not classes`. |
| G17 | output canonical + conf in `[0,1]`. |

Two different YAML files → two profiles, no shared τ object.

---

## 12. Done when

- `GateProfile` + `load_gates` + `apply` exist; port/remap do not import them.
- G1–G17 pass with no camera.
- Product `config/perception/yoloe.yaml` is a file (`identity`), not hardcoded τ.
- T07 is specified to pass `collapse_candidate` into degraded / `producer_ok`, not to read a T04 `degraded` field.

## 13. Non-goals

Publishing, freshness, OpenVINO, minmax, learned τ, T02, Depth Anything.
