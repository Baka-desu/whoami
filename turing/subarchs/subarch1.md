# Sub-architecture 1 — Port kernel (T01)

**Task:** [T01](../tasks/T01-types-and-port-contract.md)  
**Authority:** [`architecture.md`](../../architecture.md) §3, §8.1, §8.3–§8.6, §12 (perception watch), §16 kill list  
**Not authority:** `dev.md` hour boxes; `interfaces.md` as a dump of all Dev 1 types  
**Code later:** `ugv_perception` types + validators only. No node, no adapter, no YAML.

This file is the architecture of T01. The task file is the build checklist. If they disagree, this file wins, then `architecture.md`.

---

## 1. Role in the product

The rest of the product depends on a **canonical semantic mask, port-normalized confidence, and source-frame metadata** — not on YOLOE.

T01 is the **port kernel**: the executable form of architecture §8. It does not ingest cameras, remap labels, run models, or publish. It defines what a legal port *is*, and rejects everything else.

The port object is `mask + confidence + header + validity + informational age + scale`. Class IDs `{0,1,2}` are the semantic payload, not the whole contract.

```
later tasks produce bytes
        │
        ▼
┌─────────────── T01 ───────────────┐
│  FrameHeader  (stamp, frame_id)   │
│  CanonicalMask {0,1,2} + conf     │
│  PortMeta (valid, age, scale)     │
│  validators (fail closed)         │
└───────────────┬───────────────────┘
                │ T07 may publish only if T01 says yes
                ▼
     /segmentation/mask (+ conf, meta, degraded)
```

T01 is the type boundary that makes “YOLOE as brain” impossible: downstream of this kernel there are no prompts, no backends, no raw scores.

---

## 2. Scope (tight)

### T01 owns

| Piece | Why it is T01 |
|---|---|
| Canonical IDs `0,1,2` and names | §8.1 — brain/costmap bind **only** to these |
| Cost *intent* as comments/constants (`unknown` never free, `hazard` lethal) | §8.1; Dev 3 implements inflation, we do not |
| `FrameHeader {stamp_ns, frame_id}` | §8.4 identity of every port message |
| `CanonicalMask` | classes + port-normalized confidence + header + `valid` |
| `PortMeta` ROS field list | §8.4 “at least” stamp, frame_id, age, valid; §8.5 scale |
| Encoding names | mask `mono8`, confidence `32FC1` |
| Validators | fail closed before T07 can publish |
| CameraInfo **pairing identity** | §8.5: `CameraInfo.header.frame_id == mask.header.frame_id`. T01 does **not** validate K/D/size. T02 owns calibration truth |

### T01 does not own

| Piece | Owner |
|---|---|
| `ImageFrame` (RGB + CameraInfo blob) | T02 |
| `RawSemOutput` (model ids, names, raw scores) | T06 |
| `DepthFrame` | T08 — **not the port** (§9) |
| `Source` / `Adapter` / `InferenceBackend` protocols | T02 / T06 / T12 |
| Remap YAML | T03 |
| τ gates | T04 |
| `perception_max_age` policy | T05 |
| Topics / publishers | T07 |
| `/ugv/perception_degraded` | T05+T07 (T01 may type a bool; it does not publish) |

The original T01 brief pulled `ImageFrame`, `RawSemOutput`, and `DepthFrame` in from `interfaces.md`. That mixed **port** with **adapter** and **geometry**. Geometry on the port kernel contradicts §9. Those types stay in `interfaces.md` as a Dev 1 map; they are not T01 deliverables.

---

## 3. Types

```python
UNKNOWN, TRAVERSABLE, HAZARD = 0, 1, 2
CANONICAL = frozenset({0, 1, 2})
MASK_ENCODING = "mono8"
CONF_ENCODING = "32FC1"

@dataclass(frozen=True, slots=True)
class FrameHeader:
    stamp_ns: int          # Python int, > 0; image time, never receive time, never now
    frame_id: str          # optical / camera frame of the mask

@dataclass(frozen=True, slots=True)
class CanonicalMask:
    header: FrameHeader
    classes: NDArray[np.uint8]       # H,W  values in {0,1,2}
    confidence: NDArray[np.float32]  # H,W  port-normalized [0,1]
    valid: bool                      # see I8 — not “arrays looked legal”
    age_s: float                     # informational snapshot only (see below)
    scale: float                     # v1 product: must be 1.0
```

**`valid`:** `True` only when (1) every structural T01 invariant holds and (2) the producer passed `producer_ok=True`. T01 does not own adapters or gates and must not infer `valid` from “the arrays passed validation.” T04 (gates) and T05 (freshness / adapter-fail at the node) independently pass `producer_ok=False`. Illegal arrays **raise**; they are not represented as `valid=False` masks.

`producer_ok` is a **Python `bool`**, checked at runtime (`type(producer_ok) is bool`). Truthy stand-ins (`1`, `"yes"`, `np.True_`) raise `TypeError`. Dataclass annotations are not enforcement. Stored `CanonicalMask.valid` is also a Python `bool`.

**`age_s`:** informational metadata, `now - stamp` at construction. **Not** the freshness authority. Freshness decisions MUST use **current** time against `header.stamp` (architecture §8.4 consumer reject). Dev 5 must not trust a stored `age_s` sitting in a queued message. Same rule as `PortMeta.age`.

**`scale`:** isotropic resolution ratio, when used:

`mask_height / source_height == mask_width / source_width == scale`

A single scalar is illegal if the two axis ratios differ. **v1 product port: `scale` must be `1.0`** (mask HW equals source HW). Non-`1.0` is reserved for a later adapter that explicitly documents downscale; it is not a v1 product value.

Numpy buffers are views. `frozen=True` does not freeze array contents; validators and T07 must not mutate a published mask in place. Callers who need to edit allocate a new array.

ROS msg (compiled in T07, **specified here**):

```
# ugv_perception/msg/PortMeta.msg
std_msgs/Header header    # stamp + frame_id identical to the mask
bool valid
float32 age               # informational only. Freshness = now vs header.stamp
float32 scale             # v1: 1.0
```

**Deliberately omitted from `PortMeta`:**

- `adapter_id` — architecture port is adapter-agnostic. Putting it on the product msg invites Dev 3/5 to branch on YOLOE. Log it; do not contract it.
- `degraded` — architecture already has `/ugv/perception_degraded`. Duplicating it on meta can desync from the Bool topic. `valid=false` is the per-message bit; degraded is the stream-level bit (T05/T07).

`CanonicalMask.age_s` and `PortMeta.age` are the same informational snapshot. Neither is a substitute for `now - header.stamp` at the consumer.

---

## 4. Invariants (fail closed)

Checked by named functions. T07 calls them before any publish. T10 calls them on the wired composition.

| ID | Rule | Architecture |
|---|---|---|
| I1 | `classes.dtype == uint8`, 2-D, every pixel ∈ `{0,1,2}` | §8.1 |
| I2 | no `cautious` / `3` / `255` | §8.1, §14 deferred fourth class |
| I3 | `confidence` float32, same H×W as `classes`, finite, in `[0,1]` | §8.3 |
| I4 | `type(stamp_ns) is int` and `stamp_ns > 0`; it is the **source image** time (never receive time, never now). Reject `1.5`, `np.int64`, `np.float64`, `True` (bool is not int for this check). `age_s` is finite and `>= 0` (informational; freshness uses **current** now vs this stamp) | §8.4, §8.5 no silent reuse |
| I5 | `header.frame_id` is the optical frame, **non-empty** | §8.4, §8.5 |
| I6 | v1: `scale == 1.0` and, if source HW is provided, mask HW equals source HW. `scale` non-finite or `<= 0` fails. Non-uniform axis ratios fail (isotropic definition in §3) | §8.5 |
| I7 | mask header equals confidence header (when both exist) | §8 |
| I8 | `type(producer_ok) is bool` (else `TypeError`). `valid is True` only if I1–I7, I9–I10 hold **and** `producer_ok is True`. Passing validation of the arrays is not sufficient. T04/T05 set `producer_ok=False`; T01 does not mention adapters or τ | §8.4 |
| I9 | **Pairing identity only:** if a `CameraInfo` is supplied, `CameraInfo.header.frame_id == CanonicalMask.header.frame_id`. No K, D, width, height, or distortion checks | §8.5 |
| I10 | unknown is a legal class; it is **not** free. Validators must **not** rewrite `0 → 1` | §8.6, kill list |

No invariant requires RGB content. Tests are numeric (arrays + headers), not dummy cameras. **Every ID in this table has a corresponding test** (task T01).

Construction:

```
make_mask(..., producer_ok: bool) -> CanonicalMask
  if type(producer_ok) is not bool: raise TypeError
  if type(stamp_ns) is not int or stamp_ns <= 0: raise TypeError / ValueError
  structural I1–I7, I9–I10 fail → raise (no object)
  valid = producer_ok          # True only if we got here and producer said ok
                               # producer_ok is a real bool; do not write valid = bool(producer_ok)
```

Annotations on the dataclass are documentation. Validators enforce types. Do not use `isinstance(stamp_ns, int)` (that accepts `True`). Do not use `isinstance(producer_ok, bool)` if a numpy bool should fail — `type(...) is bool` is the rule.

`assert_canonical`, `assert_confidence`, `assert_header`, `assert_aligned`, `assert_camera_info_pair`. Raise. Do not coerce illegal pixels to `0` inside T01 — coercion is a policy of T03/T04. T01 only certifies.

---

## 5. Module layout (when coded)

```
ugv_perception/
  port/                     # T01 only
    ids.py                  # 0,1,2 + encodings
    header.py               # FrameHeader
    mask.py                 # CanonicalMask
    validate.py             # I1–I10
    port_meta.msg           # spec now; compile at T07
  tests/test_port_contract.py
```

Zero imports of Ultralytics, cv2 camera, YAML ontologies, or `rclpy` in this package slice (msg file is data, not a node).

---

## 6. Efficiency

| Choice | Why |
|---|---|
| `uint8` mask, not `int32` / `float` | three values; `mono8` is the ROS contract; 4× less bandwidth than 32-bit |
| Confidence `32FC1` for v1 | architecture requires `[0,1]` port-normalized; quantization can wait |
| `FrameHeader` shared, RGB not on the mask | T07 must not copy the camera image onto the port |
| `slots=True` dataclasses | cheap, no `__dict__` |
| Validators = C-contiguous numpy scans | ~1 ms worst case at 1280×720; negligible vs YOLOE. Keep them **on** in T07; do not make them debug-only (illegal pixel would reach Dev 3) |
| No `adapter_id` / `degraded` on meta | fewer fields, no dual-source degraded |
| Views, not copies | validators read; they do not `.copy()` unless a test needs isolation |
| CameraInfo join by `frame_id` only | do not embed or validate K/D in T01 |
| v1 `scale == 1.0` | no extra resize path in the product port |

T01 is not a hot path. Its job is to keep the hot path from shipping the wrong *kind* of array. Do not add caches, GPU, or batching here.

---

## 7. Aptness vs `architecture.md`

Scored against the original T01 brief **and** this sub-arch. Original T01 was a checklist that aliased `interfaces.md`. That is the gap this file closes.

| Architecture clause | Fit | Notes |
|---|---|---|
| §3 Port = canonical mask + conf + freshness; adapters implement the port | **High** | T01 *is* that port as types. Models stay out. |
| §8.1 three IDs, no `cautious` | **High** | I1–I2. Original T01 already had this. |
| §8.2 remap mandatory | **N/A (correct miss)** | T03. T01 must not remap. Original T01 non-goal “YAML” is right. |
| §8.3 conf in `[0,1]`, not raw multi-model | **High** | I3. T01 stores normalized conf only. |
| §8.4 stamp, frame_id, age, valid | **High** | `valid` is producer_ok ∧ structural, not “arrays looked legal.” `age_s` informational; consumers use now vs stamp. |
| §8.5 CameraInfo + resolution/scale + no stamp reuse | **High** | I9 is frame_id identity only (not calibration). v1 scale locked to 1.0. TF tree is Dev 2. |
| §8.6 degraded + unknown ≠ free | **Medium** | Original listed `degraded` on `PortMeta` (duplicates the Bool topic) and did not state “do not rewrite 0→1”. I10 + omit degraded from meta. ROI inflate remains Dev 3. |
| §9 geometry is not the port | **Fail in original / pass here** | Original T01 owned `DepthFrame`. Removed. |
| §12 perception watch | **Indirect, correct** | T01 makes `valid` + stamp checkable so Dev 5 can timeout. No watchdog in T01. |
| §16 YOLOE-as-brain, unknown-as-free, stale-as-current | **High** | Types cannot express YOLOE; I10 blocks unknown-as-free; stamp identity blocks restamp (enforced again in T05/T07). |
| DoD item 2: 3-class port + freshness/frame meta | **High** | This is the type-level half of DoD-2. |
| Kill: adapter publish without remap | **N/A** | T03/T07. T01 cannot publish. |
| Deferred `cautious` | **High** | I2 rejects id 3. |

**Overall aptness:** original T01 was **the right task, over-scoped types**. Aligned with §8.1 strongly; weakly with §8.5 CameraInfo and §9. This sub-arch brings T01 to **high** fit without stealing T02–T08.

---

## 8. Review: was T01’s architecture efficient and well done?

### What was already good

- Starting Dev 1 with an executable 3-class contract, before any model, matches architecture “port first, adapters behind it”.
- Numeric tests (not fake RGB) respect the product data policy.
- No publisher in T01 — composition stays T07.
- `uint8` + separate confidence is the right split (mask is discrete; conf is analog).
- Alignment of stamp/frame/HW is the core spatial/freshness bug class.

### What was not well done

1. **Kitchen-sink types.** Pulling `ImageFrame`, `RawSemOutput`, `DepthFrame` into T01 made the port kernel a dump of all Dev 1 structs. That is an `interfaces.md` job, not a port architecture. Cost: later tasks cannot change an adapter struct without “changing T01”. Efficiency of *design* (modularity) was poor even though runtime cost was fine.
2. **`DepthFrame` on the port task** fights §9. Geometry would look like a second canonical type from day one.
3. **`PortMeta.adapter_id`** leaks adapter identity to Dev 3/5. Architecture’s point is they must not care.
4. **`PortMeta.degraded` duplicates** `/ugv/perception_degraded`. Two sources of truth is how stale-as-current sneaks back in.
5. **CameraInfo pairing missing.** §8.5 is a v1 minimum, not deferred. T01 had frame_id but not “this frame_id has CameraInfo”.
6. **No encoding names.** `uint8` ≠ ROS `mono8` until someone forgets `bgr8`. Pin encodings in T01.
7. **`frozen` dataclasses with mutable ndarrays** look safer than they are. Needs an explicit “do not mutate after validate” rule (now in §3).

### Runtime efficiency

Original T01 was already cheap (scans, no GPU, no copies required). No performance problem. The defects are **boundary and duplication**, not FLOPs.

Do not add: object-detection proto, batched masks, GPU validators, embedding CameraInfo K/D in every mask, fourth class “for later”.

---

## 9. Done when (architecture, not just tests)

- Kernel types = `FrameHeader` + `CanonicalMask` + `PortMeta` fields above, and nothing from T02/T06/T08.
- I1–I10 each have tests (including `valid`, `age_s`, `scale`, empty `frame_id`, non-positive stamp).
- A `3` pixel cannot be represented as a *validated* mask.
- T02/T06/T08 can import T01; T01 imports none of them.
- Review in §7 stays green if `architecture.md` §8 does not change.

## 10. Non-goals

Inference, topics, YAML, `perception_max_age` comparison (T05 owns the threshold; T01 only stores `age_s`), Depth Anything, backends.
