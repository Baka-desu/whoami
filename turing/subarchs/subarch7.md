# Sub-architecture 7 — Perception port composition (T07)

**Task:** [T07](../tasks/T07-port-node.md)  
**Depends on (code/import):** T01 `make_mask`; T03 `load_remap`/`apply`; T04 `load_gates`/`apply`; T05 `evaluate`/`decide_publish`; T06 `Adapter.infer` / `AdapterError`. T07 does **not** import OpenVINO, Ultralytics, or CUDA.  
**Does not need at build for `compose_tick`:** a camera device, GPU, or ROS.  
**Status (2026-09-18):** `compose_tick`, `perception_cycle`, `wire_compose_out`, and `PerceptionAdapterNode` **shipped**. ROS 2 **Lyrical** + `rclpy` + `rmw-dds-common` on this RHEL 10 box. Node tests use a **shared executor** and fixture `Image`+`CameraInfo`. GPU YOLOE `infer` still needs IR in `weights/`. Outdoor live stream still Dev 5.  
**Authority:** [`architecture.md`](../../architecture.md) §2–§3, §8 (all), §12, §16  
**Not authority:** `dev.md` (mask+degraded only — architecture also requires confidence + meta); the obsolete T07 task loop that recomputes collapse and ignores T05 `decide_publish`; `HARDWARE.md` (T07 never picks a device)

This file is the architecture of T07. The task file is the build checklist. If they disagree, this file wins, then `architecture.md`. **T04/T05 APIs win over the old T07 pseudocode.**

---

## Files this subarch refers to

### Authority (read; do not fork)

| File | Why T07 cares |
|---|---|
| [`architecture.md`](../../architecture.md) §3, §8, §12 | port topics; stale ≠ current; degraded → Dev 5 hold |
| [`subarch1.md`](subarch1.md) | `make_mask`; `PortMeta` has no `adapter_id` / `degraded`; `age_s` informational; `scale == 1.0` |
| [`subarch3.md`](subarch3.md) | remap apply; no remap → must not publish |
| [`subarch4.md`](subarch4.md) | `apply` → `(classes, conf, known_fraction, collapse_candidate)`; T04 does not set degraded |
| [`subarch5.md`](subarch5.md) | `evaluate` + `decide_publish`; `adapter_error` bool from T07; `publish_mask is valid` |
| [`subarch6.md`](subarch6.md) | `infer` raises `AdapterError`; stamp/frame copy; no runner-up map in v1 |
| [`HARDWARE.md`](../HARDWARE.md) | T07 does not import the engine |

### Existing code T07 binds to (do not modify in T07)

| File | Binding |
|---|---|
| `turing/src/ugv_perception/port/mask.py` | `make_mask(..., producer_ok=decision.valid)` only if `publish_mask` |
| `turing/src/ugv_perception/port/port_meta.msg` | **contract** fields: header, valid, age, scale. **Not compiled** (no colcon/rosidl on this box). Node currently publishes those three numbers as `std_msgs/Float64MultiArray` on `/segmentation/port_meta` until the `.msg` is built |
| `turing/src/ugv_perception/remap/apply.py` | unpack `RawSemOutput` into `apply(...)` |
| `turing/src/ugv_perception/confidence/apply.py` | 4-tuple; `runner_up=None` in v1 |
| `turing/src/ugv_perception/freshness/evaluate.py` | `evaluate` then `decide_publish` |
| `turing/src/ugv_perception/adapter/yoloe.py` | construct at **startup** only; tick calls `infer()` |
| `turing/config/ontologies/yoloe.yaml` | must exist or node/compose refuses to start |
| `turing/config/perception/yoloe.yaml` | T04 gates |
| `turing/config/perception/port.yaml` | T05 max age |

### Shipped files

| File | Role |
|---|---|
| `turing/src/ugv_perception/compose/tick.py` | `compose_tick` — pure wiring, no `rclpy` |
| `turing/src/ugv_perception/node/cycle.py` | decode (optional) → `compose_tick` |
| `turing/src/ugv_perception/node/wire.py` | `CanonicalMask` → `sensor_msgs` Image + meta array |
| `turing/src/ugv_perception/node/adapter_node.py` | ROS 2 Lyrical node: subscribe Image+CameraInfo, publish port |
| `turing/src/ugv_perception/tests/test_compose.py` | N1–N17 + infer-`None` fail-closed |
| `turing/src/ugv_perception/tests/test_adapter_node.py` | shared `SingleThreadedExecutor` spin; `importorskip("rclpy")` |

Do **not** reimplement remap, τ, or freshness in T07.  
Do **not** put `backend:` or prompts in this module.  
Do **not** invent a dummy camera to “finish” the ROS node.

---

## When coding (after this subarch is complete)

1. Freeze this file first.  
2. Implement **`compose_tick` first** (no ROS, no GPU, no T02). T10 will reuse it.  
3. Do **not** edit `port/`, `remap/`, `confidence/`, `freshness/`, `adapter/pack.py`, or `architecture.md`.  
4. Catch `AdapterError` in the tick; set `adapter_error=True`; **do not** catch inside T05/T06.  
5. Use T04’s `collapse_candidate`. Do **not** recompute `known_fraction < min_known_fraction` in T07.  
6. Use T05 `decide_publish`. Do **not** invent a second degraded OR.  
6b. **`evaluate` before `infer`.** If `time_degraded`, do not call `adapter.infer` (or remap/gates). After a successful infer, `evaluate` again with `now_ns_after` (defaults to `now_ns` in tests).  
7. `make_mask` only when `decision.publish_mask`. Never restamp. Never pass `age_s < 0` into `make_mask`.  
8. Tick must not contain `if adapter_id == "yoloe"` or import `openvino` / `ultralytics` / `torch`.  
9. `adapter:=onnx` is illegal until T09.  
10. Tests: fixture adapter + fixture ROS msgs. Not a product camera. Node tests: **one executor**, both nodes.  
11. `infer()` returning `None` or missing `stamp_ns`/`frame_id` is `adapter_error` (no `make_mask(None)`).  
12. Node clock: `now_ns` must be Python `int > 0` — no `int()` coerce.

---

## 1. Role in the product

T07 is the **only Dev 1 process** Dev 3 and Dev 5 should care about. It wires kernels and publishes the port. It is not a model.

```
T02 decode_frame / ROS Image+CameraInfo → ImageFrame
        │
        ▼
evaluate(stamp, now_ns)                       # T05 FIRST — stamp only, no GPU
        │
        ├─ time_degraded (stale or future)
        │     decide_publish(..., collapse=False, adapter_error=False)
        │     NO infer, NO remap, NO gates
        │
        └─ is_fresh
              adapter.infer(frame)            # T06; may raise AdapterError
              remap → gates                   # T03, T04
              evaluate(stamp, now_ns_after)   # GPU time may age the stamp
              decide_publish(result, collapse, adapter_error)
        │
        ├─ always: degraded bit  → T07 publishes Bool (ROS wrapper)
        └─ if publish_mask:
              make_mask(producer_ok=valid, age_s=result.age_s,
                        stamp/frame from sensor)
```

Freshness does not need a mask. Running YOLOE on an already-stale frame is wasted Arc/NVIDIA time and fights T11 (drop oldest / one frame in flight). This is a T07 sequence rule, not a change to `architecture.md`.

Downstream of `infer()` there is no YOLOE, no prompts, no OpenVINO.

---

## 2. Scope (tight)

### T07 owns

| Piece | Why |
|---|---|
| `compose_tick` | one-frame wiring; fail closed |
| Catch `AdapterError` → `adapter_error` bool | T05 contract |
| Stamp/frame identity check frame vs `RawSemOutput` | §8.5 no silent reuse |
| Inject `now_ns` / `now_ns_after` | pre-infer stale skip; post-infer age |
| ROS wrapper (`adapter_node`) | sole publisher of port topics + `/ugv/perception_degraded` |
| Startup: load remap/gates/freshness YAML or refuse | §8.2 no remap → no publish |

### T07 does not own

| Piece | Owner |
|---|---|
| Canonical types / `make_mask` internals | T01 |
| Camera driver | Dev 5. T02 consumes Image+CameraInfo |
| LUT / ontology | T03 |
| τ / collapse math | T04 |
| max-age math / OR policy | T05 |
| YOLOE pack / prompts | T06 |
| OpenVINO GPU | T12 |
| Front ROI / costmaps | Dev 3 |
| `/cmd_vel` | Dev 5 |

---

## 3. `compose_tick` contract

```python
@dataclass(frozen=True, slots=True)
class ComposeOut:
    decision: PublishDecision     # T05
    mask: CanonicalMask | None    # None iff not publish_mask
    # no adapter_id, no prompts, no backend id

def compose_tick(
    *,
    frame: ImageFrame | None,     # None = source starve
    now_ns: int,                  # Python int > 0; clock at tick start
    adapter,                      # object with infer(frame) -> RawSemOutput
    remap_table,
    gate_profile,
    freshness_profile,
    now_ns_after: int | None = None,  # clock after infer; default now_ns
) -> ComposeOut: ...
```

### Sequence

```
adapter_error = False
collapse = False
if frame is None:
    # no stamp to evaluate honestly → degraded, no mask, no invented header
    return ComposeOut(decision=PublishDecision(valid=False, degraded=True, publish_mask=False), mask=None)

result = evaluate(freshness_profile, frame.stamp_ns, now_ns)
if result.time_degraded:
    # already stale or future: do NOT infer
    decision = decide_publish(result, False, False)
    return ComposeOut(decision=decision, mask=None)

try:
    raw = adapter.infer(frame)
except AdapterError:
    adapter_error = True
    raw = None
except Exception:
    adapter_error = True
    raw = None

if raw is not None:
    if raw.stamp_ns != frame.stamp_ns or raw.frame_id != frame.frame_id:
        adapter_error = True
        raw = None

if raw is not None:
    classes, scores = remap.apply(table, adapter_id=raw.adapter_id,
                                  label_ids=raw.label_ids, id_to_name=raw.id_to_name,
                                  raw_scores=raw.raw_scores)
    classes, conf, _frac, collapse = gates.apply(profile, adapter_id=raw.adapter_id,
                                                 classes=classes, raw_scores=scores,
                                                 runner_up=None)
else:
    classes, conf, collapse = None, None, False

now_after = now_ns if now_ns_after is None else now_ns_after
result = evaluate(freshness_profile, frame.stamp_ns, now_after)
decision = decide_publish(result, collapse, adapter_error)

if not decision.publish_mask:
    return ComposeOut(decision=decision, mask=None)

mask = make_mask(
    stamp_ns=frame.stamp_ns,
    frame_id=frame.frame_id,
    classes=classes,
    confidence=conf,
    producer_ok=decision.valid,   # True here
    age_s=result.age_s,           # >= 0 because is_fresh
    scale=1.0,
    source_hw=frame.rgb.shape[:2],
)
return ComposeOut(decision=decision, mask=mask)
```

`now_ns` is **not** written onto the mask. Mask stamp is the sensor stamp.

When `raw is None` after a **fresh** pre-check (adapter raised), still `evaluate` on the frame stamp with `now_after`. When `frame is None`, skip evaluate (no honest stamp). When pre-check `time_degraded`, never call `infer`.

---

## 4. ROS wrapper (shipped)

`PerceptionAdapterNode` (ROS 2 **Lyrical**) subscribes to Dev 5 `Image` + `CameraInfo`, runs `perception_cycle`, publishes:

| Topic | When | Invariant |
|---|---|---|
| `/ugv/perception_degraded` `std_msgs/Bool` | **every** tick | `data == decision.degraded` |
| `/segmentation/mask` `mono8` | iff `publish_mask` | pixels `{0,1,2}`; `header.stamp` = sensor; `frame_id` = optical |
| `/segmentation/confidence` `32FC1` | iff `publish_mask` | `[0,1]`; same header as mask |
| `/segmentation/port_meta` | iff `publish_mask` | **Contract:** `PortMeta.msg` (valid, age, scale). **This box:** `Float64MultiArray` `[valid, age, scale]` until colcon compiles the `.msg` |
| `/segmentation/camera_info` | iff mask published | passthrough of last `CameraInfo` |

Do **not** publish mask or meta with a new stamp when `publish_mask` is false. Bool still goes out.

Does not advertise `/cmd_vel`. Default `adapter:=yoloe`. `adapter:=onnx` illegal until T09.

---

## 5. Invariants (fail closed)

| ID | Rule |
|---|---|
| N1 | Startup without remap YAML raises / refuses (no identity remap) |
| N2 | Tick has zero `if adapter_id == ...` / no `openvino` / `ultralytics` / `torch` imports in `compose/` |
| N3 | Stamp/frame on `CanonicalMask` equal `frame` (sensor), not `now_ns` |
| N4 | `raw` stamp/frame mismatch → `adapter_error`, no mask |
| N5 | `AdapterError`, other infer exception, **`infer()` returns `None`**, or missing `stamp_ns`/`frame_id` → `adapter_error=True`, no mask, `degraded=True`. Never `make_mask(classes=None)` |
| N6 | `collapse_candidate` comes from T04 return; T07 does not recompute τ |
| N7 | `decision` comes from T05 `decide_publish` only |
| N8 | `make_mask` only if `publish_mask`; `age_s >= 0` |
| N9 | `frame is None` → degraded, no mask, no fake stamp |
| N10 | `publish_mask` ⇒ `mask is not None` and `mask.valid is True` |
| N11 | not `publish_mask` ⇒ `mask is None` (no restamp) |
| N12 | `scale == 1.0`; `source_hw` = rgb HW |
| N13 | `runner_up` is `None` (T06 v1) |
| N14 | `compose/` does not import `rclpy`. `node/adapter_node.py` may |
| N15 | No `/cmd_vel` in compose or node |
| N16 | `PortMeta` fields stay header/valid/age/scale — no `adapter_id`, no `degraded` |
| N17 | If pre-check `time_degraded`, `adapter.infer` is **not** called (no remap/gates either) |

---

## 6. Aptness vs `architecture.md`

| Clause | Fit |
|---|---|
| §3 port vs adapter vs safety | **High** — T07 is the port process; not the brain; not `/cmd_vel` |
| §8.1–8.3 3-class + conf | **High** — `make_mask` after T03/T04 |
| §8.4 freshness/valid | **High** — T05 decision; age informational |
| §8.5 stamp/frame/scale | **High** — copy sensor header; scale 1.0 |
| §8.6 degraded | **High** — Bool always; ROI is Dev 3 |
| §12 perception watch | **High** — Dev 5 reads Bool |
| §16 stale-as-current / YOLOE-as-brain | **High** — no restamp; no model in tick |

The old T07 checklist recomputed collapse and mixed T05 `degraded = not is_fresh` with a second OR. This file uses the **implemented** T04/T05 functions.

---

## 7. Tests (`test_compose.py`)

No camera. Fixture `infer` + fixture frames (T10-style). **Every N1–N16.**

| ID | Tests |
|---|---|
| N3 | mask stamp == frame stamp; ≠ injected `now_ns` |
| N4 | mismatched raw stamp → no mask, degraded |
| N5 | `AdapterError` → degraded, `mask is None` |
| N6–N7 | collapse True from T04 path → no mask |
| N8–N11 | happy path `publish_mask` + `make_mask`; stale stamp → `mask is None` |
| N9 | `frame=None` → degraded, no mask |
| N17 | stale/future stamp: fixture adapter `infer` must not run |
| N2/N14 | `compose/` sources have no `openvino` / `rclpy` |
| N1 | missing remap path → load/start error |

ROS node spin: fixture msgs + SpyAdapter + **one** `SingleThreadedExecutor`. Outdoor camera still Dev 5. YOLOE GPU `infer` still needs weights.

---

## 8. Done when

- `compose_tick` + N1–N17 pass without a camera.  
- `PerceptionAdapterNode` spin test passes on this Lyrical install (fixture topics).  
- Compiling `PortMeta.msg` is **still open** (rosidl).  
- Dev 3 can subscribe without importing `adapter/` or T12.

## 9. Non-goals

Camera driver, T12 live IR, T11 latency, costmaps, E-stop, RTAB-Map, ONNX, Depth Anything, rosidl `PortMeta`.
