# Sub-architecture 10 — Port contract tests (T01–T07 wired)

**Task:** [T10](../tasks/T10-contract-tests.md)  
**Depends on (code):** shipped T01 `make_mask`; T03 remap YAML; T04 gates; T05 `decide_publish`; T06 `Adapter.infer` **Protocol** (fixture only); T07 `compose_tick` + `wire_compose_out`.  
**Does not need:** camera device, YOLOE IR / `.pt`, OpenVINO, T08, T09, T11, T12 live `run()`.  
**Status:** **shipped.** `tests/fixtures/` + `test_port_wired.py` (W1–W16, named W9 cases) + optional `test_port_wired_ros.py`. T07 kernel tests stay. T08 still deferred.  
**Authority:** [`architecture.md`](../../architecture.md) §3, **§8.1–§8.6**, §12 perception watch, §16 kill list  
**Not authority:** `dev.md` hours; T08 geometry; T11 FPS; outdoor YOLOE quality; compiled `PortMeta.msg` (still a Float64MultiArray stopgap)

This file is the architecture of T10. The task file is the build checklist. If they disagree, **this file wins**, then `architecture.md`. If this file and shipped `compose_tick` disagree on tick order, **subarch7 wins**.

T08 is **deferred** (optional geometry). T10 does not start it.

---

## Files this subarch refers to

### Authority (read; do not fork)

| File | Why T10 cares |
|---|---|
| [`architecture.md`](../../architecture.md) §8 | 3-class port + conf + freshness + spatial min + fail-safe |
| [`00-role-and-laws.md`](../00-role-and-laws.md) | topics, encodings, Dev 3 vs Dev 5, no `/cmd_vel` |
| [`subarch1.md`](subarch1.md) | `{0,1,2}`, `mono8` / `32FC1`, `valid`, informational `age`, `scale==1.0`, `PortMeta` has no `adapter_id` / `degraded` |
| [`subarch3.md`](subarch3.md) | no remap → must not publish |
| [`subarch4.md`](subarch4.md) | collapse is T04; all-below-τ is not a valid all-traversable mask |
| [`subarch5.md`](subarch5.md) | stale → `degraded`, `publish_mask is valid` |
| [`subarch7.md`](subarch7.md) | `compose_tick` evaluate-before-infer; wire topics; no restamp |

### Existing code T10 binds to (do not modify in T10)

| File | Binding |
|---|---|
| `turing/src/ugv_perception/compose/tick.py` | **the** in-process port. T10 does not rewrite it |
| `turing/src/ugv_perception/node/wire.py` | `CanonicalMask` → `sensor_msgs` encodings + meta array |
| `turing/src/ugv_perception/node/cycle.py` | decode + CameraInfo fail → `frame is None` |
| `turing/src/ugv_perception/node/adapter_node.py` | optional ROS proof; same topics |
| `turing/src/ugv_perception/port/ids.py` | `MASK_ENCODING`, `CONF_ENCODING`, `CANONICAL` |
| `turing/src/ugv_perception/port/mask.py` | `make_mask` raises on illegal pixels (abort) |
| `turing/config/ontologies/yoloe.yaml` | product remap (dirt_path→1, person→2, sky→0, …) |
| `turing/config/perception/yoloe.yaml` | product τ (collapse case) |
| `turing/config/perception/port.yaml` | `perception_max_age: 0.50` |

T07 tests stay. T10 does **not** delete or re-home `test_compose.py`.

### Code later (T10 implementation — only after this subarch is complete)

| File | Role |
|---|---|
| `turing/src/ugv_perception/tests/fixtures/__init__.py` | test-only package |
| `turing/src/ugv_perception/tests/fixtures/source.py` | `FixtureSource` — allocated RGB HW, caller `stamp_ns` / `frame_id`. **Not** a product `Source` |
| `turing/src/ugv_perception/tests/fixtures/adapter.py` | `FixtureAdapter` — scripted `label_ids` / `raw_scores`, **copies** stamp/frame from the frame |
| `turing/src/ugv_perception/tests/test_port_wired.py` | **Mandatory.** W1–W16 in-process: `compose_tick` + `wire_compose_out`. Pure Python; no ROS. A skip here is a T10 fail. |
| `turing/src/ugv_perception/tests/test_port_wired_ros.py` | **Optional.** W-R* Lyrical spin; `importorskip` rclpy/sensor_msgs. A skip here is “no ROS in env”, **not** a T10 fail. |

Do **not** add `DummySource` / `LiveCameraSource` / V4L2.  
Do **not** put fixtures under `adapter/` or `ingest/`.  
Do **not** register them in a `live_cam` launch.  
Do **not** edit `compose/`, `port/`, `remap/`, `confidence/`, `freshness/`, `backend/`, or `architecture.md` to make T10 green.  
Do **not** use these fixtures to mark T02 / T06 / T11 / T08 done.

---

## When coding (after this subarch is complete)

1. Freeze this file first.  
2. Create **only** the files in “Code later”.  
3. Drive **shipped** `compose_tick` then `wire_compose_out`. No second `to_canonical`.  
4. RGB buffers in fixtures exist so dtype/shape match. They are **not** a camera and **not** outdoor proof.  
5. `pytest` for T10 must be green with **no camera and no `.pt` / IR**. `test_port_wired.py` is mandatory (do not skip W1–W16). `test_port_wired_ros.py` may `importorskip` if ROS packages are missing — that skip is not a T10 fail.  
5b. **W9 is one invariant.** Implement it as separately named (or parameterized) tests so a failure names the input, e.g. `test_wire_adapter_exception_degrades`, `test_wire_adapter_none_degrades`, `test_wire_adapter_missing_stamp_degrades`, `test_wire_adapter_stamp_mismatch_degrades`, `test_wire_adapter_missing_frame_degrades`, `test_wire_adapter_frame_mismatch_degrades`. Do not split W9 into six architecture IDs.  
6. Sequential starve: tick A publishes; tick B with `now_ns - stamp_ns` `> perception_max_age` must **not** re-emit A’s mask with a new header. Freshness is **always** `now − sensor stamp`. Do not inject `age_s`.  
7. If `wire.mask is not None`, pixels ∈ `{0,1,2}`, encoding `mono8`, stamp = source, `frame_id` = source.  
8. Published payloads must not contain YOLOE prompt strings, `openvino_gpu`, or `adapter_id`.  
9. Illegal class `3`: `make_mask` still **raises** (abort). Do not add a remapper-injection API to T07. Product remap YAML cannot load a `3` (T03). T10 proves the property on every published mask plus the T01 raise.  
10. T08 / T11 / outdoor `infer` stay out.

---

## 1. Role in the product

Architecture §8 is the only semantic contract Dev 3 (costmaps) and Dev 5 (degraded hold) are allowed to see. T01–T07 built the kernels and the node. T10 is the **proof that the wired port matches §8** without a live camera and without YOLOE.

```
FixtureSource header + allocated HW
        │
        ▼
FixtureAdapter.infer  →  RawSemOutput (copied stamp/frame, scripted labels)
        │
        ▼
T07 compose_tick      →  CanonicalMask | None + PublishDecision
        │
        ▼
T07 wire_compose_out  →  mask / conf / meta / degraded   (what Dev 3/5 subscribe)
        │
        ▼
T10 asserts §8 on those bytes
```

T10 is **not** a product source. `profile:=live_cam` must not be able to select `FixtureSource`.

T07 `test_compose.py` proves **composer internals** (evaluate-before-infer, no engine imports). T10 proves **consumer observables** (encodings, identity, starve, no model leak). Overlap on stamp/stale is allowed; T10 still owns the named suite.

---

## 2. Scope (tight)

### T10 owns

| Piece | Why |
|---|---|
| Named wired contract suite | Unblocks “port done” for Dev 3/5 integration |
| Test-only `FixtureSource` / `FixtureAdapter` | T10 task; not `live_cam` |
| Wire encodings `mono8` / `32FC1` | §8 + T01 constants, on **published** msgs (or `wire_compose_out`) |
| Sequential no-restamp | kill list: stale-mask-as-current |
| No model leak on the port | kill list: YOLOE-as-brain |
| Mixed `{0,1,2}` after product remap | §8.1 through T03, not T01 alone |

### T10 does not own

| Piece | Owner |
|---|---|
| `compose_tick` implementation | T07 (shipped) |
| Camera driver / V4L2 | Dev 5 / not us |
| YOLOE IR `run()` | T12 / T06 |
| Outdoor quality | T06 product infer (still needs Dev 5 stream) |
| Depth / VoxelLayer | **T08 — deferred** |
| FPS / queue | T11 |
| Tutorial ONNX swap | T09 |
| Compiling `PortMeta.msg` | still open; T10 asserts the stopgap array |
| Costmap inflation / `/cmd_vel` | Dev 3 / Dev 5 |

---

## 3. What “wired” means

**Preferred (required):** in-process, no ROS daemon.

```
compose_tick(frame, now_ns, FixtureAdapter, product YAML kernels)
    → ComposeOut
wire_compose_out(out)
    → WireOut { degraded: Bool, mask: Image|None, confidence: Image|None, port_meta: Float64MultiArray|None }
```

`WireOut` **is** the port as Dev 3 would see it, minus DDS.

**Optional (this box has Lyrical):** `PerceptionAdapterNode` + helper pubs + **one** `SingleThreadedExecutor` (same rule as T07). Assert the same fields on `/segmentation/*` and `/ugv/perception_degraded`. `importorskip` if ROS libs missing.

Do not stand up a second node implementation.

---

## 4. Architecture §8 → T10 checks

| Clause | Pass when |
|---|---|
| **§8.1** 3-class v1, no `cautious` | every published mask pixel ∈ `{0,1,2}`; mixed fixture (path / sky / person) remaps to `{1,0,2}` |
| **§8.2** no remap → no publish | missing ontology path refuses to load (reuse T07 N1 at T10; do not invent identity remap) |
| **§8.3** conf in `[0,1]` | confidence Image `32FC1`, finite, `[0,1]`, same H×W and header as mask |
| **§8.4** stamp / frame / age / valid | `mask.header.stamp` = source image time, **not** `now`; `frame_id` = optical; age informational; `valid` false when degraded |
| **§8.4** stale ≠ current | `now − stamp > perception_max_age` (0.50 s in product YAML) → `degraded=true`, `publish_mask=false`, **no new mask msg** |
| **§8.5** spatial min | `frame_id` copied; v1 `scale==1.0`; ROS: `/segmentation/camera_info.frame_id` equals mask (T02 pairing). Projection math **deferred** (§14) — T10 does not invent TF |
| **§8.6** fail-safe | starve / adapter exception / collapse → `/ugv/perception_degraded` true. T10 does **not** inflate a front ROI (Dev 3) |
| **§12** perception watch | Bool is honest; T10 does not implement the mux |
| **§16** unknown-as-free | collapse (all scores below `tau_min`) is **not** a valid all-`1` mask |
| **§16** YOLOE-as-brain | wire payloads have no prompt names, no backend id |
| **§3** not `/cmd_vel` | T10 suite does not publish it; node sources still must not contain it |

T08 / §9 geometry: **out of T10**. Missing depth is legal. T10 must not require a cloud topic.

---

## 5. Fixtures (test-only)

### `FixtureSource`

- Yields `ImageFrame(rgb=uint8 HWC allocated, stamp_ns, frame_id)`.  
- Caller sets stamp/frame. RGB is zeros or a constant buffer — **not** claimed as outdoor, not a Dev 5 stream.  
- Not registered for `live_cam` / `bag` / `rugd`.  
- Does not implement a production `Source.open/read/close` used by bringup.

### `FixtureAdapter`

- `infer(frame) -> RawSemOutput`.  
- **Must copy** `frame.stamp_ns` and `frame.frame_id` (same law as T06).  
- `label_ids` / `raw_scores` / `id_to_name` are **tables** the test sets (prompt ids that exist in `yoloe.yaml`, e.g. `1=dirt_path`, `4=sky`, `5=person`).  
- May raise `AdapterError` on command.  
- Must not import OpenVINO / Ultralytics / `backend.factory`.

These are allowed numeric tables (DATASETS: contract tests, not perception data). They must not green T06 GPU infer or T02 camera ingest.

---

## 6. Invariants (W1–W16)

| ID | Rule |
|---|---|
| W1 | Published mask encoding is `mono8` (`MASK_ENCODING`); numpy classes `uint8` |
| W2 | Every pixel ∈ `{0,1,2}`; mixed labels after **product** remap are only those ids |
| W3 | `mask.header.stamp` == source `stamp_ns`; publish at `T+10ms` still `T` (not `now`) |
| W4 | `mask.header.frame_id` == source (`"camera_optical"` in the fixture) |
| W5 | Confidence: `32FC1`, same stamp/frame/H×W as mask, values finite and in `[0,1]` |
| W6 | `port_meta` stopgap is length-3 `[valid, age, scale]` (**temporary** until `PortMeta.msg` is compiled; Dev 3/5 must not treat these indices as the permanent ROS API). `valid` is **this published mask’s** semantic validity (`producer_ok` ∧ T01), **not** “the system is healthy”. Stream health is `/ugv/perception_degraded` only. Because T07 only wires meta when a mask is published, and that mask exists only when `valid` is true, the stopgap `valid` field is `1.0` on every published meta. `scale==1.0`. No `adapter_id`. |
| W7 | Set `now_ns - source stamp_ns = 0.6 s` against product `perception_max_age=0.50`. Do **not** pass `age_s` in. Freshness is `now − sensor stamp` (T05). Fail-closed `WireOut`: `degraded=True`, `mask is None`, `confidence is None`, `port_meta is None` |
| W8 | After a **good** publish, a later stale tick does **not** re-emit that mask with a new header (no restamp) |
| W9 | Adapter fail on the wire (`AdapterError`, `infer() is None`, missing stamp/frame, or ≠ source) → **one** consumer invariant: `degraded=True`, `mask is None`, `confidence is None`, `port_meta is None`. Same shipped T07 N4/N5 path. T10 does not re-fix T07. Name the pytest cases per input (when-coding 5b). |
| W10 | All scores below product `tau_min` → published mask is all `0` (not all `1`), `degraded=false` if the frame is fresh. Collapse does not drop the mask |
| W11 | Missing **product remap configuration** must prevent the wired pipeline from producing a mask. T10 does not define T03 internals (illegal map keys, unmapped name → `0` are T03). |
| W12 | Wire bytes / `Float64MultiArray` labels do not contain YOLOE prompt strings, `"openvino_gpu"`, or `"adapter_id"` |
| W13 | `make_mask(..., classes` with a `3`) still raises; T10 never sees a published pixel `3` |
| W14 | Degraded Bool is published even when mask is not |
| W15 | Fixtures live under `tests/`; production `ingest/` / `adapter/` have no `FixtureSource` / `DummySource` |
| W16 | T10 files import neither `openvino` nor T08 depth types; no `/cmd_vel` |

ROS extras (optional file):

| ID | Rule |
|---|---|
| W-R1 | `/segmentation/mask` encoding `mono8`; stamp/frame as W3/W4 |
| W-R2 | `/segmentation/confidence` present iff mask; `32FC1` |
| W-R3 | `/ugv/perception_degraded` received; stale/fixture-fail → `true` |
| W-R4 | `/segmentation/camera_info.header.frame_id` == mask `frame_id` when mask published |
| W-R5 | Shared `SingleThreadedExecutor`; do not dual-`spin_once` |

---

## 7. Mapping to the T10 checklist (minimum cases)

| Task case | T10 id |
|---|---|
| 1. Mixed `{0,1,2}` after remap | W2 |
| 2. Sensor stamp `T`, publish at `T+10ms` | W3 |
| 3. `frame_id="camera_optical"` | W4 |
| 4. `now − stamp = 0.6 s` / max 0.5 s, no restamp | W7 + W8 |
| 5. Adapter exception | W9 |
| 6. Illegal pixel `3` aborted | W13 |

T07 N3/N5/N11 already touch stamp/error/stale. T10 still implements W* on **`wire_compose_out`**, including encodings and sequential restamp, which N* do not fully name.

No-mask ticks (W7, W9, W10, starve) always mean **confidence and port_meta are also absent**. Confidence is present iff mask is present. Do not publish an empty/stale conf without a mask.

### Already shipped — do not reopen as T10 work

These were T07/T05 bugs. They are **fixed and tested**. T10 does not add a second kernel suite for them. W9 only asserts the **same consumer `WireOut`** (`degraded=True`, mask/conf/meta all `None`). Named pytest functions diagnose which adapter input caused it; they are not extra architecture IDs.

| Failure | Where it already lives |
|---|---|
| `infer()` returns `None` → no `make_mask(None)` | `compose_tick` + `test_infer_none_is_adapter_error_no_crash` |
| missing `stamp_ns` / identity ≠ source | `compose_tick` + `test_infer_missing_stamp_is_adapter_error`, N4 |
| `now_ns` must be Python `int > 0`, no `int()` coerce | T05 `evaluate` `_require_stamp`; node `_tick` TypeError; T05 F3 |
| unmapped prompt name → class `0` | T03 R12. T10 W2 is mixed **mapped** `{0,1,2}`; W11 is missing YAML |
| missing `frame_id` | same identity branch as missing stamp (T07) |

Clock type is **not** a Dev 3 observable. Do not add W22–W26 on T10.

---

## 8. Aptness vs `architecture.md`

| Clause | Fit |
|---|---|
| §3 port vs adapter vs safety | **High** — fixtures never become the brain; no `/cmd_vel` |
| §8.1–8.3 | **High** — 3-class + remap + conf domain on the wire |
| §8.4 | **High** — stamp/age/valid/degraded |
| §8.5 | **Medium-high** — frame_id + scale 1.0 + CameraInfo pairing on ROS; no TF/projection (deferred §14) |
| §8.6 | **High** — Bool honest; ROI not ours |
| §9 / T08 | **Out** — deferred |
| §12 | **High** — perception watch input is the Bool |
| §13 DoD item 2 | **High** — this is the test evidence for “3-class port + freshness/frame meta” without claiming live_cam E2E (DoD 1, 8) |
| §16 | **High** — stale-as-current, unknown-as-free, YOLOE-as-brain |

---

## 9. Done when

- `pytest` on `test_port_wired.py` is green with **no camera, no `.pt`, no IR**.  
- W1–W16 all exist (not a subset). **W9 is still one ID**; the named/parameterized pytest cases in §5b must all exist — do not implement W9 as a single `AdapterError`-only test.  
- `test_port_wired_ros.py` green on this Lyrical box, or skipped only for missing ROS packages (**not** a T10 fail).  
- Production tree still has no `DummySource`.  
- T08 still not started.

## 10. Non-goals

FPS, OpenVINO GPU, outdoor quality, Depth Anything, stereo fusion, YOLOE-R, tutorial ONNX, costmaps, E-stop, RTAB-Map, compiling `PortMeta.msg`, TF tree, dummy cameras, training.
