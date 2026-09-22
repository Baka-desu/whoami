# Sub-architecture 11 — Performance / latency / backpressure

**Task:** [T11](../tasks/T11-performance-latency.md)  
**Depends on (code):** shipped T05 `evaluate` / `perception_max_age`; T07 `compose_tick` + `PerceptionAdapterNode`; T10 wire contract (do not reopen).  
**Does not need for policy tests:** camera device, outdoor RGB, T08.  
**Needs for “T11 done”:** Dev 5 `Image`+`CameraInfo` stream through T06/T07 + YOLOE-26s IR (already on disk).  
**Status:** **policy shipped.** KEEP_LAST depth 1, starve watchdog thread, `node/metrics.py`. Live p95 still skipped until Dev 5. T08 deferred.  
**Authority:** [`architecture.md`](../../architecture.md) **§8.4**, **§12** (perception watch), §16 stale-mask-as-current; [`HARDWARE.md`](../HARDWARE.md) one frame in flight  
**Not authority:** `dev.md` hours; T08; T09; fake FPS; compiled `PortMeta.msg`

This file is the architecture of T11. The task file is the build checklist. If they disagree, **this file wins**, then `architecture.md`. If this file and shipped `compose_tick` disagree on stamp/infer order, **subarch7 wins**. T10 W* remain the port contract; T11 must not weaken them.

T08 is **deferred**. T11 does not start it.

---

## Files this subarch refers to

### Authority (read; do not fork)

| File | Why T11 cares |
|---|---|
| [`architecture.md`](../../architecture.md) §8.4 | stale mask ≠ current; age = now − stamp |
| [`architecture.md`](../../architecture.md) §12 | no valid fresh mask / degraded → Dev 5 hold |
| [`HARDWARE.md`](../HARDWARE.md) | one frame in flight; latest-only queue |
| [`subarch5.md`](subarch5.md) | `perception_max_age` YAML; `now_ns` Python `int > 0` |
| [`subarch7.md`](subarch7.md) | evaluate-before-infer; no restamp; Bool every tick |
| [`subarch10.md`](subarch10.md) | consumer encodings; T11 must not break W1–W16 |

### Existing code T11 may instrument (product contract unchanged)

| File | Binding |
|---|---|
| `turing/src/ugv_perception/node/adapter_node.py` | **the** change site: QoS, starve watchdog, counters |
| `turing/src/ugv_perception/compose/tick.py` | **do not rewrite.** Stale still short-circuits before infer |
| `turing/src/ugv_perception/node/wire.py` | **do not rewrite.** Metrics must not appear on mask/conf/meta |
| `turing/config/perception/port.yaml` | `perception_max_age: 0.50` — T11 does not invent a second age |

### Code later (T11 implementation — only after this subarch is complete)

| File | Role |
|---|---|
| `turing/src/ugv_perception/node/metrics.py` | Counters + latency samples. Not a ROS msg Dev 3 subscribes |
| `turing/src/ugv_perception/node/adapter_node.py` | Edit: KEEP_LAST depth 1, starve watchdog, record metrics |
| `turing/src/ugv_perception/tests/test_latency.py` | **L1–L12** policy tests (fixture Image + slow `FixtureAdapter`) |
| `turing/src/ugv_perception/tests/test_latency_live.py` | **L-live** p95 on a real Dev 5 stream. **Skip** until that stream exists. Do not unblock with synthetic RGB claimed as outdoor |

Do **not** add `DummySource` / V4L2.  
Do **not** edit `compose/tick.py`, `port/`, `remap/`, `confidence/`, `freshness/`, `backend/`, or `architecture.md` to make T11 green.  
Do **not** put metrics on `/segmentation/*` or `/ugv/perception_degraded`.  
Do **not** start T08.  
Do **not** mark T11 **done** from fixture-only numbers.

---

## When coding (after this subarch is complete)

1. Freeze this file first.  
2. Create/edit **only** the files in “Code later”.  
3. v1 **queue_depth = 1** (`KEEP_LAST`) on **Image and CameraInfo**. `type(queue_depth) is int and queue_depth == 1` (not `isinstance` — `bool` is an `int` subclass). Depth 10 / 0 / unbounded / `True` **raise**.  
4. Infer stays on the executor thread (no second infer worker in v1). One frame in flight.  
5. Sensor `header.stamp` is the capture clock. T05, T11 latency, and the **watchdog** all use the **same** `now_ns` provider as the node (`now_ns_fn`, Python `int > 0`). Do **not** subtract `header.stamp` from `time.monotonic_ns()` (or any other unrelated clock). Never write `now` into `header.stamp`.  
6. After a slow infer, the next image processed is the **latest** kept message, not a backlog of old ones.  
7. Starve (no new image): a **watchdog that does not wait on infer** still publishes `/ugv/perception_degraded` `true` and **no** mask. A permanently hung `infer()` may wedge the **inference** path; the watchdog stays live. T11 v1 does **not** timeout/recover infer.  
8. If `now − stamp > perception_max_age` at tick time → T05 stale path (`age <= max_age` is still fresh); no “late but valid” mask.  
9. Policy tests use T10 `FixtureAdapter` (optionally slowed). That is **not** outdoor proof and **not** T06 GPU done.  
10. `test_latency_live.py` stays skipped until Dev 5 publishes a real stream. No fake FPS.

---

## 1. Role in the product

A correct mask that is late is a **stale** mask. Architecture §8.4 / §12: Dev 5 holds. T11 is whether capture → compose → publish stays inside `perception_max_age`, and whether overload **drops old frames** instead of restamping them.

```
Dev 5 Image      ──► KEEP_LAST depth 1 ──► last accepted image stamp
Dev 5 CameraInfo ──► KEEP_LAST depth 1 ──► last accepted calibration
                      │                    │
                      │ drop oldest        ▼
                      │              compose_tick (T07, unchanged)
                      │                    │
                      │                    ▼
                      │              wire (T10 contract unchanged)
                      │                    │
                      └── counters ──► node.metrics (not Dev 3)
```

T11 does **not** change `{0,1,2}`, encodings, or stamp identity. It changes **which frame is inferred** under load and **whether degraded still goes out** when the camera stops.

---

## 2. Scope (tight)

### T11 owns

| Piece | Why |
|---|---|
| Latest-only subscription | HARDWARE one frame in flight; drop oldest |
| Starve watchdog | §12 perception watch needs a Bool when images stop |
| Counters / latency samples | eval; not a port field |
| Policy tests (slow adapter + burst) | prove skip-ahead stamps, no restamp |
| Live p95 gate | **blocked** on Dev 5 stream |

### T11 does not own

| Piece | Owner |
|---|---|
| `compose_tick` / remap / gates | T07 / T03 / T04 |
| Port encodings / `{0,1,2}` | T10 / T01 |
| Camera driver | Dev 5 |
| YOLOE IR `run()` | T12 (already on disk) |
| Outdoor quality | T06 product infer |
| Depth / VoxelLayer | **T08 — deferred** |
| `/cmd_vel` | Dev 5 |
| A second GPU worker thread | not v1 |

---

## 3. Backpressure (required)

Shipped node: `create_subscription(..., 10)` then infer **inside** `_on_image`. While infer runs, the executor cannot take other callbacks; RMW can queue up to 10 images; after infer returns those **old** frames are processed in order. That is unbounded-enough to publish stale stamps. Illegal under T11 / HARDWARE.

**v1 policy:**

| Knob | Value |
|---|---|
| History | `KEEP_LAST` |
| Depth | **1** on **both** Image and CameraInfo |
| Reliability | **unchanged** (do not silently switch BEST_EFFORT) |
| Durability | **unchanged**. T11 sets history depth only. For static/latched CameraInfo, do **not** drop TRANSIENT_LOCAL (or the product’s existing durability) — a late subscriber must still get the last calibration |
| Infer | one in flight, on the image callback / executor. **Not** a second YOLOE worker |

After a slow infer, at most **one** pending Image remains (the latest). Next `_on_image` is that latest. Its **original** stamp goes on the mask (T07). Dropped images are not published and not restamped.

**CameraInfo (T02 pairing):** CameraInfo is **calibration state**, not temporally synchronized per-frame data. Pair the Image with the **latest valid** CameraInfo (`frame_id` + H×W; T02 **ignores** CameraInfo stamp). Do not invent stamp-sync of Image T10 with CameraInfo T10. Independent `KEEP_LAST` depth 1 on both topics is enough because pairing is not by capture time.

Must not keep a depth-10 CameraInfo backlog. If CameraInfo is static / latched, consume the latest valid calibration; do **not** queue historical samples. Decode still refuses mismatched `frame_id` / H×W (T02). Do not invent a second CameraInfo of a different `frame_id`. Depth change must **not** break latched delivery: keep the durability the rest of the stack already requires.

Per-frame CameraInfo stamp-sync is **out of T11**. If a later product needs it, that is a T02 sync rule, not “take latest independently.”

**`queue_depth`:** node parameter. Require `type(queue_depth) is int and queue_depth == 1` at init. `True`, `1.0`, `0`, `10` **raise** (do not silently keep 10). Do **not** add this field to T05 `port.yaml` (T05 owns only `perception_max_age`).

**Drop count:** RMW `KEEP_LAST` discards are not delivered, so the node cannot count them from callbacks.  

- `frames_in` = Image callbacks the node actually ran  
- `frames_dropped` is **not** a precise RMW counter in v1  
- **transport skips** = camera publishes − `frames_in` (callbacks not delivered)  
- **inference skips** = `frames_in` − `infer_calls` (accepted but not inferred: stale/short-circuit)  
These are different. Do not call both “drop rate.”  
- Policy tests: helper publishes N images while adapter sleeps; `adapter.calls < N`; published stamps **skip ahead**

Do not add a second infer thread just to count RMW drops.

---

## 4. Starve watchdog (required)

Today the node only `_tick()`s on image/info. If Dev 5 stops, **no Bool is published** and the last `degraded=false` can sit forever. §12 cannot trip.

A ROS timer on the **same** `SingleThreadedExecutor` as `_on_image` **cannot** fire while `adapter.infer()` is blocked. Hung/slow YOLOE would then also stall the starve Bool. That is a real T11 hole.

**v1:** watchdog **must not wait on infer**.

| | Infer path | Watchdog |
|---|---|---|
| Where | image callback (one in flight) | **not** that callback; not the infer lock |
| May call `adapter.infer` | yes, at most one | **never** |
| Period | — | `perception_max_age / 2` (**derived**, not a second τ) |

Period `max_age/2` is sampling frequency so a death at `t=0.01` is seen by ~`0.50s`, not ~`1.0s`. Freshness bar stays T05: `age_s <= perception_max_age`.

**Clock:** watchdog `now − last_image_stamp` uses the **same `now_ns` function and semantics as T05** (the node’s `now_ns_fn`). It must not use an unrelated Python monotonic clock when `header.stamp` is ROS/system time. One provider: T05 `evaluate`, T11 latency samples, and the watchdog. If Dev 5’s stamp domain ≠ that `now_ns`, record honesty (may look stale); do not coerce.

**`last_image_stamp`:** the stamp of the last Image callback the node **successfully accepted/observed** (`frames_in` increment). It is **not** “latest message sitting in RMW” and **not** “stamp of the infer that just finished.”

So this is legal and intended:

```
image callback accepts stamp T0 → starts infer()
watchdog sees T0 age > max_age → publishes degraded=true  (infer may still be running)
```

Dev 5 may hold while a late infer is still on the GPU. When that infer returns, T07 still refuses to publish a mask if `now − T0 > max_age` (no restamp).

Implementation (pick one; do not do both):

1. **Preferred:** a small daemon thread that only reads `last_image_stamp` (lock), calls the **same** `now_ns_fn` as `_tick`, and publishes Bool + no mask if missing/stale. Never calls infer. Image/infer tests keep `SingleThreadedExecutor`. Shutdown: signal stop → thread join/exit → then destroy publishers/node. The watchdog must not touch ROS after context teardown.  
2. ROS timer on a **separate** callback group **only if** the process uses `MultiThreadedExecutor`. Do **not** switch T07/T10 tests to MT for the whole node just to get a timer.

Watchdog actions:

- no image yet, or `now − stamp > perception_max_age` → publish degraded `true`, **no** mask, **no** infer  
- last stamp still fresh → no extra infer, no restamp  

Do not restamp a stored mask. Do not call `compose_tick` from the watchdog if that can enter `infer` (stale T07 path skips infer — still do not share the adapter with a second thread; publish Bool from the watchdog without re-entering the adapter).

---

## 5. Metrics (eval only)

`node/metrics.py` is in-process. Optional `get_logger().debug`. **Not** a topic Dev 3/5 must subscribe. **Not** fields on `PortMeta` / the stopgap array.

| Field | Meaning |
|---|---|
| `frames_in` | Image callbacks delivered |
| `masks_published` | times `wired.mask is not None` |
| `infer_calls` | times `adapter.infer` ran (from adapter spy in tests; or increment around infer if the node can see it without wrapping T06) |
| `degraded_true` / `degraded_false` | Bool publishes |
| `latencies_ns` | for each **published** mask: `now_ns_fn() − stamp_ns` (same clock as T05 **and** the watchdog) |
| `t_infer_ns` | optional; adapter-only wall if measured around `perception_cycle` |

`header.stamp` stays sensor time. If Dev 5’s stamp domain ≠ node `now_ns`, latency may look stale — **record honestly**. Do not coerce clocks.

**p95 vs T05:** T05 treats `age_s <= perception_max_age` as fresh. T11 live gate is **`p95(latencies_ns)/1e9 < perception_max_age`** — a **performance margin**, stricter than the freshness contract. Exact `0.50s` is still a legal published mask; it fails the T11 speed bar. Do not change T05 to match `<`.

p95 on published masks **alone** can hide overload (20 fast publishes, 80 dropped). Live eval **must** report together:

```
capture rate, frames_in, infer_calls, masks_published,
effective infer rate, transport skips, inference skips,
p95 published latency, degraded true/false counts
```

Do not fake FPS. Fixture-only p95 is not T11 done.

---

## 6. Invariants (L1–L12)

| ID | Rule |
|---|---|
| L1 | Image **and** CameraInfo subscriptions: KEEP_LAST, depth **1** |
| L2 | `type(queue_depth) is int and queue_depth == 1`; anything else at init **raises** |
| L3 | Metrics / logs never write `mask.header.stamp` |
| L4 | Slow adapter + burst of fixture Images → published stamps **skip ahead** (not 1:1 with publishes) |
| L5 | Those published stamps are **original** sensor stamps (no restamp, no duplicates with new headers) |
| L6 | `now − stamp > perception_max_age` at tick → degraded, no mask (T05: `<=` is still fresh) |
| L7 | Starve: watchdog uses `last_image_stamp` (last accepted callback, not RMW, not infer-complete) and the **same `now_ns_fn` as T05**; period `max_age/2`; Bool `true` and no mask **even while infer is blocked**. Watchdog stays live. A permanently hung `infer()` may wedge the inference path; **no** infer timeout/recovery in T11 v1 |
| L8 | T10 W1–W16 still hold (encodings, `{0,1,2}`, fail-closed tuple) |
| L9 | `/cmd_vel` still not advertised |
| L10 | No `DummySource`; fixtures not registered as `live_cam` |
| L11 | `test_latency_live.py` skipped until a Dev 5 stream exists; skip reason says so |
| L12 | T08 / depth topics not required |

ROS extras: image/infer tests may keep **one** `SingleThreadedExecutor` (T07/T10). L7 must still prove the watchdog Bool while a slow `FixtureAdapter.infer` is in progress — that is why the watchdog is not on that executor. `importorskip` rclpy if needed.

---

## 7. Mapping to the T11 checklist

| Task item | T11 id |
|---|---|
| Drop oldest, never unbounded | L1, L4 |
| Latest inferred; original stamp | L5 |
| Stale at publish → T05 | L6 |
| Latest-only + slowed infer | L4, L5 |
| Starve → degraded; watchdog live if infer blocked | L7 |
| Metrics do not rewrite stamp | L3 |
| Counters without camera | implement; **not** “T11 done” |
| p95 on real source | L-live / L11 |

---

## 8. Aptness vs `architecture.md`

| Clause | Fit |
|---|---|
| §8.4 freshness | **High** — overload must not turn old frames into current masks |
| §12 perception watch | **High** — starve still emits degraded |
| §16 stale-mask-as-current | **High** |
| §8.1–8.3 port classes | **Unchanged** (T10) |
| §9 / T08 | **Out** |
| §3 `/cmd_vel` | **Out** — Dev 5 |

---

## 9. Done when

**Policy (can ship without Dev 5):** L1–L12 except L-live; queue depth 1; starve watchdog; T10 still green.

**T11 task “done” (do not claim without this):** numbers from a real Dev 5 `Image`+`CameraInfo` through T06/T07; the live eval table above; p95 published latency **`< perception_max_age`** (speed bar, stricter than T05 `<=`) **or** written honesty that this hardware cannot and degraded is the outcome.

Fixture-only FPS is **not** done.

## 10. Non-goals

T08, stereo fusion, YOLOE-R, tutorial ONNX, TensorRT mandate, costmap timing, training, DummySource, second infer thread, infer timeout/recovery, new Dev 3 metrics topic, changing `perception_max_age`.
