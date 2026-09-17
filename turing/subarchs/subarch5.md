# Sub-architecture 5 — Freshness and degraded policy (T05)

**Task:** [T05](../tasks/T05-freshness-and-degraded.md)  
**Depends on (code/import):** T01 types only — `stamp_ns` is a Python `int > 0` ([subarch1](subarch1.md) I4). T05 does **not** import `make_mask`, remap, or confidence.  
**Does not import / need at build:** T02, T03, T04, T06, T07, GPU, camera. Tests are clocks and bools, not images.  
**Runtime contract (not an import):** T07 calls T05, then `make_mask(..., producer_ok=valid)`. T04’s `collapse_candidate` is an **argument** to T05 combine, not a T05→T04 import. **`adapter_error` is a Python `bool` supplied by T07 from the T06 adapter invocation. T05 does not catch, inspect, or import T06.**  
**Authority:** [`architecture.md`](../../architecture.md) §3.1 item 3, §8.4, §8.6, §12, §16 (stale ≠ current; perception-degraded → safety hold; unknown ≠ free)  
**Not authority:** `dev.md` hours; `HARDWARE.md` (no GPU); the numeric `0.50` s (starting YAML, not architecture)

This file is the architecture of T05. The task file is the build checklist. If they disagree, this file wins, then `architecture.md`.

---

## Files this subarch refers to

### Authority (read; do not fork)

| File | Why T05 cares |
|---|---|
| [`architecture.md`](../../architecture.md) §8.4, §8.6, §12 | freshness fields; stale → degraded; timeout table |
| [`subarch1.md`](subarch1.md) | `age_s` informational; `PortMeta` has **no** `degraded`; `producer_ok`; `stamp_ns` Python `int > 0` |
| [`subarch4.md`](subarch4.md) | `collapse_candidate`; T04 never sets `degraded`; T05/T07 do |

### Existing code T05 binds to (do not modify in T05)

| File | Binding |
|---|---|
| `turing/src/ugv_perception/port/header.py` | `FrameHeader.stamp_ns` |
| `turing/src/ugv_perception/port/mask.py` | T07 will pass `producer_ok=valid` into `make_mask` |
| `turing/src/ugv_perception/port/port_meta.msg` | `valid`, informational `age` — not a degraded field |
| `turing/src/ugv_perception/confidence/apply.py` | returns `collapse_candidate` (T07 feeds it here) |
| `turing/tasks/T07-port-node.md` | **publishes** `/ugv/perception_degraded`; T05 does not |

### Code later (T05 implementation — only after this subarch is complete)

| File | Role |
|---|---|
| `turing/src/ugv_perception/freshness/profile.py` | `FreshnessProfile` |
| `turing/src/ugv_perception/freshness/load.py` | `load_freshness(path)` |
| `turing/src/ugv_perception/freshness/evaluate.py` | `evaluate`, `combine_degraded`, `decide_publish` |
| `turing/src/ugv_perception/freshness/__init__.py` | exports |
| `turing/src/ugv_perception/tests/test_freshness.py` | F1–F14 |
| `turing/config/perception/port.yaml` | `perception_max_age` starting example |

Do **not** add ROS publishers, `rclpy`, or a node in T05.

---

## When coding (after this subarch is complete)

1. Do not start T05 code until this file is the agreed contract (same rule as T01/T03/T04).  
2. Create **only** the files in “Code later” above.  
3. Do **not** edit `port/`, `remap/`, `confidence/`, T07, or `architecture.md`.  
4. Do **not** hardcode `0.50` in `evaluate.py` — read `perception_max_age` from the profile.  
5. Do **not** import T04 or T06. `collapse_candidate` and `adapter_error` are `bool` arguments from T07.  
6. Do **not** publish `/ugv/perception_degraded` here. T07 does that from `decide_publish`.  
7. Tests: clocks and bools only. No camera, no dummy images, no OpenVINO.

---

## 1. Role in the product

Stale mask as current is on the kill list. Dev 5’s level-3 hold is `/ugv/perception_degraded` **or** invalid pose. Dev 1 owns the first bit.

T05 is the **clock + policy kernel**: is this stamp fresh, and should the stream be degraded? It does not publish. T07 is the only process that writes the Bool topic.

```
stamp_ns, now_ns, collapse_candidate, adapter_error
        │
        ▼
┌────────────── T05 ──────────────┐
│  load perception_max_age        │
│  evaluate → age_s, is_fresh     │
│  combine → degraded (policy)    │
│  decide_publish → valid,        │
│    publish_mask, degraded       │
│  never restamps                 │
│  never publishes                │
└──────────────┬──────────────────┘
               │
               ▼
        T07: make_mask(producer_ok=valid)
             publish degraded Bool always
             publish mask only if valid
```

`age_s` is informational (same law as subarch1). **Freshness decisions use current `now_ns` against `stamp_ns`**, not a queued `age_s`.

---

## 2. Scope (tight)

### T05 owns

| Piece | Why |
|---|---|
| `perception_max_age` YAML field | §8.4 consumer/producer age limit |
| `evaluate(stamp_ns, now_ns)` | clock math; future stamp is a lie |
| `combine_degraded(...)` | stream degraded policy (OR of causes) |
| `decide_publish(...)` | no restamp; mask only when valid |
| `time_degraded` vs stream `degraded` | T04 collapse is an input, not T05’s clock |

### T05 does not own

| Piece | Owner |
|---|---|
| `make_mask` / `PortMeta` fields | T01 / T07 |
| `collapse_candidate` computation | T04 |
| Adapter exceptions | T06 / T07 catch |
| ROS topics | T07 |
| Front ROI lethal | Dev 3 |
| `/cmd_vel` hold | Dev 5 |
| Camera | T02 (skipped) |

---

## 3. Types

```python
@dataclass(frozen=True, slots=True)
class FreshnessProfile:
    perception_max_age: float   # seconds, Python float, finite, > 0

@dataclass(frozen=True, slots=True)
class FreshnessResult:
    age_s: float                # (now_ns - stamp_ns) / 1e9; may be < 0 if clock lie
    is_fresh: bool              # 0 <= age_s <= perception_max_age
    time_degraded: bool         # not is_fresh

@dataclass(frozen=True, slots=True)
class PublishDecision:
    valid: bool                 # producer_ok for make_mask
    degraded: bool              # stream bit T07 publishes
    publish_mask: bool          # v1: same as valid
```

`load_freshness(path) -> FreshnessProfile`  
`evaluate(profile, stamp_ns, now_ns) -> FreshnessResult`  
`combine_degraded(time_degraded, collapse_candidate, adapter_error) -> bool`  
`decide_publish(result, collapse_candidate, adapter_error) -> PublishDecision`

`adapter_error` stays a `bool`. T07 sets it when T06 raises or returns empty. **T05 does not catch, inspect, or import T06** — no `try/except` around an adapter, no `RawSemOutput` checks. Same pattern as `collapse_candidate`: an input flag, not a dependency.

YAML (`config/perception/port.yaml`):

```yaml
perception_max_age: 0.50
```

**Do not architect `0.50`.** Architecture owns that a max age **exists** and stale → degraded. The number is starting YAML so Dev 5’s watchdog can share the same *intent*. Do not compile `0.50` into T05. Inclusive compare: `age_s <= perception_max_age`.

---

## 4. Invariants (fail closed)

Every ID has a test. `type(x) is T`.

### Load

| ID | Rule |
|---|---|
| F1 | File exists; missing YAML raises |
| F2 | `perception_max_age`: `type is float`, finite, `> 0`. `int` `1` / `bool` / `<= 0` → error |

### Evaluate

| ID | Rule |
|---|---|
| F3 | `type(stamp_ns) is int` and `stamp_ns > 0`; `type(now_ns) is int` and `now_ns > 0`. Reject `np.int64`, `1.5`, `True` |
| F4 | `age_s = (now_ns - stamp_ns) / 1e9` (Python float). Informational. Not the authority for a later consumer |
| F5 | `is_fresh = (age_s >= 0.0) and (age_s <= profile.perception_max_age)` |
| F6 | Future stamp (`age_s < 0`) → `is_fresh is False`, `time_degraded is True` |
| F7 | `time_degraded is (not is_fresh)`; `type is bool` |

### Combine + publish policy

| ID | Rule |
|---|---|
| F8 | `collapse_candidate` and `adapter_error`: `type is bool` (no truthy `1`). T05 does not produce `adapter_error`; T07 passes it |
| F9 | `degraded = time_degraded or collapse_candidate or adapter_error` |
| F10 | `valid = is_fresh and not collapse_candidate and not adapter_error` |
| F11 | `publish_mask is valid`. **v1 does not** publish an all-unknown mask on collapse. No new mask unless valid |
| F12 | Never restamp: this kernel does not take a previous mask or write `stamp_ns` |
| F13 | `decide_publish` does not call `make_mask` and has no `degraded` field on `PortMeta` |
| F14 | T05 modules do not import `rclpy`, `confidence`, `remap`, or T06 |

`age_s < 0` is a legal **evaluate** result (clock lie). T07 must **not** pass a negative `age_s` into `make_mask` (subarch1 I4). On `publish_mask is False`, skip `make_mask`.

---

## 5. Sequence (T07 wiring; T05 supplies the functions)

```
result = evaluate(profile, stamp_ns, now_ns)
decision = decide_publish(result, collapse_candidate, adapter_error)
# decision.degraded  → T07 always publishes Bool
# decision.valid     → T07 make_mask(..., producer_ok=valid) only if publish_mask
# decision.publish_mask False → no new mask, no new stamp
```

Causes of `degraded is True` (any one):

```
time_degraded          stamp too old or in the future
collapse_candidate     T04 known_fraction < min_known_fraction
adapter_error          T07-supplied bool: T06 raised / empty. T05 does not catch T06.
```

T04 still does not set `degraded`. T05 names the OR. T07 publishes it.

---

## 6. Module layout (when coded)

```
ugv_perception/
  freshness/                  # T05 only
    profile.py
    load.py
    evaluate.py
  tests/test_freshness.py
config/perception/
  port.yaml
```

T05 imports nothing from `confidence` / `remap`.  
`port` must not import `freshness`.

---

## 7. Aptness vs `architecture.md`

| Clause | Fit |
|---|---|
| §8.4 stamp, age, valid; stale ≠ current | **High** — evaluate + publish_mask only if valid |
| §8.6 gate fail or stale → degraded | **High** — combine OR; T07 topic |
| §12 perception port timeout | **High** — `perception_max_age` YAML |
| §3.1 level 3 hold | **High** — Dev 5 consumes the Bool; T05 does not command `/cmd_vel` |
| §16 stale-as-current | **High** — F11–F12 |

Original T05 returned `degraded = not is_fresh` only, and left collapse OR to T07 prose. This file makes the OR a tested function so T07 cannot invent restamp or skip collapse.

---

## 8. Tests (clocks, not images)

**Every F1–F14 has a test.**

| ID | Tests |
|---|---|
| F1 | Missing file raises |
| F2 | `perception_max_age: 1` TypeError; `0.0` ValueError |
| F3 | `stamp_ns=np.int64(...)` TypeError |
| F5 | `now - 0.1s` vs max_age `0.5` → fresh, not time_degraded |
| F5–F7 | `now - 0.6s` vs `0.5` → time_degraded |
| F6 | `stamp > now` → time_degraded |
| F9–F11 | collapse True, clocks fresh → degraded True, publish_mask False |
| F9–F11 | adapter_error True → same |
| F11 | valid True → publish_mask True |
| F12 | evaluate does not return a stamp |
| F14 | `freshness/` sources have no `import confidence` / `rclpy` |

Do not hardcode `0.50` in `evaluate.py`. Tests may use `0.5` in **fixture YAML**.

---

## 9. Done when

- `FreshnessProfile` + `evaluate` + `combine_degraded` + `decide_publish` exist.
- F1–F14 pass with no camera.
- `port.yaml` is a file; max age is not a constant in the kernel.
- T07 checklist still owns the Bool publisher.

## 10. Non-goals

ROS node, costmap ROI, OpenVINO, T02, restamping a last-good mask.
