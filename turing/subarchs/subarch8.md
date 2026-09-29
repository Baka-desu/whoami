# Sub-architecture 8 — Depth side-channel

**Task:** T08 optional geometry. Not a second Perception Port.  
**Depends on (code):** shipped T02 `Image`+`CameraInfo`; T07 one accepted frame (stamp + optical `frame_id`); T11 one frame in flight; T12 OpenVINO `device=GPU`.  
**Does not change:** `{0,1,2}` mask, τ, remap, `/segmentation/mask`, `/ugv/perception_degraded`.  
**Status:** **frozen.** Implementation started. Weights are not in git. The live adapter stays RUGD SegFormer-B5. Depth runs only when its OpenVINO IR is on disk.  
**Authority:** [`architecture.md`](../../architecture.md) **§9** (qualified occupied geometry stays lethal), §8.4 (same stamp; missing or stale geometry is not free space); [`HARDWARE.md`](../HARDWARE.md) one frame in flight.  
**Not authority:** `dev.md` hours; stereo disparity; T09; a depth value written into the class mask; the safetensors file size as a GPU-memory number.

This file wins over a T08 checklist. `architecture.md` wins over this file.

OpenVINO compile is a startup action. It is not a per-frame action.

T08 does **not** delay publication of the current frame's mask. It **does** occupy the image callback after that publish, so it can increase transport skips and the capture-to-mask age of a **later** frame. v1 accepts that (choice A). T11's live check runs with T08 enabled. A separate depth worker is not v1.

---

## Pin

| | |
|---|---|
| Model | **Depth Anything 3 Metric Large** |
| Repo | `depth-anything/DA3METRIC-LARGE` |
| License | Apache 2.0 |
| Checkpoint | 0.35B parameters, about 1.3 GB safetensors. That size is not GPU peak. |
| What it is | One RGB image → canonical depth + a sky score tensor |
| Meters | `depth_m = focal_model * raw / 300` at model resolution |
| `focal_model` | `(fx_model + fy_model) / 2` from `K_model` |
| `K_camera` | Original `CameraInfo` K. Back-projection only, after depth is on camera `H×W`. |
| Runtime | OpenVINO IR, `device=GPU`, no CPU compile. PyTorch loads the safetensors only to export the IR. |
| Not this pin | V1. V2 indoor metric. V2 relative. DA3-LARGE any-view (CC BY-NC). DA3 Giant. |

`depth.yaml` records `process_res: 504`, `resize_mode: upper_bound_resize`, `patch: 14`, `sky_threshold: 0.3`, `valid_coverage: 0.5`, `metric_scale: 300`, and the measured GPU peak.

---

## Files

### Authority (read; do not fork)

| File | Why T08 cares |
|---|---|
| [`architecture.md`](../../architecture.md) §9 | Lethal geometry is not cleared by class 1 |
| [`architecture.md`](../../architecture.md) §8.4 | Same image stamp. Missing or stale geometry is not free space |
| [`HARDWARE.md`](../HARDWARE.md) | One frame in flight. SegFormer-B5 already ~856 MiB before depth |
| [`subarch7.md`](subarch7.md) | The mask tick stays evaluate-before-infer. Depth does not reopen it |
| [`subarch11.md`](subarch11.md) | `KEEP_LAST` 1. Depth after publish still holds the callback |

### Existing code T08 must not rewrite

| File | Binding |
|---|---|
| `turing/src/ugv_perception/compose/tick.py` | Mask path. Depth failure does not set `adapter_error` |
| `turing/src/ugv_perception/adapter/rugd.py` | Live segmenter. No depth head inside it |
| `turing/config/perception/rugd.yaml` | τ stays as written |
| `turing/src/ugv_perception/node/wire.py` | Mask, confidence, and port meta stay the segmenter's |

### Code later (only after this subarch is frozen)

| File | Role |
|---|---|
| `turing/src/ugv_perception/depth/` | Preprocess, `K_model`, meters, hole-safe resize, back-project |
| `turing/src/ugv_perception/backend/` | Compile the depth IR once at startup. Raw tensors out. No `{0,1,2}` |
| `turing/config/perception/depth.yaml` | Pin fields above, weights path, `enabled` |
| `turing/src/ugv_perception/tests/test_depth_meters.py` | Whole chain, including one synthetic plane and one hole-resize case |

Do **not** add `DummySource`.  
Do **not** publish `/cmd_vel`.  
Do **not** train.  
Do **not** mark T08 done from a still that has no real `CameraInfo`.  
Do **not** compile or destroy the depth IR inside the frame loop.  
Do **not** add a second depth thread in v1.

---

## When coding (after this file is frozen)

1. User runs `turing/scripts/fetch_da3metric_large.sh`. Do not commit the safetensors.
2. Export with a **dedicated wrapper** around the DA3 metric net. Its only outputs are `depth_raw` and `sky` from the depth head, taken **before** `_process_mono_sky_estimation`. Exporting `DepthAnything3Net.forward` or the stock prediction object is not accepted: that path writes sky pixels to a far depth. No CPU fallback.
3. **Startup, once:** compile SegFormer, compile DA3, warm up each, then alternate one SegFormer infer and one DA3 infer while both stay resident. Record that OpenVINO GPU peak. If it fits, `enabled: true` and both stay compiled. If not, `enabled: false` for v1 and DA3 is not loaded again.
4. On each accepted frame, run SegFormer, **publish mask and confidence**, then run DA3 on that same RGB if enabled. The current mask does not wait. The callback does not return until DA3 returns, so the next frame can wait in `KEEP_LAST` 1. T11 live numbers include that wait.
5. Build `K_model` from `K_camera` with the preprocess below. Scale raw depth with `focal_model`. Then hole-safe resize the meter map to camera `H×W` and back-project with `K_camera`.
6. A hole is `sky >= 0.3` or a non-finite raw value. Holes are omitted from the cloud. They are not 0 m and they are not class 0.
7. If depth is disabled, skipped, or fails, publish **no** cloud. The `{0,1,2}` mask for that frame still publishes.
8. Dev 3 uses a cloud only while `now − cloud.stamp ≤ perception_max_age` (0.50 s, same clock as T05). Older geometry is ignored, never turned into free space. That check is Dev 3's. It is not a second T11 age.
9. The cloud is metric geometry. Dev 3 rejects ground and traversable surface, then applies height, slope, and range. Section 9 applies to that occupied set. A trail point in the cloud is not automatically lethal.

---

## 1. What T08 is

Execution is sequential on one GPU. The data dependency is "same image T", not "mask waits for depth".

```
startup (once)
  compile SegFormer, compile DA3
  warm up each, then alternate one infer of each
  measure GPU peak with both resident
  fit    → keep both compiled
  no fit → T08 off for v1

accepted image T
       │
       ├──► SegFormer ──► publish mask + confidence     ◄── this frame's T11 age ends here
       │
       └──► DA3 on image T          (same callback, after the publish)
                depth_raw + sky, before sky-fill
                K_model from the preprocess
                depth_m = focal_model * raw / 300
                holes: sky >= 0.3 or non-finite
                hole-safe resize → camera H×W
                PointCloud2 with K_camera
                        │
                        ▼
                 Dev 3
                   ignore if age > 0.50 s (not free)
                   reject ground / traversable surface
                   height, slope, range
                   occupied geometry
                   (§9: that set stays lethal over class 1)
```

Stereo disparity, when a pair exists, is a later cloud into the same Dev 3 slot. It does not replace this mono model, and this model does not read the right eye.

---

## 2. IR contract

Export smoke shows these tensors from the depth head, before sky-fill:

| Output | Type | Meaning |
|---|---|---|
| `depth_raw` | float32 `[1, 1, H', W']` or `[H', W']` | Canonical depth at model resolution. Not meters. |
| `sky` | float32, same `H'×W'` | Sky score. A hole is `sky >= 0.3`. |

If an export path emits a boolean sky mask instead, that boolean is the hole mask and the 0.3 test is not applied again.

---

## 3. Preprocess and the two K's

Official mono default, frozen:

```
process_res: 504
resize_mode: upper_bound_resize
patch: 14
```

Steps, in order. No letterbox, no pad, no crop.

1. Longest side of the camera image becomes 504. The other side scales by the same factor and is rounded to an integer. Aspect ratio stays.
2. Each side is then resized to the nearest multiple of 14. That second resize is a few pixels and may change the aspect slightly.
3. ImageNet normalize. That does not move pixels.

`K_camera` is scaled the same way, twice. Width ratio scales `fx` and `cx`. Height ratio scales `fy` and `cy`. The result is `K_model`.

```
focal_model = (fx_model + fy_model) / 2
depth_m     = focal_model * raw / 300     # still at H'×W'
```

`K_camera` is not used in that multiply.

Because both steps are full-frame resizes, every model pixel corresponds to the full camera image. After the hole-safe resize below, back-projection uses `K_camera` unchanged.

---

## 4. Hole-safe resize

Plain bilinear is forbidden. Invalid source pixels must not create a finite output.

```
valid = finite(depth_m) AND sky < 0.3

weighted = resize(depth_m * valid)    # bilinear, invalid contributes 0
weight   = resize(valid)              # bilinear of 0/1

where weight >= 0.5:
    depth = weighted / weight
elsewhere:
    hole
```

`0.5` is `valid_coverage`. An output pixel that is only partly supported by valid source pixels stays a hole.

---

## 5. PointCloud2

Holes are omitted. The cloud is **not** one point per pixel. Dev 3 must not recover a pixel index from point order.

| Field | Value |
|---|---|
| `x`, `y`, `z` | float32, optical frame, meters |
| `is_dense` | false |
| Layout | unorganized valid points only |
| `header.stamp` | the image stamp |
| `header.frame_id` | the image optical frame |

No intensity, no class id, no sky flag.

---

## 6. Tests (`test_depth_meters.py`)

One synthetic plane:

```
Z = 2.0 m at the camera-image center
K_camera known
preprocess → K_model, H'×W'
raw chosen so focal_model * raw / 300 = 2.0
hole-safe resize back to H×W
back-project the center with K_camera
→ X = 0, Y = 0, Z = 2.0
```

Also:

- A 2 m, hole, 3 m triple resized with bilinear-through-hole does **not** become a finite middle value. The weighted rule does not invent that pixel when coverage is under 0.5.
- `sky == 0.3` is a hole. `sky == 0.29` is not, if the depth is finite.
- A NaN raw value is absent from the cloud.
- A cloud older than 0.50 s is ignored by the Dev 3 rule and is not free space.
- Depth disabled or failed publishes no cloud and does not block a mask that already passed T07.
- Compile is not called from the frame path. A second frame reuses the startup compiled model.
