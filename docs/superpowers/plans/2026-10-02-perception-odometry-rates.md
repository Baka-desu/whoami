# Perception and odometry rates: are they a problem, and the fix plan

**Status:** written 2026-10-02, not started. Revises Task 5 of
[`2026-10-02-3d-mapping-elevation.md`](2026-10-02-3d-mapping-elevation.md) and depends on its Task 23.

## Context

The Task 4 baseline (`docs/mapping/baseline.md`) measured camera 7.4 Hz, mask 3.1 Hz, depth 3.2 Hz, visual
odometry 1.1–1.4 Hz, segmentation 167 ms and depth 182 ms per frame. The owner asked whether these speeds are a
problem and for a plan to fix them.

They are a problem. Three of the four rates break a limit the stack already enforces, and the Phase 0 gate
(depth ≥ 5 Hz, `pose_valid` ≥ 90 %) cannot pass at these numbers. The existing Task 5 only trims the depth
path, which is not enough on its own: segmentation is half of the frame time and is untouched.

When this was written, another session was executing the mapping plan in this tree (Task 23 uncommitted, a
Task 5 "before" benchmark running). **Owner decision, 2026-10-02: this plan is implemented later**, after
Task 23 and that Task 5 work are committed, on top of whatever they did.

## Verdict

Limits are the ones in the repo today. "Peak age" = age on arrival + one period, which is what a 20 Hz checker
sees just before the next sample.

| Rate | Limit it meets | Result now |
|---|---|---|
| Visual odometry 1.1–1.4 Hz | pose age ≤ 0.5 s default, ≤ 1.5 s laptop (`pose_validity*.yaml`) | **Worst.** Peak pose age 1.1–1.3 s. Default timing: pose never valid, permanent safety hold. Laptop timing: passes with 0.2–0.4 s to spare, one lost frame trips it and costs a 1.0 s recovery hold. At the 0.8 rad/s limit the robot turns 38° between updates, against a 57° wide camera |
| Depth 3.2 Hz, 0.38 s old | Phase 0 gate ≥ 5 Hz; depth age ≤ 0.5 s default | **Fails the gate.** Peak age 0.69 s fails default timing every period; passes laptop (1.5 s) |
| Mask 3.1 Hz | safety arbiter and semantic costmap: a new sample within 0.5 s; gateway: mask stamp age ≤ 0.5 s (`watches.py:101`) | **Marginal.** 0.18 s spare for safety and costmap; one slow cycle gives a hold. Gateway peak age about 0.52 s, so the goal gate (HTTP 409) flickers |
| Camera 7.4 Hz | safety 1.0 s; must exceed the perception rate | **Not the limit today** (perception uses 1 frame in 2.3). It becomes the ceiling as soon as perception passes 7.4 Hz. Cause not proven |
| TF `odom->base_link` | contract ≥ 15 Hz (`dev.md`) | Equals the odometry rate. Cannot reach 15 Hz with visual odometry at any rate this plan produces; already recorded as CONFLICTS C9. Recorded, not fixed |

## Causes

1. **Odometry at a third of the depth rate: transport, not compute.** `rgbd_odometry` and `rtabmap` subscribed
   best effort to 2.1 MB messages and received fewer than half; each estimate takes 0.03 s. Fix is Task 23.
   After it, odometry should equal the depth rate.
2. **Mask and depth at 3 Hz: two fp32 models run one after the other, with CPU numpy work around each.**
   - DA3 runs fully fp32. Upstream runs its backbone under autocast and only the head in fp32
     (`/opt/Depth-Anything-3/src/depth_anything_3/api.py:126-128`, `model/da3.py:127-139`); our `_Da3Head`
     (`backend/cuda_pytorch.py:212-228`) bypasses that.
   - SegFormer-B5 runs fp32 on a 640x640 stretch of the 640x480 frame (`config/adapters/rugd.yaml:5`).
   - CPU work inside the timed stages: float64 resize and normalise before DA3, float64 hole-safe resize after
     it (23 ms), float32 resize before SegFormer, `np.unique` over 307,200 labels in the remap, the image
     decoded twice.
   - `seg` is one 167 ms number; nothing says how it splits between CPU and GPU.
3. **Camera 7.4 Hz: unproven.** The baseline blames long exposure in the dark; the calibration file notes
   "~8 fps" on another day. The negotiated frame rate is never read back (`webcam_stream.py:107-115`).

Rejected: GPU contention (GPU idle at 1 % with the stack stopped, no other compute process); a frame backlog
(every hop is newest-only, queue depth 1); slow odometry compute (0.03 s per estimate).

## Target

- Pass line: depth ≥ 5 Hz (the gate), mask period ≤ 0.3 s, odometry results ≥ 90 % of synced frames.
- Needed for the default 0.5 s budgets with margin: ≤ 150 ms per frame for both models (6.5 Hz).
- Stop optimising at depth ≥ 8 Hz (existing Task 5 rule).

No GPU benchmark was run for this plan (the other session was using the GPU), so the gains below are
estimates: autocast roughly halves each forward pass, GPU pre/post removes about 40 ms, giving about 180 ms
per frame (5.5 Hz) before the scheduler.

## Steps

### 0. Start condition
- `git status` shows no uncommitted changes under `turing/src/ugv_perception/` and
  `ugv_nav/ugv_localization/`, and no `bench_task5` / `pytest` process runs in `ugv-run`.
- Read `git log` and the numbers appended to `docs/mapping/baseline.md`. **Skip any step below that is already
  committed**; Task 23 and the "always done" items of Task 5 (depth independent of the mask, single decode) are
  expected to be.
- Re-measure 60 s on the live stack in a lit room. This is the "before" for everything below.

### 1. Split the timers (measure first)
Add stages `seg_pre`, `seg_infer`, `depth_pre` to `STAGES` (`node/metrics.py:15`); `seg` stays the whole cycle
so `/ugv/perception/stats` consumers keep working.
- `seg_infer`: wrap `adapter.infer` in `_CountingAdapter` (`adapter_node.py:54`).
- `seg_pre` and `depth_pre`: the same optional `stage` hook `DepthChannel.maps` already takes
  (`backend/depth_live.py:36`), added to `RugdSegformerAdapter`.
- Test with the injected clock already used in `tests/test_adapter_node.py`.
- Record the split in `baseline.md`. Any item in step 2 whose stage costs under 5 ms is skipped.

### 2. Faster, same outputs (in this order; re-measure after each; stop at the target)
| # | Change | Where | Parity test |
|---|---|---|---|
| a | DA3 backbone under `torch.autocast`, depth head with autocast disabled, as upstream does | `_Da3Head.__call__`, `cuda_pytorch.py:218` | depth vs fp32: relative difference < 1 % on valid pixels, same hole pattern |
| b | SegFormer forward under fp16 autocast; logits cast to float32 before the finite check, resize and softmax; labels cast to int32 on the GPU | `run_seg`, `run_decoded`, `cuda_pytorch.py:74,142` | canonical mask equal on ≥ 99.5 % of pixels, score difference < 0.01 |
| c | Pre-processing on the GPU in float32 for both models (resize, normalise), keeping the uint8 rounding of `_resize_u8` | new optional backend methods, found with `getattr` like `run_seg` is today (`adapter/rugd.py:161`); OpenVINO path unchanged | depth < 1 mm on the synthetic plane of `test_depth_meters.py`; mask as in (b) |
| d | DA3 metres + hole-safe resize on the GPU, one copy back instead of two | same backend call as (c); `depth/geometry.py` stays the reference | < 1 mm vs `hole_safe_resize`; holes survive; sky threshold |
| e | `np.unique` replaced by `np.bincount` in the remap check | `remap/apply.py:55` | existing `test_remap.py` |

GPU parity tests run only in the container (skip without CUDA and weights). The kernel tests keep their fakes.

### 3. Scheduler (only if depth is still under 8 Hz)
Existing Task 5 item 4: depth on every frame, mask on every Nth so the mask period stays ≤ 0.3 s. The degraded
flag is still published on every tick (the arbiter needs it within 0.5 s). Pure kernel with an injected clock,
tested without ROS.

### 4. Input size (only if the 5 Hz gate still fails; needs the owner's yes)
SegFormer at the native 480x640 instead of 640x640, or DA3 `PROCESS_RES` 504 → 392. Both change model output,
so this step stops and reports numbers instead of applying itself.

### 5. Camera rate
- `webcam_stream.py`: read back and print the negotiated fps and FOURCC; print delivered fps every 10 s.
- `camera_driver.py`: log published fps every 10 s.
- Owner step: run in a lit room and read both numbers.
  - ≥ 15 fps in light: the cause was exposure. Add one line to `ugv_bringup/README.md`. No more code.
  - Still about 7.5 fps: the two logs say which hop loses frames. Then try, as opt-in flags with defaults
    unchanged: the MSMF backend, no MJPG request, fixed exposure.

### 6. Timing budgets that match what is measured
- `bag_eval.launch.py` gains a `timing` argument passed through to `localization.launch.py`. Today it always
  replays with the 0.5 s default, so the Task 7 gate would judge `pose_valid` against budgets the live run does
  not use.
- Measure p99 depth age, `/odom` age and `map->base_link` age after steps 2–3. If p99 pose age ≤ 0.4 s,
  `localization_timing:=default` is usable and is recorded as such. Otherwise keep `laptop` and correct its
  stale header ("DA3 depth ~1 Hz, ~0.7 s old", `localization.launch.py:73` and the `_laptop` YAMLs).
- No change to `safety_timeouts.yaml` or `api.yaml`: above 5 Hz both pass with margin.

### 7. Record
- `docs/mapping/baseline.md`: the verdict table and before/after numbers per step.
- `mindmap.md`: one D-row with the measured `odom->base_link` rate against the 15 Hz contract.
  `architecture.md` is not edited.
- `2026-10-02-3d-mapping-elevation.md`: Task 5 text brought in line with what was done.

## Files

- Perception: `turing/src/ugv_perception/node/{metrics,adapter_node}.py`, `backend/{cuda_pytorch,depth_live}.py`,
  `adapter/rugd.py`, `remap/apply.py`, `tests/test_adapter_node.py`, `tests/test_depth_meters.py`,
  `tests/test_rugd_decode.py`, one new GPU parity test file.
- Camera: `ugv_nav/ugv_bringup/scripts/webcam_stream.py`, `ugv_bringup/nodes/camera_driver.py`, `README.md`.
- Localization: `ugv_nav/ugv_localization/launch/bag_eval.launch.py`, `launch/localization.launch.py` (comment),
  `config/*_laptop.yaml` (comments), `test/test_ros_smoke.py`.
- Docs: `docs/mapping/baseline.md`, `mindmap.md`, the mapping plan.

## Verification

| What | Command | Proves |
|---|---|---|
| Kernels | `cd turing && python -m pytest` | timers, remap, depth maths unchanged |
| GPU parity | `docker exec ugv-run bash -lc 'cd /repo/turing && python3 -m pytest src/ugv_perception/tests -k parity -v'` | fp16 and GPU paths match fp32 within the limits above |
| Per-step speed | the `bench_task5.py` method (real node, fixed frame, 50 frames, `REF=` comparison) before and after each item of step 2 | each change is faster and output-equal |
| ROS stack | container: `/ws/sync_ws.sh`, then `colcon test --packages-select ugv_localization ugv_bringup && colcon test-result --verbose` | `timing` passthrough, Task 23 test still green |
| Live | `/ws/restart_live.sh`, 60 s in a lit room: `/ugv/perception/stats`, `ros2 topic hz /camera/image_raw /odom`, `rgbd_image` vs `odom_info` counts, `pose_valid` duty | the target rates and ages on the real stack |

Done means: the live numbers meet the pass line, or the plan stops at step 4 with the numbers reported.

## Found while reading, not in this plan

- The mask's age excludes inference time: the second freshness check reuses the tick-start time
  (`compose/tick.py:84`, `adapter_node.py:180`).
- `turing/config/perception/depth.yaml` is not read by any code; `enabled: false` has no effect.
- The voxel layer has no age limit on the depth cloud (`expected_update_rate: 0.0`, `costmaps.yaml`).
- `test_ros_smoke.py::test_s4_launch_nodes_per_odom_source` may predate the `timing` argument (not run).
