# Baseline measurement (plan Task 4)

**When:** 2026-10-02, branch `mapping-3d` at 7f26879, `live_cam` profile in the `ugv-run` container (RTX 4060 laptop),
laptop webcam through `webcam_stream.py`, `localization_timing:=laptop`, 60 s sample.

## What is valid and what is not

The camera image was **black** during this run: mean brightness 1.46 on a 0–255 scale, Laplacian variance 0.69,
and no pixel changed between frames. The room was dark or the lens was covered.

- **Valid:** rates and per-stage compute times. They do not depend on what the image shows.
- **Not valid:** anything about visual odometry quality or depth stability. With a black image the feature
  detector locks onto sensor noise and DA3 returns arbitrary depth. Those rows are recorded below only to show
  the symptom, and must be re-measured with a lit scene.

## Rates and stage times (valid)

| Quantity | Value | Source |
|---|---|---|
| Camera frames delivered | 7.4–7.5 Hz | `/camera/camera_info`, `/camera/image_raw` |
| Mask rate | 3.1 Hz | `/ugv/perception/stats` |
| Depth rate | 3.2 Hz | `/ugv/perception/stats`, `/perception/depth/image` with a reliable subscriber |
| Synced RGB-D rate | 3.3 Hz | `/rtabmap/rgbd_image` |
| Visual odometry updates | 1.1–1.4 Hz | `/rtabmap/odom_info` |
| `/odom` | 0 Hz | see "Odometry" |
| Depth image age at the subscriber | 0.38 s mean | stamp vs wall clock |

| Stage | Mean per frame |
|---|---|
| `seg` (whole perception cycle: decode, SegFormer, remap, gates) | 166.6 ms |
| `depth_infer` (DA3 forward pass, fp32) | 159.5 ms |
| `depth_post` (metres, resize, back-project; numpy on CPU) | 22.7 ms |
| `publish` | 2.4 ms |
| `cloud` | 1.7 ms |
| `decode` | 1.2 ms |

One frame costs about 355 ms, which is the 3 Hz seen. The old "~1 Hz depth" figure is stale: depth is 3.2 Hz.
The camera delivered only 7.4 fps (long exposure in the dark), so no stage can exceed that in this run.

Depth errors: 0.

## Odometry (two separate findings)

1. **Visual odometry receives fewer than half the frames it is sent.** `rgbd_sync` publishes
   `/rtabmap/rgbd_image` RELIABLE at 3.3 Hz, but `rgbd_odometry` and `rtabmap` subscribe to it BEST_EFFORT, and
   odometry produced only 1.1–1.4 updates per second while each estimate took about 0.03 s. Each message is
   about 2.1 MB (RGB + 32FC1 depth). This is the likely source of the "odometry ~1 Hz" in the earlier status
   report. Likely cause, not yet proven: large best-effort messages being dropped. It is independent of the
   dark image and is fixed and verified in plan Task 5b.
2. **Tracking was lost on every second frame, with 0 inliers** (features 689–806 per frame, matches 0–65,
   inliers always 0; the node reset after each loss). `odom_selector` dropped every visual odometry message, so
   `/odom` never published and `/ugv/pose_valid` was false for all 1200 samples. With a black image this is
   expected and says nothing about the stack; it must be re-measured in light.

## Depth stability (not valid in this run)

Frame-to-frame global scale ratio (median over pixels of `depth_t / depth_t-1`): p10 0.73, p50 1.00, p90 1.33,
standard deviation 0.26. Centre-patch median 0.75–2.89 m. Both are DA3's response to a black frame.

## Still to measure (needs a lit scene and the owner)

- Visual odometry: lost fraction, inliers, `/odom` rate, `pose_valid` duty, on a static lit scene.
- DA3 scale wobble on a static lit scene (the framewise scale ratio above).
- Wall test: tape-measured distances at 1, 2, 3, 4, 5 m against DA3 depth. This sets `NoiseModel.sigma0` / `k`
  for the elevation map and confirms the 5.0 m fusion range.

## Decisions taken from these numbers

- Task 5's target of 8 Hz depth cannot be met by trimming the depth path alone: segmentation (167 ms) and depth
  (182 ms) share one GPU serially. The Phase 0 gate needs 5 Hz.
- Skipping the point cloud when nobody subscribes (Task 5 step 3) would save 1.7 ms and is not worth changing
  the `/perception/depth_cloud` contract for.
- Getting every synced frame to visual odometry (Task 5b) is the cheapest improvement: it should raise the
  odometry rate from about 1.3 Hz to the depth rate without touching any model.

## Task 5: before/after (depth for every frame, at a usable rate)

**When:** 2026-10-02, `ugv-run` container, RTX 4060 laptop, branch `mapping-3d`. Target set by the controller:
**depth >= 5 Hz** (not 8 Hz). Step 3 of the brief (skip the point cloud with no subscriber) was dropped by the
controller: it saves 1.7 ms and changes a published contract.

### How it was measured

A throwaway benchmark (not committed) builds the real RUGD adapter and the real DA3 channel the way
`adapter_node.py:main()` does (`build_live_adapter`, `build_depth_channel`, both on the CUDA backend), wraps them in the
real `PerceptionAdapterNode` and feeds one fixed, textured 640x480 RGB frame with the laptop webcam K, back to back,
through the node's own image callback: 5 warm-up frames, then 50 timed frames. Back to back means this is the rate the
node can sustain, not the rate the camera delivers. Stage columns are the mean per frame **on the frames where the
stage ran** (the same rule as `/ugv/perception/stats`). The mask and the depth image of the last frame are saved and
compared with the "before (b)" run. Another worker's ROS tests were using the CPU in the same container, so each
configuration was measured twice.

| Configuration | Run | ms / frame | Mask Hz | Depth Hz | decode | seg | depth_infer | depth_post | cloud | publish |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Before (commit 2734f12) | a | 365.2 | 2.19 (1) | 2.19 (1) | 1.77 | 202.18 | 163.97 | 29.66 | 2.11 | 1.87 |
| | b | 335.3 | 2.98 | 2.98 | 1.63 | 144.49 | 157.89 | 26.57 | 1.73 | 1.55 |
| Depth independent of the mask, one decode | a | 301.9 | 3.31 | 3.31 | 1.08 | 123.49 | 150.47 | 22.83 | 1.74 | 1.13 |
| | b | not taken (2) | | | | | | | | |
| + DA3 pre/post-processing on the GPU | a | 271.4 | 3.68 | 3.68 | 1.12 | 121.02 | 132.92 | 12.57 | 1.53 | 1.09 |
| | b | 277.1 | 3.61 | 3.61 | 1.25 | 124.75 | 133.56 | 13.45 | 1.68 | 1.12 |
| + DA3 under fp16 autocast | a | 204.9 | 4.88 | 4.88 | 1.13 | 122.66 | 65.03 | 12.14 | 1.61 | 1.15 |
| | b | 208.7 | 4.79 | 4.79 | 1.07 | 125.55 | 66.46 | 11.89 | 1.50 | 1.07 |
| + first scheduler, minimum mask spacing 0.25 s (superseded, see fix round 1) | a | 140.4 | 3.56 | **7.12** | 1.08 | 119.06 (3) | 64.41 | 11.96 | 1.56 | 0.91 |
| | b | 138.0 | 3.62 | **7.24** | 1.00 | 116.36 (3) | 64.27 | 11.42 | 1.42 | 0.86 |

1. Run a had 10 frames of 50 on which no mask was published, and so no depth either (old code ties depth to the mask).
   Cause not captured: the adapter-error counter was added to the benchmark afterwards. The same thing was seen in
   the first 25-frame diagnostic run (3 frames) and never again in 9 later runs (more than 450 frames, 0 adapter
   errors). Both early runs were at process start-up under CPU contention. It is in the segmentation path, which this task
   does not touch, so it is only recorded here.
2. The container's Docker engine stopped (host disk full) before the second run of this configuration.
3. Segmentation ran on 25 of the 50 frames. Depth ran on all 50.

The first scheduler row (a minimum spacing of 0.25 s between segmentations, commit c5e273e) reached **depth 7.1 to
7.2 Hz**, against 3.0 to 3.3 Hz before, with a mask rate of 3.6 Hz. Without any scheduler the same code reaches 4.8 to
4.9 Hz, just short of 5 Hz, because segmentation (120 ms) and depth (about 77 ms now) share one GPU and run one after
the other. That scheduler was replaced in fix round 1 (below), because it published the degraded flag only on segmented
frames and skipped every second frame of a 4 to 6 Hz camera for no gain. These two rows and the DDS run in the next
paragraph are the **first** scheduler; they were not repeated for the new one in-process.

Over DDS, with the real node (`main()`), a synthetic camera at 12 Hz and a subscriber for every output, 20 s after
warm-up, first scheduler: before 3.25 Hz mask and 3.25 Hz depth; after 3.45 Hz mask and 6.7 to 6.8 Hz depth (two runs).
The `/ugv/perception_degraded` messages came every 0.29 s on average (longest gap 0.34 and 0.40 s; before: 0.30 s,
longest 0.75 s), but only because the camera was fast: the gap was max(2T, one segmented tick) and would have reached
the arbiter's 0.5 s timeout at 4 Hz.

### Numerical checks

- The last mask is **byte-identical** to the one before the changes, in every run.
- The last depth image against "before (b)": identical NaN mask; largest absolute difference 0.48 mm after the GPU
  pre/post-processing, 0.89 mm (0.077 % relative) after fp16 as well.
- Tests with the real DA3 weights on three synthetic frames: GPU against numpy pre/post-processing (fp32) differs by
  0.2 to 0.7 mm at most; fp16 against fp32 by at most 0.17 % on valid pixels (the gate is 1 %). Both skip when CUDA or the
  weights are absent.
- Hole-safe resize on the 2 m plane, GPU against numpy: well under 1 mm.
- Segmentation never runs under fp16: a test records the autocast state during `run_seg`, `run_decoded` and `run_all`
  of the RUGD net and requires it off.
- **Superseded by the PR #40 review (mindmap D19, D21).** DA3 runs **FP32 on every backend**; the fp16 path measured
  above was removed (FP32 3.6 Hz vs fp16 4.9 Hz end to end on the RTX 4060 is history, not an option). OpenVINO
  networks compile with an explicit f32 `INFERENCE_PRECISION_HINT`, because the GPU plugin would otherwise pick f16 on
  Arc. RUGD never runs in fp16. The stage timing, `/ugv/perception/stats`, the forced `cuda.synchronize`, depth on
  every frame and the segmentation scheduler described in this file were deferred (removed) as well; depth is again
  published after each mask.

### What each stage covers now

| Stage | Covers |
|---|---|
| `decode` | ROS `Image` to `ImageView` (`image_msg_to_view`) plus the one `decode_frame` of the tick (view to RGB array, checks on K and frame id). Before: the first, plus a second `decode_frame` inside the depth step, while the decode of the cycle was counted under `seg`. |
| `seg` | The cycle after the decode, and only on frames that are segmented: freshness check, `adapter.infer` (RUGD preprocess, SegFormer, GPU decode), remap, gates, mask. Before: it included the decode. |
| `depth_infer` | Backend sizing, DA3 preprocess (on the GPU for CUDA, numpy for OpenVINO), forward pass (fp16 autocast at the time of this measurement; FP32 only since the PR #40 review). On CUDA the device is synchronised before it closes, so the time is real. |
| `depth_post` | Metres, hole-safe resize and the copy back (GPU for CUDA, numpy otherwise), then the back-projection to XYZ (numpy). |
| `cloud` | Building the 32FC1 depth `Image` and the `PointCloud2` messages. |
| `publish` | All `publish()` calls of the tick. |

Frames on which segmentation is skipped have no `seg` sample and publish no mask. They still publish the degraded flag.

### Fix round 1: segmentation has priority, the flag goes out on every frame

The review of the first scheduler found that `/ugv/perception_degraded` was published only on segmented, stale and
undecodable frames, that the watchdog (keyed on the newest image stamp) said nothing while fresh images arrived and no
mask was produced, and that a minimum spacing is the wrong rule for a 4 to 6 Hz camera. Changed:

- **Scheduler.** A fresh frame is depth-only only if skipping it is predicted to keep the start-to-start gap between
  segmentations at or below `mask_max_gap_s` (ROS parameter, default 0.30 s, must be above 0 and below
  `perception_max_age`). Next start predicted = max(now + depth-only tick, arrival of this frame + camera interval).
  `seg_due` is a pure function of those numbers; `SegScheduler` keeps the estimates. Stale and undecodable frames are
  never skipped. With no estimate yet the frame is segmented, so the first two frames always are.
- **Estimates.** The depth-only tick is a running average (0.3 weight) of decode + depth time, from every frame
  that completed depth, ignoring the first sample (lazy initialisation). The camera interval is **not** a mean of the
  stamp deltas the node sees: the node takes only the newest frame of a depth-1 queue, so under load those deltas are
  multiples of the interval and a 30 fps camera would look like a 5 Hz one and never alternate. It is measured exactly
  when the node was idle waiting for a frame (the next frame is then the very next one) and kept until a shorter stamp
  delta contradicts it. A camera that never lets the node idle is faster than a depth tick, and then the interval cannot
  be the larger term, so it is left out. A wrong skip is paid for once (the node then idles and measures it).
- **Degraded flag.** Published on every processed frame again. On a depth-only frame it is: the last segmented decision
  OR no mask ever published OR no mask published for longer than `perception_max_age` (monotonic time since the newest
  one went out). The watchdog thread, while its image rule says the input is live, additionally raises it when no mask has
  been published for longer than `perception_max_age` (counted from the first image while there is none); its image rule
  is unchanged. This is liveness of the mask stream, **not** the age of the newest mask's image stamp: that is latency plus
  the time the mask is held until the next one replaces it (see "Open for the owner"), and compared with 0.5 s it would
  trip in normal operation. Freshness (latency) is still checked for every segmented frame by `evaluate()`.

Simulation of the real node (injected clocks, seg 123 ms, depth-only tick 80 ms, camera interval T, 30 s, queue depth 1):

| T (camera) | depth Hz | mask Hz | longest gap between degraded flags | longest start-to-start mask gap | gaps over 0.30 s | steady state |
|---|---:|---:|---:|---:|---:|---|
| 33 ms (30 fps) | 7.09 | 3.58 | 203 ms | 283 ms | 0 | alternates |
| 83 ms (12 Hz) | 7.08 | 3.57 | 206 ms | 286 ms | 0 | alternates |
| 135 ms (7.4 Hz) | 6.81 | 3.71 | 257 ms | 337 ms | 1 (start-up) | mostly alternates |
| 200 ms (5 Hz) | 5.03 | 4.86 | 317 ms | 397 ms | 1 (start-up) | nearly every frame segmented |
| 240 ms (4.2 Hz) | 4.20 | 4.20 | 240 ms | 240 ms | 0 | every frame segmented |

The single gap over 0.30 s at 7.4 Hz and 5 Hz is the one wrong skip before the camera interval is known (337 and 397 ms,
within the bound plus one frame interval). At 7.4 Hz the camera and the 283 ms cycle beat against each other, so now
and then two frames in a row are segmented; at 5 Hz, a tick (203 ms) slightly longer than the camera interval (200 ms)
slips 3 ms a frame and now and then a depth-only frame fits within the bound.

Over DDS, real node, real models, synthetic camera at 7.4 Hz, 20 s after warm-up (one run; another worker's RTAB-Map test
was using the CPU): **depth 6.25 Hz, mask 3.70 Hz**; 126 degraded flags (one per processed frame), longest gap 0.286 s,
none true; mask gaps 0.273 s mean, 0.369 s longest; mask age at arrival 0.174 s mean, 0.267 s longest; depth age 0.236 s mean.
Stage means from `/ugv/perception/stats`: seg 126.6 ms, depth_infer 67.2, depth_post 12.6, cloud 1.7, publish 1.9.

Margin of the liveness rules (time between two mask publications against the 0.5 s limit): over DDS at 7.4 Hz the longest
was 0.369 s (margin 0.13 s); in the simulation (33 to 300 ms cameras) the longest was 0.397 s (margin 0.10 s, the one wrong
skip at 200 ms before its interval is known) and in steady state 0.283 to 0.300 s (margin 0.20 s or more). No degraded flag
was true in any healthy run, and in the simulation a depth-only frame never saw a mask published more than 80 ms earlier.

**Open for the owner (not changed here).** A mask is held until the next one replaces it, so the mask in use just before
a replacement is (mask gap + latency) old. Over DDS at 7.4 Hz that was **0.447 s on average and 0.584 s at worst** (7 of 73
replacements over 0.5 s, camera with no latency), against `perception_max_age` = 0.5 s; the simulation gives 0.44 to
0.49 s at 30 fps and 12 Hz. It was already so before this task: the old node produced masks at 3.25 Hz (gap about 0.31 s)
with a 0.176 s mean arrival age, so about 0.48 s held on average (derived from those two measured numbers, not measured
directly), and one mask arrived 0.536 s old (measured) with the flag false, because freshness is checked before inference
only. The degraded rules above do not use this age, so they do not flicker; but a consumer that applies architecture §8.4
literally (reject a mask older than `perception_max_age`) would reject a held mask in the last part of some cycles.
Options: (a) raise `perception_max_age` (about 0.65 s would clear the worst case measured) together with the safety
timeouts that follow it; (b) lower `mask_max_gap_s` to about 0.20 s, which removes the alternation at 7.4 Hz (predicted gap
283 ms) and puts depth back near 4.9 Hz; (c) accept it. No budget was changed.

## Task 23: odometry input QoS, before and after

**When:** 2026-10-02, branch `mapping-3d`. The real stack (`localization.launch.py`: `rgbd_sync`, `rgbd_odometry`, `rtabmap`,
`odom_selector`, `pose_validity`, `odom_source:=auto`) in the `ugv-run` container, fed synthetic sensors by
`test/test_ros_stack.py` (static textured scene, 640x480 rgb8 + 32FC1 depth + CameraInfo, one complete frame about every
0.27 s, so 3.6 frames/s against 3.3 Hz live). Window 32 s (30 s of frames plus 2 s drain). Each `rgbd_image` is ~2.1 MB.
Counts are messages in the window; "reliable subscriber" is a counting subscriber in the test process, so it sees what
`rgbd_sync` actually published.

### The cause is transport loss on a best-effort subscription, not odometry

| `laptop` timing, 640x480, reliable camera | Before | After |
|---|---|---|
| Complete frames sent | 116-117 | 116-117 |
| `rgbd_image` seen by a reliable subscriber | 116 of 116 | 113-117 of 116-117 |
| `rgbd_image` seen by a best-effort subscriber (test process) | 0-2 of 116 | 6-48 of 116 (still lossy) |
| `rgbd_odometry` `odom_info` | **23-59 (0.7-1.8 Hz), 20-51 %** (typical 33, 1.0 Hz) | **116-117 (3.65 Hz), 100 %** |
| `rtabmap` `/rtabmap/info` (a SLAM step; `Rtabmap/DetectionRate` caps it at 2 Hz) | 4-6 (0.12-0.19 Hz) | 58 (1.8 Hz) |
| `rgbd_odometry` / `rtabmap` subscription to `/rtabmap/rgbd_image` | BEST_EFFORT / BEST_EFFORT | RELIABLE / RELIABLE |

- `rgbd_sync` delivered every frame to a reliable reader (116 of 116) while the best-effort readers got a fraction. After
  the change the same odometry node, same data, same machine processes all of them (3.65 Hz, each estimate 0.03-0.05 s
  against a 0.27 s frame period). So the missing frames were lost before odometry, not skipped by it.
- Mechanism (likely, not separately proven): Fast DDS cannot create its shared-memory transport in this container
  (`RTPS_TRANSPORT_SHM Error: Failed to create segment ... SHM Transport is not supported` in every node log), so the 2.1 MB
  message travels as UDP fragments. Best effort never retransmits, so one lost fragment loses the whole message. A 0.5 MB
  message (320x240) is hit too, less often (before: 29-31 of 117 reached odometry).
- The synthetic ratio (20-51 %) brackets the live one (1.1-1.4 Hz of 3.3 Hz, 33-42 %).

| `default` timing, 320x240, best-effort camera (the documented contract) | Before | After |
|---|---|---|
| `rgbd_sync` publishes `rgbd_image` | BEST_EFFORT (no reliable subscriber can connect) | RELIABLE |
| `rgbd_odometry` `odom_info` of 117 frames | 29-31 (0.9-1.0 Hz) | 116-117 (3.65 Hz) |
| `/rtabmap/info` | 5-6 | 58 |

### What changed (configuration only)

- `config/rgbd_odometry.yaml`, `config/rtabmap_rgbd.yaml`: `qos: 1` (reliable) and `topic_queue_size: 10 -> 2`, so a
  slow consumer skips old frames rather than replaying a backlog of 2.1 MB messages. Both files are shared by the `default`
  and `laptop` timing profiles.
- `config/rgbd_sync.yaml` (`default` timing): `qos: 2 -> 0`. This was needed: rgbd_sync's `qos` also sets the QoS of
  the `rgbd_image` it publishes, so with `qos: 2` the publisher was best effort and reliable consumers could never connect.
  `0` (system default) is a best-effort subscriber on the camera inputs, so they still connect to reliable and best-effort
  drivers alike (`docs/localization/interfaces.md`), and a reliable publisher. `rgbd_sync_laptop.yaml` already had `qos: 1`.
- Pinned by `test_x3_every_synced_frame_reaches_odometry_and_slam` (both timing profiles).

### Found, not fixed

- `default` timing at 640x480 loses most frames **before** `rgbd_sync`: its camera, depth and CameraInfo inputs are
  best effort by design, and only 7 of 117 frames paired in the synthetic run (the odometry fix does not change that).
  The `laptop` profile avoids it with reliable inputs, because Dev 5's `camera_driver` is reliable. Whether `default`
  should do the same is a decision for the owner (it would stop `default` connecting to a best-effort camera).
- Not re-measured on the live robot; the live rate should now follow the synced rate (3.2-3.3 Hz at the measured depth rate).

## Review fixes for 0c5f1d9 (Task 23)

**When:** 2026-10-02, branch `mapping-3d`. Same real stack and harness as Task 23 (`ugv-run` container, synthetic sensors from
`test/test_ros_stack.py`), measured with throwaway variations of the harness.

### The x3 guard on the camera to `rgbd_sync` hop was too tight

`test_x3` asserted that `rgbd_sync` delivered at least 90 % of the frames sent. That hop is not what Task 23 changed and it is
best effort (lossy by design) in the `default` profile. In a full-suite run the `default-320x240-best-effort-camera` case got
`rgbd_image=96` of `sent=117` (82 %) while `odom_info` was 100 % of `rgbd_image`. Three more runs of that case gave 87, 98 and
90 of 116-117 (74-84 %, each would have failed the old guard; `odom_info` 100 % of `rgbd_image` every time), while the
measurement runs below saw 105-120 of 120 at 4 Hz: the loss before `rgbd_sync` varies a lot with host load. The guard is now a
vacuity floor (at least 50 % of the frames sent, so there is something to count). The requirement stays as it was: `odom_info` is at least 90 % of
`rgbd_image`.

### `topic_queue_size: 2` also sets the depth of the rtabmap node's `/odom` subscription: measured, no effect

`rtabmap_rgbd.yaml` has `topic_queue_size: 2` for the 2.1 MB `rgbd_image`. The rtabmap node applies it to every subscription, so
`/odom` is a reader queue of depth 2 as well (`get_subscriptions_info_by_topic('/odom')`: rtabmap depth 2 RELIABLE; pose_validity
and distance_tracker depth 20). `test_x3` publishes wheel odometry with each frame (about 4 Hz), so it could not show a problem with a faster
odometry stream. Check: the same x3 window (30 s of frames at 4 Hz plus 2 s drain, 120 frames sent), wheel odometry on its own timer
at 30 Hz instead of 4 Hz. `/rtabmap/info` is one SLAM step (`Rtabmap/DetectionRate` caps it at 2 Hz).

| Profile | Wheel odometry | Windows | `/rtabmap/info` per 32 s | Rate |
|---|---|:---:|---|---|
| `laptop`, 640x480, reliable camera | 4 Hz (with the frames) | 2 | 60, 60 | 1.87 Hz |
| `laptop`, 640x480, reliable camera | 30 Hz | 3 | 60, 58, 56 | 1.87, 1.81, 1.75 Hz |
| `default`, 320x240, best-effort camera | 4 Hz (with the frames) | 2 | 58, 55 | 1.81, 1.72 Hz |
| `default`, 320x240, best-effort camera | 30 Hz | 2 | 59, 56 | 1.84, 1.75 Hz |

The spread inside one row (55-60) is as large as the difference between rows, so there is no degradation. Pose pairing, checked
separately: moving scene (0.3 m/s sideways), `default` timing, wheel odometry at 30 Hz, depth-2 queue. For each graph node the error
of its `pose.y` against the true y at the node's image stamp had a standard deviation of 0.000 m (maximum deviation from a constant
offset 0.000 m) in two runs, so each image still gets the pose from its own stamp (consistent with rtabmap reading it from TF at
the image stamp). The constant offset itself (0.000 m and -1.419 m) is a `map -> odom` shift, probably from the odom source switch at
start-up (not investigated). Decision: keep 2 and say so in the YAML comment.

### Found, not fixed

- The harness starts `ros2 launch` with `stdout=PIPE` and does not read it until `close()`. A pipe holds 64 KB; rtabmap logs one
  line per SLAM step and `rgbd_odometry` warns per frame, so a long run blocks the nodes in `write()`. Seen in every run longer
  than about 190 s (5 of 5, moving scene): `/rtabmap/info` stops at step 344-346, `rgbd_odometry` stops after about 30 s, and the
  `rtabmap` and `rgbd_odometry` main threads sit in `pipe_write`. With the launch output sent to a file the same 330 s runs
  completed (608 and 609 steps, no gap over 0.65 s). The existing tests stay under the limit (`test_x1` runs up to about 150 s);
  I did not check whether it explains the historical flakiness of `test_x1[image]`. Fix: write the launch output to a file.

## Task 8 fix round 1: a viewer attaching `cloud_map` mid-mission (review I1)

**When:** 2026-10-02, branch `mapping-3d`, `ugv-run` container, real launch file and RTAB-Map nodes fed synthetic sensors
(`test/test_ros_stack.py`). **Question:** the gateway (`ugv_api`) subscribes `/rtabmap/cloud_map` and `/rtabmap/mapPath` only
while a viewer is open (destroyed `idle_timeout_s` = 10 s after the last request). rtabmap builds `cloud_map` inside its SLAM
callback, so the first step after a late attach assembles the whole map. Does that stall SLAM enough to matter
(`/rtabmap/info` gaps, `slam_stale` on `/ugv/pose_valid`)? The Task 8 risk check only covered subscribers attached at t = 0.

**Method** (`test_m1_measure_late_cloud_map_attach`, opt-in, skipped by default):
```
MSYS_NO_PATHCONV=1 docker exec -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -e UGV_MEASURE_LATE_ATTACH=1 -e UGV_MEASURE_MAP_S=150 \
  -e UGV_MEASURE_CYCLES=3 -e UGV_MEASURE_OUT=/tmp/late_attach.jsonl ugv-run bash -lc 'ulimit -c 0; \
  cd /ws/src/ugv_nav/ugv_localization && source /opt/ros/lyrical/setup.bash && source /ws/install/setup.bash && \
  python3 -m pytest -p no:cacheprovider test/test_ros_stack.py -k "m1 and default" -q'      # or "m1 and laptop"
```
The robot slides sideways at 0.3 m/s (about 2 graph nodes per second). Nothing subscribes any map topic while mapping (the
probe's own `/map` subscription is off: nothing subscribes `/map` on the robot either, and it would keep rtabmap's map cache
warm). After `MAP_S` seconds the probe attaches like the gateway (`cloud_map` RELIABLE + TRANSIENT_LOCAL depth 1, `mapPath`
RELIABLE depth 1) for 15 s, detaches for 20 s, and repeats. Per attach: worst `/rtabmap/info` arrival gap in the 10 s after it
(the gap spanning the attach counts), the largest stamp age `/rtabmap/info` reached before the next one (what `slam_max_age_s`,
2.0 s `default` / 4.0 s `laptop`, is compared with), the pose status reasons, the cloud size, rtabmap's own `Maps update` +
`pub` time from its per-step log line and (after the fix) map_assembler's `Updating` + `Publishing data` time.
Profiles: `default` timing, 320x240, best-effort camera, frames at 10 Hz; `laptop` timing (the live profile), 640x480,
reliable camera, frames at 4 Hz.

### Before: rtabmap assembles `cloud_map` in its SLAM step (16 attaches, 6 runs)

"Clean gap" = worst gap in the 60 s before the first attach of the run (no subscriber at all).

| Run | Attach at s | Graph node | Cloud | Cloud after s | Clean gap s | Gap after s | Stamp age after s | Attach step maps+pub | `slam_stale` |
|---|---|---|---|---|---|---|---|---|---|
| default a | 150 | 275 | 38.9k | 0.43 | 0.67 | 0.64 | 0.96 | 172 ms | no |
|  | 185 | 340 | 46.6k | 0.49 | - | 0.69 | 0.99 | 207 ms | no |
|  | 220 | 405 | 54.4k | 0.64 | - | **1.07** | 1.37 | 239 ms | no |
| laptop a | 150 | 284 | 44.2k | 0.34 | 0.60 | **1.56** | 1.90 | 221 ms | no |
|  | 185 | 348 | 52.1k | 0.32 | - | 0.78 | 1.16 | 237 ms | no |
|  | 220 | 415 | 60.2k | 0.64 | - | 0.80 | 1.33 | 270 ms | no |
| default b | 150 | 275 | 41.4k | 0.60 | 0.64 | 0.68 | 1.00 | 193 ms | no |
|  | 185 | 339 | 49.1k | 0.27 | - | 0.79 | 1.10 | 254 ms | no |
|  | 220 | 404 | 56.8k | 0.32 | - | 0.85 | 1.15 | 273 ms | no |
| laptop b | 150 | 280 | 45.0k | 0.22 | 0.76 | 0.68 | 1.03 | 187 ms | no |
|  | 185 | 343 | 52.9k | 0.32 | - | 0.73 | 1.14 | 215 ms | no |
|  | 220 | 405 | 60.9k | 0.47 | - | 0.85 | 1.22 | 350 ms | no |
| laptop c | 190 | 350 | 65.7k | 0.60 | 0.64 | **2.35** | 2.71 | 223 ms | no |
|  | 225 | 403 | 76.2k | 0.45 | - | **2.84** | 3.22 | 279 ms | no |
| default c | 190 | 348 | 47.3k | 0.75 | 0.77 | 0.80 | 1.06 | 236 ms | no |
|  | 225 | 413 | 55.1k | 0.55 | - | 0.77 | 1.06 | 252 ms | no |

- With no subscriber rtabmap's `Maps update` + `pub` is 1-4 ms per step. The step right after an attach costs **172-350 ms more,
  about 0.6-0.9 ms per graph node**, and **every re-attach pays it again** (`map_cleanup` true, the default, clears the cache when the
  last subscriber leaves). At 2 nodes/s that is about 1.2-1.7 s in one step after 15-20 minutes of mapping.
- `slam_stale` appeared in none of the 6 runs. 4 of 16 attaches had a gap of 1.0 s or more in the 10 s after them. The two laptop c
  gaps (2.35, 2.84 s) are probably mostly input stalls of the 640x480 harness (the laptop runs show 2-2.9 s gaps with `camera_stale`
  when no viewer is attached too; known since the Task 8 risk check), so not all of the 4 are the attach itself.

**Decision:** the controller's threshold was "apply the `map_assembler` fallback if the worst gap on attach is >= 1.0 s OR `slam_stale`
appears on attach in any run". The gap condition is met (1.07 s in a `default` run, where the harness shows no input stalls), and the
attach cost grows linearly with the map, so the **fallback is applied**: `rtabmap_util/map_assembler` (`/rtabmap/assembler/map_assembler`,
same `rtabmap_rgbd.yaml`) builds `/rtabmap/cloud_map` from `/rtabmap/mapData` in its own process; rtabmap's own copy is remapped to
`/rtabmap/slam/cloud_map`, which nothing may subscribe (`localization.launch.py`).

### After: `map_assembler` (20 attaches, 7 runs)

First with the assembler's defaults (`map_cleanup` true, 5 runs), then with `map_cleanup: false` on the assembler only (2 runs, the
shipped configuration). "Gap before" = worst gap in the 20 s before the attach (no viewer). "Assembler s" = its update + publish time
for the message after the attach.

| Run | Attach at s | Graph node | Cloud | Cloud after s | Gap before s | Gap after s | Stamp age after s | rtabmap maps+pub | Assembler s | `slam_stale` |
|---|---|---|---|---|---|---|---|---|---|---|
| default a | 150 | 276 | 40.1k | 1.90 | 0.62 | 0.61 | 0.91 | 5 ms | 1.58 | no |
|  | 185 | 341 | 48.3k | 1.99 | 0.61 | 0.61 | 0.90 | 4 ms | 1.80 | no |
|  | 220 | 406 | 56.0k | 2.25 | 0.60 | 0.60 | 0.94 | 3 ms | 2.14 | no |
| laptop a | 150 | 269 | 47.5k | 1.70 | 0.55 | 0.64 | 1.24 | 8 ms | 1.36 | no |
|  | 185 | 335 | 56.9k | 2.09 | 0.67 | 0.57 | 0.94 | 4 ms | 1.70 | no |
|  | 220 | 402 | 65.4k | 2.43 | 0.65 | 0.65 | 1.00 | 5 ms | 2.04 | no |
| default b | 150 | 272 | 37.7k | 1.39 | 0.58 | 0.58 | 0.84 | 3 ms | 1.37 | no |
|  | 185 | 338 | 45.9k | 2.04 | 0.58 | 0.59 | 0.87 | 3 ms | 1.73 | no |
|  | 220 | 403 | 53.2k | 2.87 | 0.67 | 0.62 | 1.03 | 5 ms | 2.80 | no |
| laptop b | 150 | 268 | 47.2k | 1.51 | 0.55 | 0.73 | 1.06 | 4 ms | 1.37 | no |
|  | 185 | 336 | 55.2k | 2.34 | 0.61 | 0.71 | 1.04 | 4 ms | 1.72 | no |
|  | 220 | 403 | 63.1k | 2.26 | 0.55 | 0.71 | 1.07 | 4 ms | 2.05 | no |
| laptop c | 190 | 352 | 52.5k | 2.23 | 2.06 | 0.61 | 0.94 | 8 ms | 1.87 | no |
|  | 225 | 407 | 60.1k | 2.09 | 2.88 | 0.56 | 0.91 | 4 ms | 1.98 | no |
| default, `map_cleanup: false` | 150 | 275 | 39.3k | 1.85 | 0.61 | 0.57 | 0.83 | 10 ms | 1.46 | no |
|  | 185 | 340 | 47.0k | 0.61 | 0.60 | 0.60 | 0.88 | 4 ms | 0.19 | no |
|  | 220 | 405 | 54.8k | 0.50 | 0.59 | 0.65 | 0.94 | 5 ms | 0.19 | no |
| laptop, `map_cleanup: false` | 150 | 285 | 46.7k | 2.31 | 0.57 | 0.69 | 1.06 | 4 ms | 1.68 | no |
|  | 185 | 351 | 54.6k | 0.31 | 0.66 | 0.59 | 0.97 | 5 ms | 0.20 | no |
|  | 220 | 419 | 62.7k | 0.85 | 0.70 | 0.67 | 1.05 | 4 ms | 0.20 | no |

- SLAM no longer notices the viewer: worst gap after an attach 0.56-0.73 s, the same as before it; rtabmap's map work per step
  3-10 ms with the assembler permanently subscribed to `mapData`; `slam_stale` in none of the 7 runs. (The 2.06 and 2.88 s "gap before"
  in laptop c are harness input stalls with no viewer attached, with `camera_stale`.)
- The assembler is slower than rtabmap at the first assembly: about 5 ms per node (1.4-2.8 s at 270-406 nodes) against rtabmap's
  0.6-0.9 ms, and the first cloud reaches the viewer 1.4-2.9 s after the attach (0.2-0.75 s before the fix).
- Its `mapData` subscription keeps one message (`rclcpp::QoS(1)`, hard-coded in rtabmap_util 0.23.7). Messages arriving while it
  assembles are dropped, and with them those nodes' local maps (a node's data arrives only once). Whole runs, rtabmap steps minus
  messages the assembler processed: 4-7 per run with `map_cleanup` true (1-2 at every attach plus 1-2 at start-up), **3 per run with
  `map_cleanup: false`** (re-attaches cost 0.19-0.20 s and drop nothing; the cloud arrives 0.3-0.9 s after the attach). Hence
  `map_cleanup: false` on the assembler. Peak memory is the same either way, because the cache is only built while a viewer is attached.
- Memory (RSS sampled every 5 s, `map_cleanup: false` runs): map_assembler 195 -> 283 MB (`default`) and 197 -> 353 MB (`laptop`) during
  the first 150 s with no viewer (it keeps every node's data), 624-730 MB right after the first attach (the grid cache), 894 / 1062 MB at
  about 470 nodes. rtabmap itself: 1.20 / 1.34 GB at the same time.

### Found, not fixed

- The assembler drops nodes (above): a few local maps are missing from the viewer's cloud for the rest of the run. Display only; the
  elevation mapper subscribes `mapData` itself with a deep queue. Fixing it needs a deeper subscription in rtabmap_util (upstream) or a
  republish of the whole map through `/rtabmap/rtabmap/publish_maps`, which runs in rtabmap and would bring the stall back.
- Assembler memory grows with the map (about 1.5-1.8 MB per node once a viewer has been opened, from about 290 -> 470 nodes); check it on
  the live laptop on a long mission.
- `localize` mode not measured: the assembler loads the existing map once, 1 s after start, through `/rtabmap/rtabmap/get_map_data`
  (waiting at most 5 s for the service). If rtabmap takes longer to load its database, the assembler has only the nodes it receives later.

## Final review I2: map_assembler memory per node, `map_cleanup` false vs true

**When:** 2026-10-02, branch `mapping-3d`, `ugv-run` container, same harness as above. **Question** (final review I2, ruling R28):
how much memory do map_assembler and rtabmap take per graph node on the live profile, does `map_cleanup: true` (cache freed when
the viewer closes) bound it, and what mission length fits the laptop?

**Method:** `test_m1_measure_late_cloud_map_attach`, laptop profile only (`timing:=laptop`, 640x480, reliable camera, 4 frames/s),
which now also samples the resident memory (VmRSS from `/proc`) of `map_assembler` and `rtabmap` under the launch every 5 s,
with the graph node rtabmap had reached at that moment (`rss_mb` in the JSON line):
```
MSYS_NO_PATHCONV=1 docker exec -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -e UGV_MEASURE_LATE_ATTACH=1 -e UGV_MEASURE_MAP_S=120 \
  -e UGV_MEASURE_CYCLES=5 -e UGV_MEASURE_OUT=/tmp/i2.jsonl ugv-run bash -lc 'ulimit -c 0; \
  cd /ws/src/ugv_nav/ugv_localization && source /opt/ros/lyrical/setup.bash && source /ws/install/setup.bash && \
  python3 -m pytest -p no:cacheprovider test/test_ros_stack.py -k "m1 and laptop" -q'
```
120 s of mapping with no viewer, then 5 cycles of viewer attached 15 s / detached 20 s, about 295 s and 530-550 nodes per run.
One run with `map_cleanup: false` (shipped) and one with `map_cleanup: true` (the launch file's value changed for that run only).
The texture lasts 300 s of travel, so a run cannot be much longer without changing the harness.

### Memory against graph nodes (MB)

| Node | false: assembler | false: rtabmap | true: assembler | true: rtabmap | Phase (both runs) |
|---|---|---|---|---|---|
| ~5 | 193 | 348 | 188 | 289 | start-up, no viewer |
| ~110 | 266 | 856 | 263 | 862 | no viewer |
| 224 | 336 | 1008 | 336 | 1019 | no viewer, just before the first attach |
| ~252 | 716 | 1045 | 712 | 1054 | 1st attach (peak) |
| 281 | 716 | 1080 | 523 | 1087 | 1st detach |
| ~318 | 851 | 1128 | 844 | 1123 | 2nd attach (peak) |
| ~348 | 851 | 1164 | 591 | 1169 | 2nd detach |
| ~380 | 971 | 1211 | 959 | 1205 | 3rd attach (peak) |
| ~410 | 971 | 1240 | 697 | 1249 | 3rd detach |
| ~445 | 1065-1088 | 1272-1284 | 1124-1133 | 1287-1299 | 4th attach (peak) |
| ~475 | 1088 | 1319 | 782 | 1320 | 4th detach |
| ~505 | 1189-1205 | 1351-1362 | 1258-1282 | 1366-1381 | 5th attach (peak) |
| ~535 | 1205 | 1381-1392 | 840 | 1401-1412 | 5th detach |

Linear fits (MB = a + b x node):

| | `map_cleanup: false` | `map_cleanup: true` |
|---|---|---|
| map_assembler, no viewer yet (nodes 30-224) | 196 + 0.62/node | 197 + 0.62/node |
| map_assembler, peak while a viewer is attached | 237 + 1.92/node | 182 + 2.10/node |
| map_assembler, viewer closed | stays at the last peak (until the next attach adds the new nodes) | about 1.2/node (523 at 281 -> 840 at 549) |
| rtabmap (nodes >= 50) | 711 + 1.30/node | 714 + 1.30/node |
| both, peak with a viewer attached | 967 + 3.17/node | 932 + 3.31/node |

### Viewer and SLAM per attach

| Attach | Node false / true | false: first cloud after s | true: first cloud after s | false: assembler s | true: assembler s | false / true: worst `/rtabmap/info` gap after s | rtabmap maps+pub |
|---|---|---|---|---|---|---|---|
| 1 | 224 / 224 | 1.78 | 1.65 | 1.17 | 1.31 | 0.69 / 0.59 | 3-5 ms |
| 2 | 290 / 291 | 0.31 | 1.99 | 0.19 | 1.67 | 1.03 / 0.73 | 3-5 ms |
| 3 | 356 / 358 | 0.30 | 2.34 | 0.20 | 1.97 | 0.70 / 0.64 | 4-5 ms |
| 4 | 415 / 425 | 0.24 | 2.11 | 0.18 | 2.05 | 0.71 / 0.63 | 4-5 ms |
| 5 | 478 / 493 | 0.37 | 3.00 | 0.19 | 2.60 | 0.79 / 0.58 | 4-5 ms |

- `mapData` messages dropped by the assembler over the whole run (rtabmap steps minus messages it processed): **1** of 540 with
  `false`, **8** of 559 with `true` (1-3 at every re-attach, while it rebuilds the whole map).
- `slam_stale` in neither run. The 1.03 s gap after the second attach of the `false` run comes with `depth_stale` and the same run
  shows 1.8-2.2 s gaps with no viewer attached (`camera_stale`): harness input stalls of the 640x480 profile, known since the Task 8
  risk check. rtabmap's map work stayed at 3-5 ms per step after every attach in both runs.

**Decision (ruling R28 fallback):** `map_cleanup: true` does not bound the memory either. Its peak with a viewer open grows as fast
as with `false` (2.1 against 1.9 MB/node), and the cache it frees on detach is only partly given back (the assembler still grows
about 1.2 MB/node with the viewer closed, against 0.62 before any attach). It costs 1.7-3.0 s per re-open (5-10 times slower,
growing with the map) and 7 more nodes missing from the viewer cloud per run. The figure that limits a mission is the peak with the
viewer open, which the operator can reach at any time, so **`map_cleanup: false` stays**, and the limit is documented instead:
`docs/mapping/README.md` "Memory and mission length" (about 1500 nodes, about 12 minutes of continuous driving at 2 nodes/s,
with the map view; about 4000 nodes, about 30 minutes, with `map_assembler:=false`, the launch argument added for this). The
figures are from a synthetic plane at 3 m; a real scene may cost more per node, so re-measure in the owner's lit run.
