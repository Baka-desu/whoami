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
