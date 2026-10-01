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
