# Sensor honesty — mono camera + Depth Anything 3 depth (architecture §10, DoD item 7)

**No numbers in this file are measured yet.** Results rows are filled only from `drift_eval` / `depth_eval` output of real runs.

## What mono + DA3 depth can and cannot do

| Aspect | Mono + DA3 Metric depth (what we ship) | Real stereo / RGB-D (recommended, §6) |
|---|---|---|
| Depth source | Learned, **per frame**, from one image; scale from the network + focal length (a wrong K = wrong meters) | Measured (triangulation / active light) |
| Depth error vs range | Grows with range; unknown on our terrain until `depth_eval` runs → capped by `Vis/MaxDepth 8 m`, `Grid/RangeMax 5 m` | Grows with range, but bounded by sensor physics |
| Temporal consistency | Frame-to-frame depth can flicker / rescale → visual odometry jitter, scale drift | Consistent |
| Metric scale | DA3 metric scale; with `odom_source:=wheel|auto` also wheel odom — the two can **disagree** (scale bias → biased loop-closure constraints) | From the camera itself |
| Rate | DA3-Large on one GPU: **not measured**; visual odometry and `odom->base_link` (visual mode) run at this rate (CONFLICTS C9) | Camera rate |
| Odometry source (default: camera only, no wheel sensor) | Visual odometry = DA3 rate; lost on low texture / fast turns / depth dropout → no odometry → hold (no fallback without wheels) | Visual odometry at camera rate |
| Odometry when wheels slip (mud, grass, sand) | `auto` keeps wheel while it publishes (slip is silent!); `visual` tracks through slip if texture + depth hold | Visual odometry keeps tracking |
| Distance travelled (`/ugv/localization/distance_travelled`) | Visual-odometry estimate: DA3 scale bias → same % distance bias; lost-VO stretches not counted (under-reports); `distance_basis` = `visual_odometry_estimate`. Not measured yet | From calibrated depth / wheel encoders |
| Loop closure / relocalization | Yes (bag-of-words + depth-backed 3D words) | Yes |
| Occupancy grid `/map` | Yes, from DA3 depth — quality = DA3 quality; optional for Dev 3 | Yes |
| Sky, glass, water, thin branches | NaN or wrong depth; coverage below `min_depth_coverage` → `depth_invalid` hold | Also hard (glass/water), better on thin structure |
| Low texture (flat dirt, uniform grass) | Few features → VO lost → `auto` falls back to wheel; `visual` holds | Also weak |
| Lighting change (sun/shade, dawn vs noon) | Relocalization against a map from different lighting may fail; DA3 depth also shifts with lighting | Appearance-based loop closure has the same issue |

Product consequence: `/ugv/pose_valid` goes false on missing / off-contract depth, on visual-odom loss with no
wheel fallback, after an odom source switch (`odom_switch_hold_s`), and in localize mode after
`max_dead_reckon_m` without a map constraint. This setup **will** hold more often than a real RGB-D sensor would;
that is the honest outcome.

## Measurement protocol (sim first, real later)

1. Record: `scripts/record_eval_bag.sh eval_bags/<run>` while driving a loop — records RGB, CameraInfo, DA3 depth,
   sim GT depth, `/wheel/odom`, ground truth, `/tf_static` (no `/clock`, no `/tf`).
2. **Depth accuracy:** replay the bag with
   `ros2 run ugv_localization depth_eval --ros-args -r gt_depth:=<sim depth topic> -p use_sim_time:=true -p out_dir:=eval_out/<run>_depth`
   → AbsRel, RMSE, δ<1.25, coverage, per-range AbsRel, median GT/DA3 scale. Use per-range AbsRel to set `Vis/MaxDepth` / `Grid/RangeMax`.
3. **Drift matrix** — for each `odom_source ∈ {wheel, visual, auto}` × depth `∈ {sim GT depth image, DA3 depth image}`:
   `ros2 launch ugv_localization bag_eval.launch.py bag:=eval_bags/<run> mode:=mapping fresh_db:=true odom_source:=… [depth_topic:=<gt depth topic>]`
   with `ros2 run ugv_localization drift_eval --ros-args -r ground_truth:=/ground_truth/odom -p use_sim_time:=true -p out_dir:=eval_out/<run>_<src>_<depth>`.
   Repeat with `-p with_scale:=true` to separate scale drift from shape drift.
4. **Fallback:** an `auto` run with `/wheel/odom` removed mid-bag (e.g. `ros2 bag play --topics` excluding it for a segment) —
   `/ugv/localization/odom_source` must flip and `tf2_echo odom base_link` must not jump.
5. Localize: replay a **different** run with `mode:=localize` against the mapping db; same `drift_eval`.
6. Record the pose_valid duty cycle (% time true) from `/ugv/localization_status`, and `tf_rate_check` per odom source.

## Results — depth (DA3 vs sim GT)

| Run | Frames | Coverage | AbsRel | RMSE m | δ<1.25 | AbsRel 0–2 / 2–4 / 4–8 / 8+ m | Median GT/DA3 | Notes |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | not yet measured |

## Results — drift

| Run | odom_source | Depth | Mode | Path m | ATE RMSE m | RPE 10 m % | Endpoint drift % | pose_valid % | odom->base_link Hz | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | — | not yet measured |
