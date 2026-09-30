# Dev 2 interface contract

Topic names marked **TBD** are proposals until the owning dev confirms; every one is a launch argument.
QoS: Dev 2 subscribes to camera / CameraInfo / depth **best-effort**, so it connects to reliable or best-effort
publishers alike (verified end-to-end with a best-effort camera, `test/test_ros_stack.py`).

## Inputs (Dev 2 consumes)

| Topic / TF | Type | From | Rule |
|---|---|---|---|
| `/camera/image_raw` (**TBD**) | `sensor_msgs/Image` | Dev 5 | stamp = exposure time; `frame_id` = optical frame (`camera_optical_frame`) |
| `/camera/camera_info` (**TBD**) | `sensor_msgs/CameraInfo` | Dev 5 | **one per image, identical stamp** + frame (rgbd_sync pairs RGB + depth + CameraInfo by exact stamp; a latched-once CameraInfo never pairs — Dev 1's transient-local subscriber accepts either); real K (zero K → `camera_info_invalid`) |
| `/perception/depth/image` | `sensor_msgs/Image` 32FC1 m | **Dev 1** (DA3) | primary — see **Depth input** below |
| `/perception/depth_cloud` | `sensor_msgs/PointCloud2` | **Dev 1** (DA3) | fallback (`depth_input:=cloud`) |
| `/wheel/odom` (**TBD**) | `nav_msgs/Odometry` | Dev 5 diff-drive (sim plugin / motor driver) | `frame_id=odom`, `child_frame_id=base_link`, pose covariance filled, ≥ 15 Hz, **no TF**. **Not needed by default** (`odom_source:=visual`); only for `auto` / `wheel` |
| `/tf_static` `base_link->camera_link->camera_optical_frame` | TF | Dev 5 URDF / robot_state_publisher | camera extrinsics (rgbd_odometry + RTAB-Map need them) |
| `/clock` | `rosgraph_msgs/Clock` | Dev 5 sim / `ros2 bag play --clock` | `profile:=sim|bag` → `use_sim_time`. Never recorded into eval bags |
| `/ground_truth/odom` (**TBD**, eval only) | `nav_msgs/Odometry` | Dev 5 sim | drift benchmark only; never used by the product path |
| `/camera/depth/image_raw` (**TBD**, eval/bring-up only) | `sensor_msgs/Image` 32FC1 | Dev 5 sim depth camera | same pose, size and `frame_id` as the RGB camera; DA3 accuracy benchmark + bring-up before DA3 is live |

## Depth input (Dev 1 — `af7ebbf` "Publish a metric depth Image for Dev 2", branch `dev1-turing-perception`)

**Primary (launch default `depth_input:=image`):** `/perception/depth/image`. Checked against Dev 1's code 2026-09-29,
and his real `depth_to_image()` output passes Dev 2's depth gate (stamp preserved to the ns):

| Field | Required | Dev 1 code |
|---|---|---|
| Type / encoding | `sensor_msgs/Image`, `32FC1`, meters | ✓ `node/cloud.py:depth_to_image` |
| Holes / sky | `NaN` | ✓ `hole_safe_resize` output kept as-is |
| Size | camera `width × height` (same as the RGB) | ✓ resized to the camera size before publishing |
| `header.stamp` | the RGB image's stamp (rgbd_sync pairs RGB + depth + CameraInfo by exact stamp) | ✓ `frame.stamp_ns` |
| `header.frame_id` | the RGB image's optical frame (= `CameraInfo.frame_id`) | ✓ `frame.frame_id` |
| QoS | reliable, depth 10 (Dev 2 subscribes best-effort: compatible) | ✓ |
| Enabled | only when `turing/weights/da3metric-large.xml` exists | **weights not in the repo yet** |
| Rate | one depth image per published mask (same DA3 infer as the cloud) | latency / fps not measurable yet (static test images) |

**Fallback (`depth_input:=cloud`):** `/perception/depth_cloud` (PointCloud2) → `rtabmap_util/pointcloud_to_depthimage`
(`config/cloud_to_depth.yaml`). Exact only while the cloud is back-projected with the raw K at camera resolution.

Dev 2 checks the depth image at runtime (`depth/gate.py`): wrong encoding / frame / size, future stamp or coverage below
`min_depth_coverage` → `/ugv/pose_valid=false` (`depth_invalid`); nothing for `depth_max_age_s` → `depth_stale`.
Sim ground-truth depth camera for bring-up / DA3 benchmark: `depth_topic:=<its topic>`.

## Outputs (Dev 2 publishes)

| Topic / TF | Type | To | Rule |
|---|---|---|---|
| TF `odom->base_link` | TF | Dev 3, 4, 5 | `odom_selector` only; stamp = selected odometry stamp; continuous across source switches |
| TF `map->odom` | TF | Dev 3, 4, 5 | RTAB-Map, 20 Hz (`tf_delay 0.05`) |
| `/odom` | `nav_msgs/Odometry` | Dev 4 (controller velocity), RTAB-Map | selected source (wheel or visual), re-anchored; twist passed through |
| `/ugv/localization/odom_source` | `std_msgs/String` | humans / eval / pose_validity | `wheel` or `visual`; latched, on change |
| `/map` | `nav_msgs/OccupancyGrid` | Dev 3 (**optional**) | RTAB-Map grid from DA3 depth, 5 cm, range ≤ 5 m. Quality = DA3 quality (SENSOR_HONESTY.md) |
| `/ugv/pose_valid` | `std_msgs/Bool` | **Dev 5** level-3 hold | 20 Hz always (heartbeat); `false` at startup and on any failure; `true` only after `recover_hold_s` clean |
| `/ugv/localization_status` | `std_msgs/String` | humans / eval | comma-separated reasons (`tf_stale,depth_stale,…`) or `valid`; on change |
| `/ugv/localization/distance_travelled` | `std_msgs/Float64` | humans / eval / Dev 4 testbench | metres driven along `/odom` since start or reset, 5 Hz always. An **odometry estimate**, not surveyed: with no wheel sensor it is visual odometry, so DA3 scale bias = distance bias; motion while VO is lost is not counted; jitter < 5 cm and jumps > 3 m/s are ignored (`config/distance.yaml`) |
| `/ugv/localization/distance_basis` | `std_msgs/String` | whoever reads the distance | what the total is made of: `none`, `visual_odometry_estimate`, `wheel_odometry`, `odometry_estimate` (mixed / unattributed); latched, on change |
| `/ugv/localization/reset_distance` | `std_srvs/Empty` (service) | Dev 4 / operator | zero the total, e.g. at each new goal |
| `/rtabmap/info` | `rtabmap_msgs/Info` | eval | RTAB-Map native |
| `rtabmap.db` | file | Dev 2 localize mode | `~/.ros/ugv/rtabmap.db` by default |

**Not published:** `/cmd_vel*`, anything from the perception mask.

## TF for Dev 3 (answers 2026-09-29; measured on the real stack with synthetic sensors unless marked)

| Question | Answer |
|---|---|
| TF tree / frame names | `map → odom → base_link` (Dev 2) → `camera_link → camera_optical_frame` (Dev 5 URDF, static). `map`, `odom`, `base_link` are fixed in Dev 2 configs. Camera frame names are Dev 5's (proposal above) — **read `header.frame_id` from the mask / depth message, don't hard-code it**. `odom_visual` is a label, never a TF frame. |
| `odom → base_link` rate | = the selected odometry's rate, stamped with the odometry message stamp (never "now"). `auto`/`wheel`: Dev 5 wheel odom rate (contract ≥ 15 Hz). `visual`: the DA3 depth rate (not measured; may be < 15 Hz). Measured: arrives ~6 ms after its stamp. Continuous across source switches (no jump). |
| `map → odom` rate / behavior | RTAB-Map, **20 Hz** (`tf_delay 0.05`), measured 20.1 Hz. Stamped **~+100 ms in the future** (`tf_tolerance 0.1`) so lookups at "now" succeed. Its *value* changes only when RTAB-Map processes a frame (`Rtabmap/DetectionRate 2` Hz) and jumps on loop closure / relocalization (Dev 2 holds `/ugv/pose_valid=false` for 1 s after a jump > 1 m / 0.35 rad). Identity until the first frame; in `localize` identity until relocalized (pose invalid meanwhile). |
| TF lookup timeout | Look up `camera_optical_frame → costmap frame` **at the mask stamp**, never "latest". Wheel odom at ≥ 15 Hz: the TF for a stamp exists ≤ ~70 ms after capture, and masks arrive later than that (segmentation latency), so the lookup normally succeeds immediately — use **0.1 s** timeout. `odom_source:=visual`: TF follows DA3 latency — use up to **0.5 s** (= Dev 5 watchdog), drop the mask on timeout. Keep the TF buffer ≥ 10 s (default). Nav2 `transform_tolerance` 0.3–0.5 s. Revisit once latency is measured. |
| Stamp / latency perception ↔ TF | Every perception message (mask, depth image, cloud) carries the **RGB capture stamp**. `odom → base_link` carries wheel (or visual) odometry stamps — tf2 interpolates to the mask stamp. `map → odom` is future-stamped. Absolute latency (capture → mask, capture → depth) is **not measurable yet** (static test images); measure with a sim/bag stream. |
| `base_link` vs `base_footprint` | Dev 2 uses **`base_link`** as the robot frame (RTAB-Map `frame_id`, odom `child_frame_id`, 3-DoF ground robot); **no `base_footprint`** in the chain today. Where `base_link` sits (ground level or axle height) is Dev 5's URDF — recommend **`base_link` at ground level** (Grid heights are relative to it). If Dev 5 adds `base_footprint`, Dev 2 switches its robot frame to it (config-only). |
| `rtabmap_util` dependency | Yes — `package.xml` `exec_depend rtabmap_util` (added 2026-09-29; needed for `depth_input:=cloud`). Also `rtabmap_slam`, `rtabmap_sync`, `rtabmap_odom`, `rtabmap_msgs`. |

## Requests to other devs

**Dev 1 (perception)**
1. Merge `af7ebbf` (depth image) to `main`. Keep its stamp / frame / size = the RGB image's.
2. Fetch + export the DA3 weights (`fetch_da3metric_large.sh`, `export_da3metric_openvino.py`): without the IR nothing is published.
3. DA3 latency / fps once a camera stream exists (sim or bag; static test images can't measure it) — sets
   `sync_queue_size`, `depth_max_age_s`, and whether visual odom meets 15 Hz.
4. Optional: publish depth even when a mask is skipped (today depth only follows a published mask).

**Dev 5 (platform/sim)**
1. Confirm camera / wheel-odom / ground-truth topic names and optical `frame_id`. Publish CameraInfo **with every image, same stamp**.
2. Diff-drive: publish `/wheel/odom` with covariance, `publish_odom_tf=false` (Gazebo `DiffDrive` plugin: do not bridge its TF; motor driver: don't broadcast).
3. Sim: a **ground-truth depth camera** co-located with the RGB camera (same intrinsics / size / frame) — DA3 benchmark and bring-up.
4. Sim world: textured (feature-rich ground, walls, objects) — visual odometry and loop closure need features.
5. Sim ground-truth pose topic for drift benchmarks.
6. Safety mux: treat missing `/ugv/pose_valid` messages (> watchdog timeout) the same as `false`.

**Dev 3 (costmaps)**
1. `/map` from Dev 2 is **optional** and only as good as DA3 depth; don't make the global costmap depend on it.
2. Use `map` (global) and `odom` (local) frames from Dev 2's TF.
3. In `odom_source:=visual` the `odom->base_link` rate is the DA3 depth rate (CONFLICTS.md C9).

**Dev 4 (planning)**
1. `/odom` (selected, re-anchored) is the odometry topic for the controller.
2. Goals in `map` frame; valid only while `/ugv/pose_valid` is true (Dev 5 enforces).
