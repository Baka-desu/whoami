# Dev 2 interface contract

Topic names marked **TBD** are proposals until the owning dev confirms; every one is a launch argument.
QoS: Dev 2 subscribes to camera / CameraInfo / depth **best-effort**, so it connects to reliable or best-effort
publishers alike (verified end-to-end with a best-effort camera, `test/test_ros_stack.py`).

## Inputs (Dev 2 consumes)

| Topic / TF | Type | From | Rule |
|---|---|---|---|
| `/camera/image_raw` (**TBD**) | `sensor_msgs/Image` | Dev 5 | stamp = exposure time; `frame_id` = optical frame (`camera_optical_frame`) |
| `/camera/camera_info` (**TBD**) | `sensor_msgs/CameraInfo` | Dev 5 | **one per image, identical stamp** + frame (rgbd_sync pairs RGB + depth + CameraInfo by exact stamp; a latched-once CameraInfo never pairs — Dev 1's transient-local subscriber accepts either); real K (zero K → `camera_info_invalid`) |
| `/perception/depth_cloud` | `sensor_msgs/PointCloud2` | **Dev 1** (DA3) | see **Depth input** below (exists in Dev 1's code) |
| `/wheel/odom` (**TBD**) | `nav_msgs/Odometry` | Dev 5 diff-drive (sim plugin / motor driver) | `frame_id=odom`, `child_frame_id=base_link`, pose covariance filled, ≥ 15 Hz, **no TF**. Optional with `odom_source:=visual` |
| `/tf_static` `base_link->camera_link->camera_optical_frame` | TF | Dev 5 URDF / robot_state_publisher | camera extrinsics (rgbd_odometry + RTAB-Map need them) |
| `/clock` | `rosgraph_msgs/Clock` | Dev 5 sim / `ros2 bag play --clock` | `profile:=sim|bag` → `use_sim_time`. Never recorded into eval bags |
| `/ground_truth/odom` (**TBD**, eval only) | `nav_msgs/Odometry` | Dev 5 sim | drift benchmark only; never used by the product path |
| `/camera/depth/image_raw` (**TBD**, eval/bring-up only) | `sensor_msgs/Image` 32FC1 | Dev 5 sim depth camera | same pose, size and `frame_id` as the RGB camera; DA3 accuracy benchmark + bring-up before DA3 is live |

## Depth input (Dev 1 point cloud — what his code publishes today)

Dev 1 publishes Depth Anything 3 depth as a **point cloud**; Dev 2 converts it back into the depth image RTAB-Map
RGB-D needs with `rtabmap_util/pointcloud_to_depthimage` (`config/cloud_to_depth.yaml`, launch `depth_input:=cloud`,
the default). Dev 1 back-projects with the **raw camera K at camera resolution**, so re-projecting with the same
CameraInfo puts each point back on its own pixel — no loss; dropped sky/hole points stay empty (0 = no depth).

Contract, checked against `turing/src/ugv_perception` on `main` / `dev1-turing-perception` (2026-09-29):

| Field | Value | In Dev 1 code today |
|---|---|---|
| Topic | `/perception/depth_cloud` | yes — `node/adapter_node.py:132` |
| Type | `sensor_msgs/PointCloud2`, fields `x,y,z` float32, `point_step 12`, `height 1` (unorganized), `is_dense false` | yes — `node/cloud.py` |
| Units / frame | meters, camera **optical** frame (x right, y down, z forward) | yes — `depth/geometry.py:backproject` with the raw `K` |
| `header.stamp` | the **source image stamp**, never re-stamped (rgbd_sync pairs RGB + depth + CameraInfo by exact stamp) | yes — `frame.stamp_ns` |
| `header.frame_id` | the source image optical `frame_id` (= `CameraInfo.frame_id`) | yes — `frame.frame_id` |
| Holes / sky | points dropped | yes — `valid_mask` + `hole_safe_resize` |
| QoS | reliable, depth 10 (Dev 2 subscribes best-effort: compatible) | yes |
| Enabled | only when `turing/weights/da3metric-large.xml` exists; `depth.yaml enabled` is never read | **weights not in the repo** → no cloud until Dev 1 fetches + exports DA3 |
| Rate | one cloud per published mask, inside the image callback (DA3 blocks it; frames dropped meanwhile) | not measured |

The converted image (`/rtabmap/depth/image`, 32FC1 m) is what `rgbd_sync`, `pose_validity` and `depth_eval` see.
Dev 2 checks it at runtime (`depth/gate.py`): wrong encoding / frame / size, future stamp or coverage below
`min_depth_coverage` → `/ugv/pose_valid=false` (`depth_invalid`); nothing for `depth_max_age_s` → `depth_stale`.
`depth_input:=image depth_topic:=<topic>` bypasses the conversion (sim ground-truth depth camera, DA3 benchmark).

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
| `/rtabmap/info` | `rtabmap_msgs/Info` | eval | RTAB-Map native |
| `rtabmap.db` | file | Dev 2 localize mode | `~/.ros/ugv/rtabmap.db` by default |

**Not published:** `/cmd_vel*`, anything from the perception mask.

## Requests to other devs

**Dev 1 (perception)**
1. Keep the cloud contract above (source stamp, optical frame, raw-K back-projection at camera resolution) — Dev 2's
   conversion is exact only while that holds. Tell Dev 2 before changing resolution, K, or frame.
2. Fetch + export the DA3 weights (`fetch_da3metric_large.sh`, `export_da3metric_openvino.py`): without the IR no cloud is published.
3. Share DA3 latency / fps — sets `sync_queue_size`, `depth_max_age_s`, and whether visual odom meets 15 Hz.
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
