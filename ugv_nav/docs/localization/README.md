# Dev 2 — SLAM & Localization (`ugv_localization`)

**Authority:** [`architecture.md`](../../../architecture.md) wins over [`dev.md`](../../../dev.md) on conflict.
**Decisions:** [`mindmap.md`](../../../mindmap.md) (D1–D8). **Gaps vs architecture/dev.md:** [CONFLICTS.md](CONFLICTS.md).
**Contract:** [interfaces.md](interfaces.md). **Environment:** [ENVIRONMENT.md](ENVIRONMENT.md). **Honesty:** [SENSOR_HONESTY.md](SENSOR_HONESTY.md).

## What Dev 2 is

The **pose half of the brain** (architecture §2): RTAB-Map + the TF chain + an honest "is the pose usable" signal.
Dev 2 does **not** own the camera (Dev 5), does **not** run the depth network (Dev 1), does **not** consume the
perception mask (§5), and does **not** plan (Dev 4).

Sensor (2026-09-28, D7): **one mono camera; depth comes from Depth Anything 3 Metric Large**, run by Dev 1 and
published as a 32FC1 depth image (`/perception/depth/image`; the point cloud is a fallback). RTAB-Map runs in
**RGB-D mode** on camera RGB + DA3 depth. Odometry is switchable (D5b).

```
Dev 5 camera ── Image + CameraInfo ─────────────┬──────────────────────────────┐
Dev 1 DA3   ── /perception/depth/image (32FC1) ─┤   (fallback: /perception/depth_cloud ─► cloud_to_depth)
                                                ▼                              │
                  rgbd_sync ── /rtabmap/rgbd_image (exact stamps)              │
                      │                     │                                  │
                      ▼                     ▼                                  │
     rgbd_odometry (auto|visual)       rtabmap (RGB-D) ── map->odom TF, /map, /rtabmap/info, rtabmap.db
           │ /rtabmap/odom_visual           ▲ /odom                            │
           ▼                                │                                  │
Dev 5 /wheel/odom ──► odom_selector ── odom->base_link TF + /odom              │
                           │ /ugv/localization/odom_source                     │
                           ▼                                                   ▼
                      pose_validity_node ── /ugv/pose_valid (20 Hz heartbeat, fail closed) → Dev 5
                      distance_tracker ──── /ugv/localization/distance_travelled (+ distance_basis label)
```

`odom_source:=visual` (default since 2026-09-29 — there is no wheel sensor yet) takes odometry from the camera alone
(RTAB-Map `rgbd_odometry` on RGB + DA3 depth); **no wheel odometry is needed**. `auto` uses wheel odom when it exists and
falls back to visual; `wheel` forces wheel. The selector re-anchors on every switch, so `odom` never jumps.
Don't feed simulated wheel odom alongside real images: the two motions won't agree.

## Laws

| Law | Meaning here |
|---|---|
| Mono is the minimum, not the recommendation (§6, §10, kill list) | DA3 pseudo-depth is **not** RGB-D-grade; drift + depth error documented; never marketed as outdoor-meter-ready |
| Pose valid only if valid (§10.1) | TF present + fresh, odom sane, camera + depth fresh and in contract, RTAB-Map alive, localized, drift budget |
| Fail closed | `/ugv/pose_valid=false` at startup, on any missing input, on clock reset, briefly after an odom source switch |
| Never restamp | TF edges carry the odometry stamp, never "now"; depth keeps the source image stamp |
| No fake sensors | Camera YAML only from a real CameraInfo / calibration; tests use numeric tables |
| One owner per TF edge | Dev 2 owns `map->odom->base_link` (D6); only `odom_selector` publishes `odom->base_link` |

## Build order and status (2026-09-28)

| # | Module | Code | Status |
|---|---|---|---|
| L1 | Camera calibration validate/load | `camera/` | **done, tested** |
| L2 | Odom gates + source selector → odom->base_link | `odom/`, `nodes/odom_selector.py`, `config/odom_select.yaml` | kernel **tested**; node **live-tested on Lyrical** with synthetic odom: wheel → visual → wheel, no jump |
| L3 | RTAB-Map RGB-D config + launch | `config/rtabmap_rgbd.yaml`, `rgbd_odometry.yaml`, `rgbd_sync.yaml`, `launch/localization.launch.py` | **launches** for all 3 `odom_source`; all 29 library + 25 node param names verified on live nodes; values untuned (needs camera + depth) |
| L4 | mapping / localize modes | `modes/`, `nodes/mode_cli.py` | **runs**: fail-fast, db save, localize relaunch, runtime switch |
| L5 | Pose validity (+ depth contract, switch hold) | `validity/`, `depth/gate.py`, `nodes/pose_validity_node.py` | **runs**: false @ 20 Hz with reasons; `true` path + `depth_stale` verified end-to-end on synthetic sensors (`test_ros_stack.py`); real camera + DA3 pending |
| L6 | TF rate/jitter check | `tfcheck/`, `nodes/tf_rate_check.py` | **runs** (odom->base_link); map->odom needs camera |
| L7 | Bag harness | `launch/bag_eval.launch.py`, `scripts/record_eval_bag.sh` | written (records DA3 + GT depth, no `/clock`); needs a recorded bag |
| L8 | Drift + depth metrics | `drift/`, `depth/metrics.py`, `tools/drift_report.py`, `nodes/drift_eval.py`, `nodes/depth_eval.py` | kernels **tested**; nodes need GT + map->odom / GT depth |
| L5b | Distance travelled (odometry estimate) | `odom/distance.py`, `nodes/distance_tracker.py`, `config/distance.yaml` | kernel **tested**; node **tested over ROS** (count, label, reset) |
| L9 | Sensor honesty numbers | [SENSOR_HONESTY.md](SENSOR_HONESTY.md) | protocol written; **no numbers until sim runs** |

Blockers: **Dev 1** DA3 weights exported + `af7ebbf` (depth image) merged — publishers only start when the IR is present,
**Dev 5** sim camera + GT depth camera + `/wheel/odom` + ground truth.
Until DA3 publishes, bring up on the sim ground-truth depth camera: `depth_topic:=<sim depth topic>`.

Tests: 244 under `colcon test` — kernels, every node over ROS (`test_ros_nodes.py`), and the full launch with real RTAB-Map on synthetic sensors (`test_ros_stack.py`). See ENVIRONMENT.md.

## Run the tests (no ROS needed)

```bash
cd ugv_nav/ugv_localization
python -m pip install numpy pyyaml pytest
python -m pytest            # ROS smoke tests auto-skip without rclpy/launch_ros
```

With ROS: `colcon build --packages-select ugv_localization && colcon test --packages-select ugv_localization`.

## Run the stack (once ROS + Dev 5 sim + Dev 1 depth exist)

```bash
ros2 launch ugv_localization localization.launch.py mode:=mapping profile:=sim fresh_db:=true
# drive the robot around a loop, Ctrl-C → ~/.ros/ugv/rtabmap.db saved
ros2 launch ugv_localization localization.launch.py mode:=localize profile:=sim
ros2 launch ugv_localization localization.launch.py mode:=localize profile:=sim odom_source:=visual
ros2 topic echo /ugv/pose_valid
ros2 topic echo /ugv/localization_status
ros2 topic echo /ugv/localization/odom_source
ros2 topic echo /ugv/localization/distance_travelled   # estimate; see distance_basis
ros2 service call /ugv/localization/reset_distance std_srvs/srv/Empty
ros2 run ugv_localization tf_rate_check --ros-args -p use_sim_time:=true
ros2 run tf2_ros tf2_echo map base_link
```

Re-check parameter names after any rtabmap upgrade (stack running):

```bash
rtabmap --params > /tmp/lib.txt
ros2 param list /rtabmap/rtabmap > /tmp/slam.txt
ros2 param list /rtabmap/rgbd_odometry > /tmp/vo.txt
ros2 param list /rtabmap/rgbd_sync > /tmp/sync.txt
C=$(ros2 pkg prefix ugv_localization)/share/ugv_localization/config
ros2 run ugv_localization check_rtabmap_params --dump /tmp/lib.txt \
  --node-check $C/rtabmap_rgbd.yaml=/tmp/slam.txt --node-check $C/rgbd_odometry.yaml=/tmp/vo.txt \
  --node-check $C/rgbd_sync.yaml=/tmp/sync.txt
```
