# 3D mapping and the web map view

What was built on branch `mapping-3d` (plan: `docs/superpowers/plans/2026-10-02-3d-mapping-elevation.md`), how to run it
and what it can and cannot tell you. Where the plan and the code disagree, this document follows the code.

**Status.** The map, live cloud, trajectory, cost grid, depth and camera layers and the statistics widget are built and
tested on synthetic data and on the real stack with synthetic sensors. **Nothing has run on the UGV yet** (see "Pending
owner runs"). The **elevation map (plan Tasks 10-12) is not built yet**: the owner deferred it. The gateway endpoint, the
decoder and the viewer's ELEV layer exist, but nothing publishes elevation, so the layer stays empty ("no map yet", the
endpoint answers 503). Nothing here feeds Nav2: the map is mapping and display only (mindmap D9, architecture §9 unchanged).

## Data flow

```
phone camera -> tunnel -> ugv_bringup camera driver -- /camera/image_raw + /camera/camera_info --> Dev 1 perception
                                   |                                                              (DA3 depth, SegFormer mask)
                                   +-- /image_raw/compressed (camera panel)                        /perception/depth/image (32FC1 m)
                                                                                                   /ugv/perception/stats (JSON, 1 Hz)
                                                                                                          |
        rgbd_sync (RGB + depth, exact stamps) -> rgbd_odometry -> odom_selector -> /odom                  |
                                         |                                                                |
                                         v                                                                |
        RTAB-Map (rgbd, Grid/3D true): TF map->odom, /rtabmap/info, /rtabmap/cloud_map, /rtabmap/mapPath,
                                       /rtabmap/mapGraph, /rtabmap/mapData
                                         |
        map_stats node (reads /rtabmap/mapGraph + /rtabmap/info) -> /ugv/map/stats
                                         |
                                         v
        ugv_api gateway (on-demand subscriptions, TF poll map->base_link) -- GET /api/v1/map/... binary v1 --> web UI map view (three.js)
```

RTAB-Map is the only fusion engine. The gateway and UI add no mapping; they reshape what RTAB-Map, perception and Nav2's
global costmap already publish. `rtabmap` builds `cloud_map`, `mapPath` and `mapData` only while something subscribes, so
the gateway subscribes to the heavy topics only while a client keeps calling `GET /api/v1/map`.

## Topics

| Topic | Type | Publisher | Consumer | Notes |
|---|---|---|---|---|
| `/rtabmap/cloud_map` | `sensor_msgs/PointCloud2` | `rtabmap` | gateway (on demand) | whole coloured 3D map, voxelised at 5 cm, republished every SLAM step while subscribed. Reliable, transient local |
| `/rtabmap/mapPath` | `nav_msgs/Path` | `rtabmap` | gateway (on demand) | optimised graph poses in `map` |
| `/rtabmap/mapGraph` | `rtabmap_msgs/MapGraph` | `rtabmap` | `map_stats` | cheap, transient local |
| `/rtabmap/mapData` | `rtabmap_msgs/MapData` | `rtabmap` | none today | meant for the elevation mapper (not built); contract in `ugv_nav/docs/localization/interfaces.md` |
| `/ugv/map/stats` | `std_msgs/String` (JSON) | `map_stats` (`ugv_localization`) | gateway, always on | `keyframes`, `loop_closures` (distinct closure-type graph links), `path_length_m`, `db_bytes`, `last_update_age_s` (null before the first graph and in `localize` mode), `mode`, `calibration_placeholder` |
| `/ugv/perception/stats` | `std_msgs/String` (JSON) | Dev 1 perception | gateway, always on | `mask_hz`, `depth_hz`, `stage_ms`, `depth_errors` |
| `/perception/depth/image` | `sensor_msgs/Image` 32FC1 | Dev 1 perception | RTAB-Map, gateway (depth layer, live cloud) | published for every processed frame, not only when a mask is published (D14) |
| `/global_costmap/costmap` | `nav_msgs/OccupancyGrid` | Nav2 | gateway (on demand) | the cost grid layer, display only |
| `/image_raw/compressed` | `sensor_msgs/CompressedImage` | camera driver | gateway (on demand) | camera panel, passed through |
| `/ugv/elevation/cloud`, `/ugv/elevation/obstacles` | `sensor_msgs/PointCloud2` | **nobody yet** | gateway (configured, on demand) | elevation deferred; the subscriptions exist and stay silent |

Topic names are gateway parameters under `map:` in `ugv_nav/ugv_api/config/api.yaml`.

## Gateway endpoints (`ugv_api`, read only)

| Endpoint | Body | Notes |
|---|---|---|
| `GET /api/v1/map` | `MapStatus` JSON: `epoch`, `seq` for the seven layers, flat `stats` | also the demand heartbeat: only an explicit GET keeps the heavy subscriptions alive (`idle_timeout_s`, default 10 s) |
| `GET /api/v1/map/pose` | `Pose` JSON, `map -> base_link` from TF | `available: false` and null fields until a transform is seen |
| `GET /api/v1/map/{cloud,elevation,trajectory,grid,live,depth}` | binary format v1 (below) | 503 `application/problem+json` until the layer has data |
| `GET /api/v1/map/camera` | `image/jpeg` | the camera driver's JPEG, unchanged |
| `GET /api/v1/telemetry/stream` | SSE | gains the `map` (MapStatus) and `pose` events |

`epoch` is random per gateway process; `seq[layer]` counts changes (0 = nothing yet). A client compares `epoch:seq`, not
`seq` alone, and refetches every layer when the epoch changes. `stats` passes the ROS stats keys through unchanged and adds
the gateway's own input health: `map_inputs_alive`, `map_rejects`, `map_restarts`, `map_last_reject`. JSON contract:
`.superpowers/sdd/2026-10-02-3d-mapping-elevation/map-json-contract.md`.

### Binary format v1

Little endian; a 24-byte prelude (`magic`, `format = 1`, `header_bytes`, `epoch`, `seq`, `stamp_s`), then a layer header and
body. Decoders reject an unknown format and any length that is not exact.

| Layer | Magic | Body |
|---|---|---|
| cloud, live | `UGVC` | `f32 xyz`, then `u8 rgb` if flag bit 0; at most `cloud_point_budget` points (default 500000) picked by spatial hash at `cloud_spacing_m` (0.05). `live` is the current depth scan back-projected with CameraInfo K (every 4th pixel, 0.3-8 m) and transformed to `map` |
| trajectory | `UGVT` | `f32 x y z qx qy qz qw` per pose |
| grid | `UGVG` | `i8` cells (-1 unknown, 0..100); size, resolution, origin and yaw in the header |
| depth | `UGVD` | `u16` millimetres, every 2nd pixel (320x240), 0 = hole |
| elevation | `UGVE` | `f32` height, `u8` obstacle (5 cm units), `u8` confidence. **Encoder and decoder exist; no data source yet** |

Authoritative layout and the golden files the Python and TypeScript tests share:
`.superpowers/sdd/2026-10-02-3d-mapping-elevation/binary-format-v1.md`, `ugv_nav/ugv_api/test/fixtures/map/*.bin`.

## Run mapping and view it

Full stack with the live camera (arguments as in `ugv_nav/ugv_bringup/README.md`):

```
ros2 launch ugv_bringup bringup.launch.py profile:=live_cam mode:=mapping \
    calibration_file:=<phone yaml> device:=<stream URL> transport_latency_s:=<measured s> \
    camera_x:=.. camera_y:=.. camera_z:=.. camera_pitch_deg:=.. perception_src:=<repo>/turing/src
cd ui && npm install && npm run dev      # http://localhost:5173, proxies /api to the gateway (UGV_API_URL)
```

Open the console and pick **map** in the top bar. The view shows the accumulated cloud (camera colours), the live depth scan
(height colours), the trajectory, the Nav2 cost grid halo, the robot pose, the depth and camera image panels and the
statistics widget. Layer buttons: cloud, live, path, elev (empty, see above), cost, img; the choice is kept in
`localStorage`. Banners: `NO MAP YET`, `STALE · map not updating`, `STALE · telemetry lost`, `MAP INPUTS STOPPED` (the
gateway's input thread is gone).

Drive the loop, then stop the stack: the database is saved at `~/.ros/ugv/rtabmap.db` (override with `database_path`). To
localize on it, relaunch with `mode:=localize`. The localization stack alone:
`ros2 launch ugv_localization localization.launch.py mode:=mapping|localize` (`fresh_db:=true` starts an empty database).
Task 8 may move map assembly into a separate `map_assembler` process if assembling inside the SLAM callback delays
`/rtabmap/info` and trips `slam_stale`; check `ugv_nav/ugv_localization/launch/localization.launch.py` for which is in force.

Tests: `python -m pytest ugv_nav/ugv_api/test -k map` (codec, store, endpoints); `cd ui && npm run lint && npm test &&
npm run build`; in the container `colcon test --packages-select ugv_localization ugv_bringup ugv_api`.

## Honest limits

- **Monocular scale wobble.** Depth is DA3 pseudo-depth from one camera. Its scale varies from frame to frame and with
  scene content (see `docs/mapping/baseline.md`; mono is the architecture's minimum tier). Loop closure corrects pose, not a
  depth scale that differed between keyframes: a wall seen twice can sit at two distances.
- **5 m fusion range.** The 3D map fuses depth from 0.3 to 5.0 m (`Grid/RangeMin`, `Grid/RangeMax`); far DA3 depth smears.
  The live scan shows up to 8 m, but it is not map.
- **Heights are relative to the driving plane.** `Reg/Force3DoF` is true: pose is x, y, yaw only and `base_link` z = 0 is
  the ground. Pitch and slopes are not tracked, so a ramp can read as a wall or a drop.
- **Clipped about 1 m above the robot.** `Grid/MaxObstacleHeight: "1.0"` also clips `cloud_map` (measured: z max 0.96 m, 1.42 m
  with it off). Upper walls and trees are missing. Owner decision pending.
- **Placeholder calibration.** `ugv_nav/config/cameras/phone_640x480.yaml` is the laptop webcam's intrinsics with
  `placeholder: true`. Scale and projection are wrong until the phone is calibrated; `calibration_placeholder` shows in the
  statistics and the driver logs a WARN. Do not trust any measurement made with it.
- **Stamps are arrival time minus `transport_latency_s`** for a network camera, not exposure time (mindmap D10).
- **A mask is held up to about 0.58 s** against the 0.5 s limit (owner decision below).
- **Display only.** Not a Nav2 input, not a safety input; elevation into Nav2 needs its own §9 decision.

## Pending owner runs

None of this has happened and no numbers exist for it. The synthetic screenshots in this folder are not UGV evidence.

1. The recorded moving run (a 2-3 minute closed loop: `DEPTH_CLOUD_TOPIC="" GT_DEPTH_TOPIC="" bash record_eval_bag.sh eval_bags/loop1`, replayed through `bag_eval.launch.py`) and the four go/no-go gate numbers of Task 7: depth image rate >= 5 Hz, `/ugv/pose_valid` true while moving >= 90 %, visual odometry lost < 5 % of frames, closed-loop start-to-end error < 5 % of path length. Record them in `docs/mapping/gate-phase0.md`.
2. Phone camera: lock focus and exposure, calibrate (`calibration_mode:=true` flow in `ugv_nav/ugv_bringup/README.md`), replace the placeholder YAML and remove the flag, measure the tunnel latency and set `transport_latency_s`.
3. A re-measure in a lit scene (the baseline run used a black image, so its odometry and depth-stability rows are invalid) and the tape-measured wall test at 1-5 m (depth error per distance).
4. The closed-loop end-to-end run on the UGV: mapping, save the database, restart in `localize`. Keep a screenshot of the map view and the stats values in this folder.
5. Two open owner decisions: (a) the mask freshness budget, since a mask can be held up to about 0.58 s against the 0.5 s limit; (b) whether to keep `Grid/MaxObstacleHeight 1.0`, which clips the 3D cloud at about 1 m above the robot.

## Files

| File | What |
|---|---|
| `docs/mapping/baseline.md` | baseline measurement; before/after numbers for depth rate and odometry QoS |
| `docs/mapping/map-view-synthetic-*.jpg` | the map view on synthetic data (not UGV evidence) |
| `ugv_nav/docs/localization/interfaces.md` | topic contracts, 3D map outputs, `mapData` consumer contract |
| `ugv_nav/ugv_api/config/api.yaml` | gateway `map:` parameters |
| `ui/src/map/`, `ui/src/components/MapView.tsx` | decoders, scheduler, geometry, three.js scene, view |
