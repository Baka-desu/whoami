# UGV Nav: Status Report

**As of:** 2026-10-01 (after commit `b76b636` "everything works now")
**Scope:** how the system runs today, the web UI, live camera test results, why no map showed.

## 1. Summary

The full `live_cam` stack ran end to end on the laptop (RTX 4060, Docker `ugv-live`, webcam over MJPEG). The
camera, segmentation mask and path overlay worked against real frames. The map was not visible.
The main reason is that no screen we have draws it: `ui/` intentionally shows no map, costmap or Nav2 plan.
Localization was also often invalid (`pose_valid=false`), so even in RViz the map would have been thin or empty.

## 2. How the system works now

One launch (`ugv_bringup/launch/bringup.launch.py profile:=live_cam`) starts every subsystem in data-flow order:

```
Windows webcam ─► webcam_stream.py (MJPEG :8090)
                    │
                    ▼
 Dev 5  camera driver ─► /camera/image_raw + /camera/camera_info (+ /image_raw/compressed for web)
 Dev 5  robot description ─► base_link -> camera_link -> camera_optical_frame
                    │
        ┌───────────┴─────────────┐
        ▼                         ▼
 Dev 1  perception            Dev 2  localization
   RUGD SegFormer-B5            RTAB-Map RGB-D on DA3 depth
   DA3 Metric Large             map -> odom -> base_link
   -> /segmentation/mask        -> /ugv/pose_valid
   -> /perception/depth_*
   -> /ugv/perception_degraded
        │                         │
        ▼                         │
 Dev 3  semantic costmap ◄────────┘
   mask + CameraInfo + TF -> /semantic_costmap/grid (Nav2 StaticLayer)
        │
        ▼
 Dev 4  Nav2 (Smac2D + RPP) -> /cmd_vel_nav2, /ugv/nav2_heartbeat
        │
        ▼
 Dev 5  safety arbiter -> /cmd_vel (sole publisher; e-stop > health > degraded/invalid pose > Nav2)
 Dev 5  ugv_api gateway (HTTP :8080) -> ui/ operator console
```

| Subsystem | Package / path | State |
|---|---|---|
| Camera driver (Dev 5) | `ugv_nav/ugv_bringup` | Working with the laptop webcam. Calibration `config/cameras/laptop_webcam_640x480.yaml` (RMS 1.30 px). |
| Robot description (Dev 5) | `ugv_nav/ugv_robot_description` | Camera mount from launch args (z 0.97 m, pitch 0 for a laptop on a desk). |
| Perception (Dev 1) | `turing/src/ugv_perception` (plain Python, not colcon) | Working: RUGD mask + DA3 depth on CUDA. |
| Localization (Dev 2) | `ugv_nav/ugv_localization` | Runs, but `pose_valid` often false (odometry ~1 Hz, `rgbd_sync` pairing). Laptop timing profiles added. |
| Semantic costmap (Dev 3) | `ugv_nav/ugv_costmap` | New node: 0.1 m grid, 6 m ahead, ±4 m wide, stale mask (>0.5 s) -> FOV lethal. |
| Nav2 (Dev 4) | `src/ugv_navigation` | Launched; footprint is a Jackal-size placeholder. |
| Safety arbiter (Dev 5) | `ugv_nav/ugv_safety` | Sole `/cmd_vel` publisher; e-stop latches across restarts. |
| Operator API (Dev 5) | `ugv_nav/ugv_api` (new) | HTTP gateway + SSE telemetry for `ui/`. |

Profiles `sim` and `bag` are not wired yet; bringup refuses them.

## 3. Web UI (`ui/`)

There is one web app. The perception workbench (`turing/workbench/`) was merged back into `ui/` on 2026-10-01 and
the folder was removed.

| Area | Contents | Talks to |
|---|---|---|
| Main page (centre) | Live camera with mask / depth / path overlay, freshness banners, layer toggles | rosbridge :9090, read only |
| Left sidebar | E-stop, mapping/localize mode, map-frame goal, camera source (robot / browser / photo) | `ugv_api` gateway; rosbridge for the camera |
| Right sidebar | §12 health table + arbiter, `/cmd_vel`, navigation, localization, perception widgets (ground map, classes, depth, health, source) | `ugv_api` SSE; rosbridge |

Commands (e-stop, mode, goals) still go only through the gateway. The UI never publishes to ROS, and a test
enforces both rules. `ui/README.md` is up to date.

## 4. Live test results

| Item | Result |
|---|---|
| Real webcam -> camera driver -> web stream | Worked |
| RUGD segmentation mask overlay on live frames | Worked |
| DA3 metric depth layer | Not confirmed (off by default; toggle `depth` in the camera bar) |
| Path overlay | Worked. This is the flat-ground preview from the mask (`analysis/groundmap.ts`), **not** Nav2's planned path. |
| Operator console (safety board, `/cmd_vel`, mode, goals) | Not confirmed item by item ("most was working") |
| Map | **Not seen** (section 5) |

## 5. Why the map did not show

1. **No screen draws it.** `ui/` deliberately renders no RTAB-Map map, costmap (`/global_costmap/costmap`,
   `/local_costmap/costmap`, `/semantic_costmap/grid`) or Nav2 plan. The "path" you saw is the camera-view
   ground preview. The ground-map widget in the right sidebar is also mask-only.
2. **Localization was weak.** With `pose_valid` often false and odometry around 1 Hz, RTAB-Map adds few or no
   nodes to the graph. There's no reliable `map -> odom`, so a `/map` grid would be empty or barely grow.
3. **The laptop didn't move.** A desk-mounted webcam sees one viewpoint. RTAB-Map needs motion to build a map.

**How to see it now:** run RViz in the container (WSLg is set up) and add `/map`, `/semantic_costmap/grid`,
`/global_costmap/costmap`, `/plan` and the TF tree. Or check quickly with:

```bash
ros2 topic echo /ugv/pose_valid
ros2 topic hz /semantic_costmap/grid
ros2 topic echo /global_costmap/costmap --once --no-arr
```

## 6. Next steps

1. Fix localization on the laptop: verify the `rgbd_sync_laptop.yaml` profile, raise odometry rate, get a stable
   `pose_valid=true`.
2. Decide where the map lives: a read-only map/costmap panel in `ui/` (fed through `ugv_api` to keep the
   gateway rule), or keep it in RViz.
3. ~~Update `ui/README.md`, remove dead UI code~~ Done with the workbench merge.
4. Make a mapping run that moves (walk the laptop around), save `rtabmap.db`, then test `mode:=localize` + an A->B goal.
5. Replace the Jackal placeholder footprint and add the second footprint YAML (Definition of Done item 9).
6. Wire the `sim` and `bag` profiles in bringup.
