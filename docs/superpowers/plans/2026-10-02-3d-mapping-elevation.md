# Monocular 3D Mapping + Elevation Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn per-frame DA3 depth into a consistent, continuously updated 3D representation: RTAB-Map's global map and trajectory, an elevation map (height, obstacle geometry, confidence, unknown), and a 3D viewer in the web UI.

**Architecture:** RTAB-Map stays the only fusion engine; we switch on its 3D outputs instead of writing a second one. A new elevation node builds a 2.5D grid as a pure function of (RTAB-Map keyframe depth, optimized keyframe poses), so a loop closure rebuilds it with no ghosting. All map data reaches the browser through the existing `ugv_api` gateway as versioned binary HTTP bodies; the viewer is plain three.js.

**Tech Stack:** ROS 2 Lyrical, Python 3.14, rtabmap_ros 0.23.7 (OctoMap yes, GridMap no), numpy, torch cu128, FastAPI, React 19 + Vite + TypeScript, three 0.186, pytest, vitest.

**Spec:** this document (Context, Decisions, Design). Authority order: `architecture.md` > `dev.md` > this plan. Deviations are recorded in `mindmap.md` (Task 1); `architecture.md` is **not edited**.

---

## Context

The stack runs end to end on the laptop and shows a camera feed, mask, depth and a path preview, but it is a per-frame depth application: nothing accumulates, and no screen draws a map. Exploration found that most of the fusion machinery already exists and is switched off or unseen:

- RTAB-Map already fuses RGB + DA3 depth + pose, but `Grid/3D` is `false` and nothing subscribes to `cloud_map`, `mapPath`, `mapGraph` or `mapData` (`ugv_nav/ugv_localization/config/rtabmap_rgbd.yaml:53`).
- Perception already publishes `/perception/depth/image` and `/perception/depth_cloud`; Nav2 already marks and inflates obstacles from that cloud (`src/ugv_navigation/config/costmaps.yaml`). The reference screenshot (live height-coloured cloud + costmap halo) needs a display, not new mapping code.
- No elevation mapping exists. The image has no `grid_map`/`elevation_mapping` (no apt candidates on Lyrical) and RTAB-Map was built without GridMap, so the elevation map must be our own node.
- The UI has no 3D data rendering, and images travel as base64-in-JSON over rosbridge, which bringup does not even launch.
- **The real blocker is pose.** The owner's status report says `pose_valid` was often false with odometry near 1 Hz, and no moving mapping run has ever been made. Fusing depth with bad poses gives a smeared map, so pose health is Phase 0 and gates everything after it.

## Owner decisions (fixed)

| # | Decision |
|---|---|
| 1 | Build map + elevation + visualization now. **Nav2 and the costmaps are not changed**; feeding elevation into Nav2 is a separate later plan. |
| 2 | Real differential-drive UGV. Stack runs on the RTX 4060 laptop (Docker). A **phone camera** rides the UGV and streams 640x480 through a network tunnel. |
| 3 | **Visual odometry only** (no wheel encoders). |
| 4 | Phone gets its own calibration file, seeded from the laptop calibration as a flagged placeholder until calibrated. |
| 5 | General navigation map: generic defaults, no environment-specific tuning, only four elevation layers (height, obstacle, confidence, unknown). |
| 6 | 3D viewer in the web UI (three.js) plus an RViz config for debugging. The web viewer also shows the live height-coloured cloud and Nav2's costmap, display only. |
| 7 | No deadline; do it properly with tests. |
| 8 | Do not edit `architecture.md`. Record deviations in `mindmap.md`. |

## Global Constraints

- `/cmd_vel` authority, the perception port, and Nav2 costmap inputs are untouched (architecture §3.1, §8, §9).
- The UI never publishes to ROS and never opens new ROS connections for map data; map data comes only from `ugv_api` (`ui/src/source/api.test.ts` guard).
- No new message packages and no new Python or npm dependencies. Topics use standard types (`sensor_msgs/PointCloud2`, `nav_msgs/OccupancyGrid`, `nav_msgs/Path`, `std_msgs/String` JSON).
- Repo style: pure-Python kernel + thin ROS wrapper node; kernels tested with plain pytest, ROS behaviour in `test_ros_stack.py`.
- `Reg/Force3DoF` stays `"true"`. Elevation height is **relief relative to the driving plane** (`base_link` z = 0 is the ground, `ugv_robot_description/urdf.py`), not absolute altitude.
- Fuse near range only: 0.3 m to 5.0 m from the camera (matches `Grid/RangeMax`).
- Wire format is little-endian; every binary body starts with the 24-byte prelude in "Binary format v1".
- Commit after every task. Branch: `mapping-3d`.

## Review Focus

Failure modes the design implies that are most likely to bite; each has a test in the named task.

1. **Tunnel stalls, then delivers a burst of old frames.** The driver must publish only the newest frame, never a backlog (Task 6).
2. **A loop closure moves old keyframes.** The elevation map must rebuild at the new poses and leave nothing at the old ones (Task 11).
3. **RTAB-Map restarts with `fresh_db`, so node ids restart at 1.** Elevation tiles for ids no longer in the graph must be dropped; an empty graph clears the map (Task 11).
4. **The gateway restarts while the map view is open.** Sequence numbers restart, so the client must compare `epoch:seq`, not `seq` alone, and refetch every layer (Tasks 13, 18).
5. **Map view opened before anything is mapped, or with a truncated body.** Endpoints answer 503 problem JSON, decoders return `null`, and the view shows "no map yet" instead of crashing (Tasks 14, 16).

---

## File structure

| Area (owner) | Create | Modify |
|---|---|---|
| Perception (Dev 1) `turing/src/ugv_perception/` | — | `node/adapter_node.py`, `node/metrics.py`, `backend/cuda_pytorch.py`, `backend/depth_live.py`, `depth/geometry.py`, `tests/test_adapter_node.py`, `tests/test_depth_meters.py` |
| Camera (Dev 5) `ugv_nav/ugv_bringup/` | `rviz/mapping.rviz` | `ugv_bringup/camera_core.py`, `ugv_bringup/nodes/camera_driver.py`, `test/test_camera_core.py`, `docker/live.Dockerfile`, `setup.py`, `README.md` |
| Calibration (Dev 2) | `ugv_nav/config/cameras/phone_640x480.yaml` | `ugv_localization/camera/calib.py`, `test/test_camera_calib.py`, `config/cameras/README.md` |
| 3D map (Dev 2) `ugv_nav/ugv_localization/` | `ugv_localization/mapstats/{__init__,stats}.py`, `nodes/map_stats_node.py`, `test/test_mapstats.py` | `config/rtabmap_rgbd.yaml`, `launch/localization.launch.py`, `scripts/record_eval_bag.sh`, `setup.py`, `test/test_ros_stack.py`, `test/test_ros_smoke.py` |
| Elevation (Dev 2) `ugv_nav/ugv_localization/` | `ugv_localization/elevation/{__init__,contracts,decode,tile,store,fuse}.py`, `nodes/elevation_map_node.py`, `config/elevation.yaml`, `test/test_elevation_{decode,tile,fuse,store}.py` | `launch/localization.launch.py`, `setup.py`, `test/test_ros_stack.py` |
| Gateway (Dev 5) `ugv_nav/ugv_api/` | `ugv_api/mapcodec.py`, `ugv_api/mapstore.py`, `test/test_mapcodec.py`, `test/test_mapstore.py`, `test/test_map_api.py`, `test/fixtures/map/*.bin` | `ugv_api/app.py`, `ros_node.py`, `schemas.py`, `state.py`, `main.py`, `config/api.yaml`, `package.xml`, `test/test_api_ros.py` |
| Viewer (Dev 5) `ui/src/` | `map/codec.ts`, `map/codec.test.ts`, `map/useMapData.ts`, `map/geometry.ts`, `map/geometry.test.ts`, `map/scene.ts`, `components/MapView.tsx` | `App.tsx`, `index.css`, `source/api.ts`, `source/api.test.ts`, `components/StatusWidgets.tsx`, `README.md` |
| Docs / CI | `docs/superpowers/plans/2026-10-02-3d-mapping-elevation.md`, `docs/mapping/{baseline,gate-phase0,README}.md` | `mindmap.md`, `ugv_nav/docs/localization/interfaces.md`, `.github/workflows/ugv_api.yml`, `.github/workflows/ui.yml` |

---

## Phase 0 — Base, baseline, pose health

### Task 1: Branch, merge, record decisions

**Files:** Modify `mindmap.md`; Create `docs/superpowers/plans/2026-10-02-3d-mapping-elevation.md` (copy of this plan).

- [ ] `git switch -c mapping-3d main && git merge --no-ff full-withcamera`. Expected: no conflicts (verified: the two branches changed no file in common since b76b636).
- [ ] Record the baseline before any change: `cd ui && npm ci && npm run lint && npm test && npm run build`; `cd turing && python -m pytest`; in the container `colcon test --packages-select ugv_localization ugv_api ugv_bringup`. Write pass/fail counts into the commit message; pre-existing failures are noted, not fixed.
- [ ] Append rows to the `## Decisions` table of `mindmap.md` (current last row is D8), one per deviation from `architecture.md`:

| ID | Topic | Decision (what deviates) |
|---|---|---|
| D9 | Map products | RTAB-Map `Grid/3D true` + `cloud_map`/`mapPath`/`mapGraph`/`mapData` consumed; new elevation map and map-stats nodes in `ugv_localization`. Architecture §5–§7 name no map products. Mapping and display only: not a second perception port, not a Nav2 input (§9 unchanged). |
| D10 | Camera transport | Phone camera on the UGV streamed through a network tunnel to the laptop. Stamps are **arrival time** minus a measured `transport_latency_s`, where §8.4 says "image time". |
| D11 | Calibration | `phone_640x480.yaml` seeded from the laptop calibration with `placeholder: true` until the phone is calibrated. §17 assumes calibrated vision and `config/cameras/README.md` forbids placeholders; mapping runs for the Phase 0 gate require the real calibration. |
| D12 | Elevation semantics | `Reg/Force3DoF true` kept; elevation height is relief relative to the driving plane. General-purpose defaults, four layers only. |
| D13 | Operator UI | Map, live cloud and Nav2 costmap shown in the web UI through `ugv_api` binary endpoints, display only. The UI guard's blanket ban on the word "costmap" is narrowed to costmap topic names. |
| D14 | Depth vs mask | DA3 depth is published for every processed frame, no longer only when a mask was published, so a degraded mask does not starve SLAM. |

- [ ] Commit: `chore: start mapping-3d from main + full-withcamera; record decisions D9-D14`.

### Task 2: RViz live view (reproduces the reference screenshot)

**Files:** Create `ugv_nav/ugv_bringup/rviz/mapping.rviz`; Modify `ugv_nav/ugv_bringup/docker/live.Dockerfile` (add `ros-lyrical-rviz2 ros-lyrical-rtabmap-rviz-plugins`, both have apt candidates), `ugv_nav/ugv_bringup/setup.py` (install `rviz/`), `README.md`.

- [ ] Displays: Image `/camera/image_raw`; Image `/perception/depth/image`; PointCloud2 `/perception/depth_cloud` (AxisColor on Z, fixed frame `map`); Map `/global_costmap/costmap` (costmap colour scheme); TF; plus `/rtabmap/cloud_map`, Path `/rtabmap/mapPath`, PointCloud2 `/ugv/elevation/cloud`, Map `/ugv/elevation/obstacles` (empty until Tasks 8 and 12).
- [ ] Verify by observation: run `live_cam` with the laptop webcam, `rviz2 -d <installed path>/mapping.rviz` over WSLg; capture a screenshot showing depth panel, RGB panel, height-coloured cloud and the inflation halo. Save it to `docs/mapping/`.

### Task 3: Perception stage timing and visible depth failures

**Files:** Modify `turing/src/ugv_perception/node/metrics.py`, `node/adapter_node.py`; Test `tests/test_adapter_node.py`.

**Produces:** topic `/ugv/perception/stats` (`std_msgs/String`, JSON, 1 Hz): `{"mask_hz","depth_hz","stage_ms":{"decode","seg","depth_infer","depth_post","cloud","publish"},"depth_errors","last_depth_error"}`.

- [ ] Failing tests first: (a) a depth channel that raises increments `metrics.depth_errors` and stores the message (today `adapter_node.py:221-222` swallows it silently); (b) `metrics.latencies_ns` is bounded (a `deque(maxlen=600)`; today it grows forever); (c) stage timers accumulate with an injected clock.
- [ ] Implement; log the first depth error and then every 100th at WARN.
- [ ] Run `cd turing && python -m pytest src/ugv_perception/tests/test_adapter_node.py -v`. Expected: PASS.

### Task 4: Measure the baseline (no code)

**Files:** Create `docs/mapping/baseline.md`.

- [ ] On `mapping-3d` with the laptop webcam, record for 60 s: camera fps delivered, `mask_hz`, `depth_hz`, every `stage_ms`, `ros2 topic hz /odom`, fraction of time `/ugv/pose_valid` is true. The "~1 Hz" figure predates the GPU decode commits now on `main` and may be stale.
- [ ] Static scene, 60 s: per-frame median depth of a fixed image patch; report its standard deviation (DA3 scale wobble).
- [ ] Wall test: tape-measured distances at 1, 2, 3, 4, 5 m; report DA3 error at each. These numbers set `NoiseModel.sigma0`/`k` (Task 10) and confirm the 5.0 m fusion range.
- [ ] Decision recorded in the doc: Task 5 optimisation steps are applied only until `depth_hz >= 8`.

### Task 5: Depth for every frame, at a usable rate

**Files:** Modify `node/adapter_node.py`, `backend/cuda_pytorch.py`, `backend/depth_live.py`, `depth/geometry.py`; Test `tests/test_adapter_node.py`, `tests/test_depth_meters.py`.

- [ ] **Depth independent of the mask** (always done). Failing test: when `perception_cycle` yields no mask (stale input), `/perception/depth/image` is still published for a new image+info pair. Implement by calling `_publish_depth()` on `had_pair` rather than inside `if wired.mask is not None` (`adapter_node.py:187-199`).
- [ ] **Single decode** (always done): reuse the frame decoded by the cycle instead of calling `decode_frame` a second time at `adapter_node.py:211`.
- [ ] If Task 4 shows `depth_hz < 8`, apply in order, re-measuring after each and stopping when the target is met:
  1. float32 pre/post-processing on the GPU (`preprocess_nchw`, `hole_safe_resize`); parity test: max abs difference vs the numpy path < 1 mm on the synthetic plane in `test_depth_meters.py`.
  2. `torch.autocast("cuda", dtype=torch.float16)` around the DA3 call (`cuda_pytorch.py:126-135`); parity test: relative difference vs fp32 < 1 % on valid pixels.
  3. Skip building `/perception/depth_cloud` when `get_subscription_count() == 0`.
  4. Only if still short: a scheduling kernel (depth every frame, mask at least every 0.25 s), kernel-tested with an injected clock.
- [ ] Append before/after numbers to `docs/mapping/baseline.md`.

### Task 6: Camera driver for a tunnelled phone stream

**Files:** Modify `ugv_nav/ugv_bringup/ugv_bringup/camera_core.py`, `nodes/camera_driver.py`, `ugv_localization/camera/calib.py`, `ugv_nav/config/cameras/README.md`; Create `ugv_nav/config/cameras/phone_640x480.yaml`; Test `ugv_bringup/test/test_camera_core.py`, `ugv_localization/test/test_camera_calib.py`.

**Produces:**
```python
class LatestFrameReader:            # camera_core.py; thread that drains a capture and keeps only the newest frame
    def __init__(self, read: Callable[[], tuple[bool, object]], now_s: Callable[[], float]) -> None
    def start(self) -> None
    def take(self) -> tuple[object, float] | None   # (frame, arrival_s) once per frame, else None
    @property
    def dropped(self) -> int                        # frames superseded before being taken
    def stop(self) -> None
```
Driver param `transport_latency_s` (default `0.0`): stamp = arrival − latency. `Calibration` gains `placeholder: bool = False`.

Assumption: the phone serves a stream OpenCV can open by URL (MJPEG over HTTP or RTSP), reached through the tunnel with the existing `device:=http://…` argument. Such streams carry no capture timestamps, which is why latency is measured once and configured rather than read per frame.

- [ ] Failing tests: (Review Focus 1) a fake `read` that yields 10 frames instantly then blocks → `take()` returns only the 10th and `dropped == 9`; `take()` returns `None` when no new frame arrived; a YAML with `placeholder: true` loads with `cal.placeholder is True`.
- [ ] Implement. The driver uses `LatestFrameReader` for non-V4L2 sources (`camera_driver.py:101`), logs a WARN every 10 s while the calibration is a placeholder, and reports `dropped` at 0.1 Hz.
- [ ] `phone_640x480.yaml`: K and D copied from `laptop_webcam_640x480.yaml`, `camera_name: phone`, `placeholder: true`, header stating it is not a calibration. Add the exception paragraph to `config/cameras/README.md`.
- [ ] Hardware steps (owner): lock phone focus and exposure; calibrate with the existing `calibration_mode:=true` + `cameracalibrator` flow; replace the YAML and remove `placeholder`. Measure tunnel latency by filming a millisecond clock shown on the laptop and set `transport_latency_s`.

### Task 7: Recorded moving run and go/no-go gate

**Files:** Modify `ugv_nav/ugv_localization/scripts/record_eval_bag.sh` (an empty `DEPTH_CLOUD_TOPIC` omits the cloud; today line 16 cannot disable it); Create `docs/mapping/gate-phase0.md`.

- [ ] Record a 2–3 minute closed loop with the phone on the UGV (a cart is acceptable until the UGV is ready): `DEPTH_CLOUD_TOPIC="" GT_DEPTH_TOPIC="" bash record_eval_bag.sh eval_bags/loop1`.
- [ ] Replay through the existing `bag_eval.launch.py` and record the four gate numbers.

| Gate | Threshold |
|---|---|
| Depth image rate | ≥ 5 Hz |
| `/ugv/pose_valid` true while moving | ≥ 90 % |
| Visual odometry lost | < 5 % of frames |
| Start-to-end error on the closed loop, before loop closure | < 5 % of path length |

- [ ] **If any gate fails: stop.** Report the numbers and decide with the owner (slower driving, more depth rate, or a phone IMU/ARCore pose source) before any Phase 2 work. The thresholds are generic starting values, not tuned.

---

## Phase 1 — Unified 3D map

### Task 8: RTAB-Map 3D outputs

**Files:** Modify `config/rtabmap_rgbd.yaml`, `test/test_ros_stack.py`, `test/test_ros_smoke.py`.

- [ ] Failing test `test_x3` in `test_ros_stack.py`: with moving synthetic input so graph nodes are added, assert `/map` still publishes, `/rtabmap/cloud_map` is non-empty with `rgb`, `/rtabmap/mapPath` grows, and every `mapData` node whose id is in `graph.poses_id` has non-empty `data.right_compressed`.
- [ ] Config: `Grid/3D: "true"` (line 53); pin node params `cloud_output_voxelized: true`, `cloud_subtract_filtering: false`, `map_always_update: false`, `latch: true`. Leave `Reg/Force3DoF` and all `Vis/*` as they are. No remaps are needed: the node runs in namespace `rtabmap`.
- [ ] Run in the container: `colcon test --packages-select ugv_localization --pytest-args -k "x3 or smoke"`. Expected: PASS, and `check_rtabmap_params` accepts the file.
- [ ] Risk check on the bag: with a `cloud_map` subscriber attached, `/ugv/pose_valid` must not gain `slam_stale` (map assembly runs before `/rtabmap/info` in the same callback). If it does, move assembly to `rtabmap_util/map_assembler` in its own process and record that in the commit.

### Task 9: Map statistics

**Files:** Create `ugv_localization/mapstats/stats.py`, `nodes/map_stats_node.py`, `test/test_mapstats.py`; Modify `launch/localization.launch.py`, `setup.py` (entry point `map_stats_node`).

**Produces:** `/ugv/map/stats` (`std_msgs/String` JSON, transient local, 1 Hz):
`{"keyframes","loop_closures","path_length_m","db_bytes","last_update_age_s","mode","calibration_placeholder"}`.
Kernel: `path_length(poses: Sequence[tuple[float,float]]) -> float`, `class MapStats` with `on_graph(ids, poses, now_s)`, `on_info(loop_closure_id, proximity_id)`, `snapshot(now_s, db_bytes) -> dict`.

- [ ] Kernel tests: path length of a 3-4-5 polyline; a loop-closure id counted once; `last_update_age_s` from an injected clock. The node subscribes to `/rtabmap/mapGraph` and `/rtabmap/info` only, never `cloud_map` (subscribing forces assembly).

---

## Phase 2 — Elevation map

Design: each RTAB-Map keyframe becomes a **tile**, a compact per-cell summary in `base_link`. The grid is `fuse(tiles, optimized poses)`. Obstacles are classified inside each tile, where a pitch bump tilts ground and obstacle together, not in the fused grid.

Keyframe depth comes from RTAB-Map itself, not from a private cache: `mapData.nodes[].data.right_compressed` (32FC1 stored as PNG of the float bytes viewed as 8UC4), with `left_camera_info[0].k` and `local_transform[0]` (base → camera). On start, the node calls `/rtabmap/rtabmap/get_map_data` (`rtabmap_msgs/srv/GetMap`: `global_map=true, optimized=true, graph_only=false`), which covers `localize` mode, restarts and existing databases. All three message layouts were read from the installed 0.23.7 image.

### Task 10: Decode and tile kernel

**Files:** Create `ugv_localization/elevation/{__init__,contracts,decode,tile}.py`, `test/test_elevation_decode.py`, `test/test_elevation_tile.py`.

**Produces:**
```python
# contracts.py
@dataclass(frozen=True)
class GridSpec:
    resolution: float = 0.10      # m per cell
    range_min: float = 0.3        # m from camera
    range_max: float = 5.0
    z_min: float = -0.5           # m in base_link; outside [z_min, z_max] is discarded
    z_max: float = 2.0
    step_height: float = 0.15     # in-cell height spread above this = obstacle
    max_ground_dev: float = 0.30  # lowest point this far from z=0 with no ground seen = obstacle
    stride: int = 2               # depth pixel decimation

@dataclass(frozen=True)
class NoiseModel:
    sigma0: float = 0.02          # m; sigma(r) = sigma0 + k * r**2  (set from Task 4's wall test)
    k: float = 0.01               # 1/m
    sigma_ref: float = 0.10       # m; confidence = 1 / (1 + (sigma_total / sigma_ref)**2)

@dataclass(frozen=True)
class Tile:
    node_id: int
    ij: np.ndarray          # (N, 2) int16: cell index in base_link, i along +x, j along +y
    z: np.ndarray           # (N,) float32: lowest height in the cell, metres
    var: np.ndarray         # (N,) float32
    obstacle_h: np.ndarray  # (N,) float32: height of obstacle above z, 0 = not an obstacle

Pose2D = tuple[float, float, float]   # x, y, yaw of base_link in map

# decode.py
def depth_from_rtabmap(png: bytes) -> np.ndarray          # (H, W) float32 metres; ValueError if not 8UC4
# tile.py
def make_tile(node_id: int, depth_m: np.ndarray, k: tuple[float, float, float, float],
              base_T_cam: np.ndarray, spec: GridSpec, noise: NoiseModel) -> Tile
```

- [ ] Failing tests (helpers live at the top of `test_elevation_tile.py`):
```python
R_LINK_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical (x right, y down, z fwd) -> link (x fwd, y left, z up)

def _camera(cam_z, pitch_deg, w=64, h=48, f=60.0):
    p = math.radians(pitch_deg)                                      # positive = looking down, as in urdf.py
    r_pitch = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    base_T_cam = np.eye(4); base_T_cam[:3, :3] = r_pitch @ R_LINK_OPT; base_T_cam[2, 3] = cam_z
    k = (f, f, w / 2 - 0.5, h / 2 - 0.5)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([(u - k[2]) / f, (v - k[3]) / f, np.ones((h, w))], -1) @ base_T_cam[:3, :3].T   # base_link, per unit optical z
    return k, base_T_cam, rays

def synth_plane(cam_z=0.5, pitch_deg=20.0):
    """Depth of the ground plane z = 0. Rays at or above the horizon are NaN."""
    k, base_T_cam, rays = _camera(cam_z, pitch_deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(rays[..., 2] < 0, -cam_z / rays[..., 2], np.nan)
    return depth.astype(np.float32), k, base_T_cam

def synth_plane_with_box(cam_z=0.5, pitch_deg=20.0, box_x=2.05, box_h=0.6, half_w=0.3):
    """Ground plane plus the front face of a box at x = box_x (cell i = 20 at 0.10 m resolution)."""
    depth, k, base_T_cam = synth_plane(cam_z, pitch_deg)
    _, _, rays = _camera(cam_z, pitch_deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = box_x / rays[..., 0]
    y, z = t * rays[..., 1], cam_z + t * rays[..., 2]
    hit = (t > 0) & (np.abs(y) <= half_w) & (z >= 0) & (z <= box_h) & ~(t > depth)
    return np.where(hit, t, depth).astype(np.float32), k, base_T_cam

def test_depth_png_round_trip():
    depth = np.linspace(0.3, 5.0, 12, dtype=np.float32).reshape(3, 4)
    depth[0, 0] = np.nan
    ok, png = cv2.imencode(".png", depth.view(np.uint8).reshape(3, 4, 4))
    out = depth_from_rtabmap(png.tobytes())
    assert out.dtype == np.float32 and out.shape == (3, 4)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(depth))
    np.testing.assert_array_equal(out[~np.isnan(out)], depth[~np.isnan(depth)])

def test_flat_ground_is_height_zero_and_not_obstacle():
    depth, k, base_T_cam = synth_plane(cam_z=0.5, pitch_deg=20)   # helper in the test file
    t = make_tile(7, depth, k, base_T_cam, GridSpec(), NoiseModel())
    assert len(t.z) > 50 and np.all(np.abs(t.z) < 0.03) and np.all(t.obstacle_h == 0)

def test_box_is_obstacle_with_its_height():
    depth, k, base_T_cam = synth_plane_with_box()
    t = make_tile(7, depth, k, base_T_cam, GridSpec(), NoiseModel())
    at_box = (t.ij[:, 0] == 20) & (t.ij[:, 1] == 0)
    assert at_box.any() and np.all(t.obstacle_h[at_box] > 0.3)

def test_variance_grows_with_range_and_empty_depth_gives_empty_tile():
    depth, k, base_T_cam = synth_plane(cam_z=0.5, pitch_deg=20)
    t = make_tile(7, depth, k, base_T_cam, GridSpec(), NoiseModel())
    near, far = t.var[np.argmin(t.ij[:, 0])], t.var[np.argmax(t.ij[:, 0])]
    assert far > near
    empty = make_tile(8, np.full((48, 64), np.nan, np.float32), k, base_T_cam, GridSpec(), NoiseModel())
    assert len(empty.z) == 0
```
- [ ] Implement: back-project with stride, gate by range and z, transform to `base_link`, bin by `floor(xy / resolution)`; per cell `z = min`, spread = `max − min`; `obstacle_h = spread` if `spread > step_height`, or `max` if `abs(min) > max_ground_dev`, else 0; `var = (sigma0 + k * r**2) ** 2` at the cell's mean range.
- [ ] Run on the host: `python -m pytest ugv_nav/ugv_localization/test/test_elevation_decode.py ugv_nav/ugv_localization/test/test_elevation_tile.py -v`. Expected: PASS.

### Task 11: Store and fuse kernel

**Files:** Create `ugv_localization/elevation/{store,fuse}.py`, `test/test_elevation_fuse.py`, `test/test_elevation_store.py`.

**Consumes:** `Tile`, `Pose2D`, `GridSpec`, `NoiseModel` (Task 10).
**Produces:**
```python
@dataclass(frozen=True)
class ElevationGrid:
    origin_xy: tuple[float, float]   # map coordinates of the corner of cell (row 0, col 0)
    resolution: float
    height: np.ndarray       # (H, W) float32, NaN = unknown; row = y index, col = x index
    obstacle_h: np.ndarray   # (H, W) float32, 0 = none
    confidence: np.ndarray   # (H, W) float32 in 0..1, 0 where unknown
    version: int

def fuse(tiles: Mapping[int, Tile], poses: Mapping[int, Pose2D], spec: GridSpec,
         noise: NoiseModel, version: int) -> ElevationGrid           # tiles without a pose are ignored

@dataclass(frozen=True)
class GraphDelta:
    dropped: frozenset[int]   # ids we hold tiles for that left the graph
    moved: bool               # any shared id moved beyond the thresholds
def graph_delta(prev: Mapping[int, Pose2D], new: Mapping[int, Pose2D], have: Set[int],
                lin_thr: float = 0.05, ang_thr: float = 0.02) -> GraphDelta
def accept_tile(last: Pose2D | None, pose: Pose2D, min_trans: float = 0.5, min_rot: float = 0.35) -> bool
```

- [ ] Failing tests (Review Focus 2 and 3):
```python
def box_tile(node_id):   # one obstacle cell 2 m ahead plus ground cells
    ij = np.array([[20, 0], [10, 0], [11, 0]], np.int16)
    return Tile(node_id, ij, np.zeros(3, np.float32), np.full(3, 4e-4, np.float32),
                np.array([0.4, 0, 0], np.float32))

def cell(grid, x, y):
    c = int((x - grid.origin_xy[0]) / grid.resolution); r = int((y - grid.origin_xy[1]) / grid.resolution)
    return r, c

def test_pose_shift_moves_the_obstacle_and_leaves_no_ghost():
    tiles = {1: box_tile(1)}
    a = fuse(tiles, {1: (0.0, 0.0, 0.0)}, GridSpec(), NoiseModel(), 1)
    assert a.obstacle_h[cell(a, 2.05, 0.05)] > 0.3
    b = fuse(tiles, {1: (1.0, 0.0, 0.0)}, GridSpec(), NoiseModel(), 2)
    assert b.obstacle_h[cell(b, 3.05, 0.05)] > 0.3
    r, c = cell(b, 2.05, 0.05)
    assert b.obstacle_h[r, c] == 0            # the old position is ground or unknown, never an obstacle

def test_two_consistent_views_raise_confidence_and_disagreement_lowers_it():
    one = fuse({1: box_tile(1)}, {1: (0, 0, 0)}, GridSpec(), NoiseModel(), 1)
    two = fuse({1: box_tile(1), 2: box_tile(2)}, {1: (0, 0, 0), 2: (0, 0, 0)}, GridSpec(), NoiseModel(), 1)
    r, c = cell(one, 1.05, 0.05)
    assert two.confidence[r, c] > one.confidence[r, c]
    off = box_tile(3); off = Tile(3, off.ij, off.z + 0.5, off.var, off.obstacle_h)
    bad = fuse({1: box_tile(1), 3: off}, {1: (0, 0, 0), 3: (0, 0, 0)}, GridSpec(), NoiseModel(), 1)
    assert bad.confidence[cell(bad, 1.05, 0.05)] < one.confidence[r, c]

def test_no_tiles_gives_an_empty_grid():
    g = fuse({}, {}, GridSpec(), NoiseModel(), 1)
    assert g.height.size == 0

def test_ids_that_left_the_graph_are_dropped_and_restart_clears_everything():
    prev = {1: (0, 0, 0), 2: (1, 0, 0)}
    assert graph_delta(prev, {1: (0, 0, 0)}, have={1, 2}).dropped == frozenset({2})
    assert graph_delta(prev, {}, have={1, 2}).dropped == frozenset({1, 2})
    assert graph_delta(prev, {1: (0.2, 0, 0), 2: (1, 0, 0)}, have={1, 2}).moved is True
    assert graph_delta(prev, {1: (0.01, 0, 0), 2: (1, 0, 0)}, have={1, 2}).moved is False
```
- [ ] Implement `fuse`: rotate and translate each tile's cell centres by its pose, snap to map cells, concatenate, group with `np.unique(..., return_inverse=True)` and `np.bincount`; weights `w = 1/var`; `height = Σwz/Σw`; `sigma_total² = 1/Σw + (Σwz²/Σw − height²)`; `confidence = 1/(1 + sigma_total²/sigma_ref²)`; a cell is an obstacle when the weighted obstacle vote exceeds 0.5, with `obstacle_h` the weighted mean over the voting tiles.
- [ ] Run: `python -m pytest ugv_nav/ugv_localization/test/test_elevation_fuse.py ugv_nav/ugv_localization/test/test_elevation_store.py -v`. Expected: PASS.

### Task 12: Elevation node

**Files:** Create `nodes/elevation_map_node.py`, `config/elevation.yaml`; Modify `launch/localization.launch.py`, `setup.py`, `test/test_ros_stack.py`, `ugv_nav/docs/localization/interfaces.md`.

**Produces (both transient local, same `header.stamp`, frame `map`, at most 1 Hz, only on change):**
- `/ugv/elevation/cloud` — `PointCloud2`, one point per known cell at the cell centre, float32 fields `x, y, z, confidence, obstacle_h`.
- `/ugv/elevation/obstacles` — `OccupancyGrid`: 100 obstacle, 0 known, −1 unknown; `info` carries origin and resolution.

- [ ] Behaviour: subscribe `/rtabmap/mapData`; admit a node as a tile only if its id is in `graph.poses_id`, it has depth, and `accept_tile` passes (frames RTAB-Map publishes but does not keep are not graph nodes); apply `graph_delta`; re-fuse in a worker thread with newest-wins; bulk-load through `get_map_data` at start.
- [ ] Tests in `test_ros_stack.py`: after the moving synthetic run, both topics publish with equal stamps and the cloud has points; restarting the stack in `localize` mode on the saved database reproduces a grid with the same known-cell count (± 2 %).
- [ ] Verify on the Task 7 bag: view both topics in `mapping.rviz`; RTAB-Map's own `/rtabmap/cloud_ground` is the cross-check for the height layer.

---

## Phase 3 — Gateway transport (`ugv_api`)

### Binary format v1

Little-endian. Decoders reject an unknown `format` and any length that does not match exactly.

```
prelude (24 bytes, all layers)
 0 char[4] magic   "UGVC" cloud | "UGVE" elevation | "UGVT" trajectory | "UGVG" cost grid
 4 u16 format = 1          6 u16 header_bytes
 8 u32 epoch (random per gateway process)      12 u32 seq       16 f64 stamp_s

UGVC  header 64: 24 u32 count | 28 u32 source_count | 32 f32 spacing_m | 36 u32 flags (bit0 = has rgb)
                 40 f32[3] bbox_min | 52 f32[3] bbox_max
      body: f32 xyz[3*count], then u8 rgb[3*count] if bit0          = 64 + (12 or 15) * count
UGVE  header 48: 24 u32 width (+x) | 28 u32 height (+y) | 32 f32 resolution_m | 36 f32 origin_x | 40 f32 origin_y
                 44 u32 known_cells
      body: f32 height[w*h] row-major (NaN = unknown), u8 obstacle[w*h] (5 cm units, 0 = none, saturates at 255),
            u8 confidence[w*h] (0..255)                             = 48 + 6 * w * h
UGVT  header 32: 24 u32 count | 28 f32 length_m
      body: f32[7*count]  x,y,z,qx,qy,qz,qw                         = 32 + 28 * count
UGVG  header 48: 24 u32 width | 28 u32 height | 32 f32 resolution_m | 36 f32 origin_x | 40 f32 origin_y | 44 f32 origin_yaw
      body: i8[w*h] row-major (-1 unknown, 0..100)                  = 48 + w * h
```

### Task 13: Codec

**Files:** Create `ugv_api/mapcodec.py`, `test/test_mapcodec.py`, `test/fixtures/map/{cloud,elevation,trajectory,grid}.bin`.

```python
def cloud_view(*, fields, point_step: int, n_points: int, is_bigendian: bool, data: bytes) -> np.ndarray  # ValueError if xyz is not float32
def encode_cloud(xyz: np.ndarray, rgb: np.ndarray | None, *, epoch: int, seq: int, stamp_s: float, budget: int, spacing_m: float) -> bytes
def grid_from_cells(x, y, z, confidence, obstacle_h, *, origin_xy, resolution, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # sparse known cells (the /ugv/elevation/cloud fields) -> dense (height, obstacle_h, confidence); NaN height where unknown
def encode_elevation(height, obstacle_h, confidence, *, epoch, seq, stamp_s, origin_xy, resolution, max_side: int) -> bytes
def encode_trajectory(poses: np.ndarray, *, epoch, seq, stamp_s) -> bytes        # (N, 7) float32
def encode_grid(cells: np.ndarray, *, epoch, seq, stamp_s, resolution, origin_xy, origin_yaw) -> bytes
def decode_cloud(b: bytes) -> dict; decode_elevation(b) -> dict; decode_trajectory(b) -> dict; decode_grid(b) -> dict   # tests only
```
- [ ] Failing tests: each layer round-trips and has exactly the length in the spec; `grid_from_cells` puts three cells at the right row/column and leaves the rest NaN; non-finite points are dropped; `count <= budget`, and decimation is a spatial-hash selection so the same input always selects the same points; `encode_elevation` crops to the known bounding box and block-reduces to `max_side` (max height, max obstacle, min confidence); a big-endian or non-float32 cloud raises `ValueError`; (Review Focus 4) two encodes with different `epoch` and equal `seq` differ at bytes 8–11; golden `.bin` fixtures are byte-identical to fresh encodes.

### Task 14: Store, routes, SSE

**Files:** Create `ugv_api/mapstore.py`, `test/test_mapstore.py`, `test/test_map_api.py`; Modify `ugv_api/app.py`, `schemas.py`, `state.py`, `main.py`, `test/test_api_ros.py`.

```python
class MapStore:                                   # ROS-free, thread-safe
    LAYERS = ("cloud", "elevation", "trajectory", "grid", "live")
    epoch: int
    def put(self, layer: str, source: object, stamp_s: float) -> None      # swaps a reference, bumps seq
    def seq(self, layer: str) -> int                                        # 0 = nothing received
    def blob(self, layer: str, encode: Callable[[object, int, int], bytes]) -> bytes | None   # encodes once per seq
    def touch(self, now_s: float) -> None; def wanted(self, now_s: float, idle_s: float) -> bool
```
- [ ] `create_app(..., maps: MapStore | None = None)`. Routes, all sync `def` so encoding runs in the threadpool: `GET /api/v1/map` (JSON `MapStatus`: `epoch`, per-layer `seq`, `stats`), `GET /api/v1/map/{cloud|elevation|trajectory|grid|live}` → `Response(body, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})`. `GET /map` calls `maps.touch()`: it is the demand heartbeat.
- [ ] SSE: append `"map": map_view, "pose": pose_view` **after** the four existing entries (`app.py:258-261`) and update the exact-set assertion in `test_api_ros.py`.
- [ ] Failing tests with a fake `Robot` and `fastapi.testclient`: (Review Focus 5) every blob route returns 503 `application/problem+json` before any data; after `put`, the body decodes and repeated GETs encode once; `GET /map` reports the new `seq`.

### Task 15: ROS subscriptions, live cloud, costmap

**Files:** Modify `ugv_api/ros_node.py`, `config/api.yaml`, `package.xml`, `test/test_api_ros.py`, `.github/workflows/ugv_api.yml` (add `python3-numpy`, fixtures path).

- [ ] Subscriptions live in their own `MutuallyExclusiveCallbackGroup` and exist only while `maps.wanted()` (a 1 Hz timer creates and destroys them; `map.idle_timeout_s: 10`). KEEP_LAST 1, RELIABLE, durability chosen from `get_publishers_info_by_topic()`.

| Layer | Source | Notes |
|---|---|---|
| cloud | `/rtabmap/cloud_map` | budget `map.cloud_point_budget: 500000` |
| trajectory | `/rtabmap/mapPath` | |
| elevation | `/ugv/elevation/cloud` + `/ugv/elevation/obstacles` | used only when both stamps match; `map.elevation_max_side: 512` |
| grid | `/global_costmap/costmap` | sent as received |
| live | `/perception/depth/image` + `/camera/camera_info` K + TF `map ← frame_id` | back-projected at stride 4, range 0.3–8 m, no rgb |
| stats | `/ugv/map/stats`, `/ugv/perception/stats` | JSON merged into `MapStatus.stats` |

- [ ] `_poll_tf` (`ros_node.py:129-134`) stores the pose tuple `(x, y, z, qx, qy, qz, qw)` instead of `None`; `watches.py` reads only the stamp, so the §12 table is unaffected.
- [ ] ROS test: publish a real `PointCloud2` and `Path`, fetch over HTTP, decode, compare. Measure with a synthetic 1M-point publisher that the 0.5 s watches stay fresh while the map view is polling.

---

## Phase 4 — Web viewer (`ui/`)

### Task 16: Decoders

**Files:** Create `ui/src/map/codec.ts`, `codec.test.ts`.

```ts
export type CloudFrame = { epoch: number; seq: number; stampS: number; count: number; xyz: Float32Array; rgb: Uint8Array | null; bboxMin: [number, number, number]; bboxMax: [number, number, number] }
export type ElevationFrame = { epoch: number; seq: number; stampS: number; width: number; height: number; resolution: number; originX: number; originY: number; heightM: Float32Array; obstacle: Uint8Array; confidence: Uint8Array }
export type TrajectoryFrame = { epoch: number; seq: number; stampS: number; count: number; lengthM: number; poses: Float32Array }
export type GridFrame = { epoch: number; seq: number; stampS: number; width: number; height: number; resolution: number; originX: number; originY: number; originYaw: number; cells: Int8Array }
export function decodeCloud(buf: ArrayBuffer): CloudFrame | null
export function decodeElevation(buf: ArrayBuffer): ElevationFrame | null
export function decodeTrajectory(buf: ArrayBuffer): TrajectoryFrame | null
export function decodeGrid(buf: ArrayBuffer): GridFrame | null
```
- [ ] Failing vitest tests: each golden fixture from Task 13 decodes to the expected counts and first values; (Review Focus 5) wrong magic, unknown `format`, a body one byte short and an empty buffer all return `null`. Typed arrays are zero-copy views.

### Task 17: API client additions

**Files:** Modify `ui/src/source/api.ts`, `api.test.ts`.

```ts
export type Layer = 'cloud' | 'elevation' | 'trajectory' | 'grid' | 'live'
export type MapStatus = { epoch: number; seq: Record<Layer, number>; stats: Record<string, number | string | boolean | null> }
export type Pose = { x: number; y: number; z: number; qx: number; qy: number; qz: number; qw: number; ageS: number | null }
```
- [ ] Add `asMapStatus`, `asPose`, the two `PARSERS` entries, `Telemetry.map` / `.pose`, and `getBinary(path: string, signal: AbortSignal): Promise<ArrayBuffer | null>` (`null` on 503). Parser tests: a valid payload parses; a missing `epoch` or a non-numeric `seq` returns `null`.
- [ ] Guard test: replace the blanket `/costmap/i` with `/['"]\/[a-z_]*costmap\//` (direct costmap topic names stay banned). Map files stay **outside** the `LIVE_VIEW` allowlist, so the no-WebSocket and no-ROS-topic rules apply to them automatically.

### Task 18: Map data hook

**Files:** Create `ui/src/map/useMapData.ts` (+ scheduler tests in `codec.test.ts` or a sibling test).

```ts
export function nextFetches(status: MapStatus, have: Record<Layer, string | null>, enabled: Record<Layer, boolean>): Layer[]   // pure
export function useMapData(active: boolean, enabled: Record<Layer, boolean>): { cloud; elevation; trajectory; grid; live; status; stale: boolean }
export type MapData = ReturnType<typeof useMapData>
```
- [ ] Failing tests for `nextFetches`: a layer is fetched when its `epoch:seq` key differs from the one held; (Review Focus 4) an epoch change with an unchanged `seq` refetches every enabled layer; `seq` 0 and disabled layers are never fetched.
- [ ] Hook in the `useCameraSource` idiom: 1 Hz status poll while `active`, one in-flight fetch per layer, minimum intervals (cloud 2 s, elevation 1 s, live 0.5 s), `AbortController` cleanup, newest-result-wins counter.

### Task 19: Geometry builders

**Files:** Create `ui/src/map/geometry.ts`, `geometry.test.ts`.

```ts
export function buildElevationGeometry(f: ElevationFrame): { positions: Float32Array; indices: Uint32Array; cellOfVertex: Uint32Array }
export function colorElevation(f: ElevationFrame, cellOfVertex: Uint32Array, mode: 'height' | 'confidence' | 'obstacle'): Uint8Array
export function colorByHeight(xyz: Float32Array, zMin: number, zMax: number): Uint8Array
export function buildGridTexture(f: GridFrame): Uint8ClampedArray   // RGBA, unknown transparent
```
- [ ] Failing tests (no WebGL needed): a 3×3 grid with one NaN cell yields quads only where all four corner cells are known, so unknown regions are holes; obstacle cells are raised by `obstacle * 0.05` m; switching colour mode changes only the colour array.

### Task 20: Scene, MapView, layout switch

**Files:** Create `ui/src/map/scene.ts`, `ui/src/components/MapView.tsx`; Modify `ui/src/App.tsx`, `ui/src/index.css`.

- [ ] `MapScene` class in the style of `RingScene` (`ui/src/components/ui/glyph-ring.tsx`): `setCloud`, `setLive`, `setElevation`, `setTrajectory`, `setGrid`, `setPose`, `setLayers`, `resize`, `dispose`. `camera.up` = +Z before `OrbitControls` (from `three/addons`); points use a small `ShaderMaterial` with sRGB byte colours; **render on demand** (controls change, data change, pose event, resize), never a continuous loop, because the GPU is shared with SegFormer and DA3; `dispose()` frees geometries, materials, controls and calls `forceContextLoss()` (StrictMode double-mounts).
- [ ] `App.tsx`: `view: 'camera' | 'map'` state persisted to localStorage, rendering `<CameraView>` or `<MapView>` in grid area `view`. `MapView` reuses `.livefeed-bar` / `.livefeed-layers` for toggles (accumulated cloud, live cloud, trajectory, elevation, cost grid) and the elevation colour mode; it shows "no map yet" when every `seq` is 0 and a STALE banner over the last map when telemetry is not live.
- [ ] Verify by observation (`npm run dev`, real browser): replay the Task 7 bag, open the map view, confirm the cloud grows, the trajectory follows, the live cloud and cost-grid halo match `mapping.rviz`, toggles work, and switching to the camera view and back leaves no WebGL context warnings in the console.

### Task 21: Map statistics widget

**Files:** Modify `ui/src/components/StatusWidgets.tsx`.

- [ ] Add a `map` widget to the `WidgetId` union, `SIZE`, `LABEL` and `render` switch: keyframes, loop closures, path length, cloud points shown / source, elevation known cells, database size, depth rate, last update age, and a visible "placeholder calibration" flag.
- [ ] `npm run lint && npm test && npm run build`. Expected: all pass.

---

## Phase 5 — Documentation and end-to-end proof

### Task 22: Docs, CI, UGV run

**Files:** Create `docs/mapping/README.md`; Modify `ugv_nav/docs/localization/interfaces.md`, `ui/README.md`, `ugv_nav/ugv_bringup/README.md`, `.github/workflows/ui.yml`, `mindmap.md` (move any item settled during execution out of "Open / to verify").

- [ ] `docs/mapping/README.md`: data flow, the topic and endpoint tables from this plan, how to run mapping and view it, and the honest limits (monocular scale wobble, 5 m fusion range, height relative to the driving plane, placeholder calibration).
- [ ] End-to-end run on the UGV with the calibrated phone camera: drive a closed loop in `mapping` mode, save the database, restart in `localize` mode. Evidence to capture in `docs/mapping/`: a screenshot of the web map view and of RViz for the same run, the stats values, and the four gate numbers from Task 7 re-measured.

---

## Verification

| Level | Command | Proves |
|---|---|---|
| Perception kernels | `cd turing && python -m pytest` | timing, depth independence, parity |
| Localization kernels | `python -m pytest ugv_nav/ugv_localization/test -k "elevation or mapstats or calib"` | tile, fuse, store, stats |
| ROS stack | container: `colcon test --packages-select ugv_localization ugv_bringup ugv_api && colcon test-result --verbose` | real RTAB-Map 3D outputs, elevation node, gateway round trip |
| Gateway | `python -m pytest ugv_nav/ugv_api/test -k "map"` | codec, store, endpoints |
| UI | `cd ui && npm run lint && npm test && npm run build` | decoders, scheduler, geometry, guard |
| Observation | RViz (`mapping.rviz`) and the web map view on the recorded bag, then on the UGV | the map actually accumulates and matches between the two viewers |

Container commands assume the existing `ugv-run` container and its `/ws/sync_ws.sh` build script (currently stopped; start and re-verify before use).

## Execution

22 tasks across five packages, joined by three contracts (elevation topics, binary format v1, `MapStatus`). Recommended: subagent-driven, one fresh implementer and reviewer per task, because the tasks are independently testable and a wrong byte layout or frame convention would otherwise surface only at the end. Hardware steps (phone calibration, latency measurement, the recorded run, the UGV run) are the owner's and are marked in Tasks 6, 7 and 22.

## Out of scope

Feeding elevation into Nav2 (needs a §9 precedence decision); 6-DoF pose or phone IMU; semantic colours or classes in the 3D map; slope and roughness layers; bag replay of perception itself (the node uses wall-clock time, `adapter_node.py:106`); undistorting depth back-projection; the `sim` and `bag` bringup profiles.

## Known risks

| Risk | Detected by |
|---|---|
| Visual odometry on DA3 depth is not stable enough to fuse | Task 7 gate, before any elevation work |
| rtabmap 0.23.7 node data differs from what was read (depth PNG layout, which nodes carry data) | Task 8 `test_x3` and Task 10 round-trip against the real binary |
| Map assembly delays `/rtabmap/info` and trips `slam_stale` | Task 8 risk check on the bag |
| Gateway work starves the 0.5 s safety watches | Task 15 measurement with a 1M-point publisher |
| Browser lands on the integrated GPU or loses its context under VRAM pressure | Task 20 observation; point budget is a config value |
