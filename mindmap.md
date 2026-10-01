# Dev 2 mindmap — SLAM & Localization decisions

**Owner:** Dev 2 · **Authority:** architecture.md > dev.md (only on conflict) · **Date:** 2026-09-23 · **Re-plan:** 2026-09-28 (mono + DA3 depth)

## Decisions
| # | Topic | Decision | Source / reason |
|---|---|---|---|
| D1 | Runtime | **ROS 2 Lyrical Luth** (CLAUDE.md overrides architecture/dev.md "Jazzy"). Devs work independently; no access to Dev 1's box. Lyrical core is Tier 1 on **Windows 11**, but rtabmap_ros / Gazebo are not confirmed on Windows → Dev 2 runs ROS in **WSL2 Ubuntu 26.04 + Lyrical** (distro `Ubuntu-26.04`; rtabmap_ros 0.23.7, Gazebo **Jetty** + ros_gz, RViz2); kernels/tests run on Windows Python. Code stays distro-agnostic | User 2026-09-23 / 2026-09-24 |
| D1a | Build scope | Build everything that does **not** need the sim camera / Dev 1 depth: docs, configs, pure kernels (TDD), rclpy nodes + launch, verified on Lyrical without sensors | User |
| D2 | Sensor | ~~Mono camera only~~ → **Mono camera + Depth Anything 3 Metric Large depth** (see D7). Still the architecture's *minimum* tier (§6/§10): drift docs + VO-lost hold mandatory | User 2026-09-28 (superseded 2026-09-23 "mono only") |
| D3 | Test data | **Dev 5 Gazebo sim camera = main focus**; real-world later. Sim also provides a GT depth camera for DA3 accuracy + bring-up | User. Sim/bag = eval profiles (§4), not product proof |
| D4 | Layout | **Architecture §7**: `ugv_nav/ugv_localization/`, `ugv_nav/config/cameras/`, `ugv_nav/docs/` | User |
| D5 | Metric scale | ~~Wheel odometry from Dev 5~~ → superseded by D5b | User 2026-09-23 |
| D5b | Odometry | **Switchable `odom_source:=visual|auto|wheel`** — default **`visual`** since 2026-09-29 (no wheel sensor; simulated wheel odom + real images would disagree). Previously default `auto`: wheel odom (Dev 5) when available, else RTAB-Map `rgbd_odometry` on RGB + DA3 depth. `odom_selector` re-anchors on every switch so `odom` never jumps | User 2026-09-28 |
| D6 | TF ownership | **Dev 2 owns full `map->odom->base_link`**. Only `odom_selector` publishes `odom->base_link`; `rgbd_odometry` runs `publish_tf:=false`; Dev 5 publishes `/wheel/odom` topic only (no TF) | User. Architecture silent → dev.md §3 contract applies |
| D7 | SLAM mode | **RGB-D only.** RTAB-Map subscribes to one `rgbd_sync` RGBDImage (camera RGB + DA3 depth, exact stamps). `rtabmap_mono.yaml` deleted; **no mono fallback** — without depth Dev 2 holds (`depth_missing`) | User 2026-09-28 |
| D8 | Depth source | **Dev 1 publishes a DA3 depth image** `/perception/depth/image` (32FC1 m, NaN holes, RGB stamp + frame + size; Dev 1 `af7ebbf`) → launch default `depth_input:=image`. Fallback `depth_input:=cloud`: `/perception/depth_cloud` converted by `rtabmap_util/pointcloud_to_depthimage`. Dev 2 never runs DA3. Contract: `ugv_nav/docs/localization/interfaces.md` "Depth input" | User 2026-09-29 |
| D9 | Map products | RTAB-Map `Grid/3D true` + `cloud_map`/`mapPath`/`mapGraph`/`mapData` consumed; new elevation map and map-stats nodes in `ugv_localization`. Architecture §5–§7 name no map products. Mapping and display only: not a second perception port, not a Nav2 input (§9 unchanged). | User 2026-10-02 |
| D10 | Camera transport | Phone camera on the UGV streamed through a network tunnel to the laptop. Stamps are **arrival time** minus a measured `transport_latency_s`, where §8.4 says "image time". | User 2026-10-02 |
| D11 | Calibration | `phone_640x480.yaml` seeded from the laptop calibration with `placeholder: true` until the phone is calibrated. §17 assumes calibrated vision and `config/cameras/README.md` forbids placeholders; mapping runs for the Phase 0 gate require the real calibration. | User 2026-10-02 |
| D12 | Elevation semantics | `Reg/Force3DoF true` kept; elevation height is relief relative to the driving plane. General-purpose defaults, four layers only. | User 2026-10-02 |
| D13 | Operator UI | Map, live cloud and Nav2 costmap shown in the web UI through `ugv_api` binary endpoints, display only. The UI guard's blanket ban on the word "costmap" is narrowed to costmap topic names. | User 2026-10-02 |
| D14 | Depth vs mask | DA3 depth is published for every processed frame, no longer only when a mask was published, so a degraded mask does not starve SLAM. | User 2026-10-02 |

## What architecture.md says (and doesn't)
- §2/§6: brain = "RTAB-Map VO/SLAM" — now true in RGB-D mode (DA3 depth).
- §6/§10: stereo/RGB-D recommended; **mono minimum + drift docs + VO-lost hold** — DA3 pseudo-depth keeps us on the minimum tier.
- §5/§6/§9: DA3 = "optional geometry side-channel → VoxelLayer" — **does not show the Dev 1 → Dev 2 depth dependency** (CONFLICTS C3; amendment drafted, not applied).
- §8.5/§10.1/§12: required TFs must exist; missing TF → invalid pose → hold. No TF owner named.
- Kill list: "mono = recommended outdoor RTAB-Map", "ORB-SLAM3 bake-off".

## Consequences
- Dev 1's depth image / cloud only exist once `af7ebbf` is merged and the DA3 weights are fetched + exported (the IR is not in the repo).
- Dev 5: `/wheel/odom` with covariance + `publish_odom_tf=false`; sim GT depth camera co-located with the RGB camera; textured world; GT pose.
- RTAB-Map produces an occupancy `/map` again (from DA3 depth) — **optional** for Dev 3.
- `odom_source:=visual` → `odom->base_link` rate = DA3 rate; may violate the ≥ 15 Hz TF contract (CONFLICTS C9). Measure, don't fake.
- Drift + DA3 depth benchmarks use Gazebo ground truth (pose + depth).

## Rejected
- Wheel+IMU EKF (no IMU planned; robot_localization on Lyrical uncertain) — revisit later.
- Dev 5 owning `odom->base_link` → splits TF ownership vs dev.md.
- Mono (RGB-only) fallback profile → user chose RGB-D only (2026-09-28).
- Dev 2 running its own DA3 instance → second DA3-Large on one GPU; Dev 1 publishes instead (D8).
- ~~Re-projecting Dev 1's point cloud into a depth image → lossy~~ — **reversed** 2026-09-29 (D8): re-projection with the same K at the same resolution is pixel-exact (verified in `test_ros_stack.py`).
- ~~Depth Anything pseudo-RGB-D~~ — rejected 2026-09-23 for the Dev1→Dev2 dependency; **reversed** 2026-09-28 (D7/D8): the team switched to DA3 depth; dependency made explicit in interfaces.md + CONFLICTS C3.

## Open / to verify
- DA3 latency / fps on the target GPU (Dev 1) → `sync_queue_size`, `depth_max_age_s`, visual-mode TF rate.
- Values in `rtabmap_rgbd.yaml` / `rgbd_odometry.yaml` (depth caps, grid heights, neighbor refining) — tune in sim.
- Dev 5 topic names: camera image/info, GT depth camera, `/wheel/odom`, ground-truth pose.
- Owner approval of the architecture.md / dev.md amendment (CONFLICTS.md).
