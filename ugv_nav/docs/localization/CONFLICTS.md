# architecture.md / dev.md vs what Dev 2 builds

| # | Source wording | Reality / architecture | What we do |
|---|---|---|---|
| C1 | architecture §2/§6 "RTAB-Map VO/SLAM" | Hardware is mono (D2). With DA3 depth (D7) RTAB-Map runs in RGB-D mode, so its VO works | `rgbd_odometry` on RGB + DA3 depth = visual odometry; RTAB-Map RGB-D SLAM corrects `map->odom` |
| C2 | architecture §6/§10 "stereo/RGB-D recommended, mono minimum" | DA3 makes a mono camera *look* RGB-D to RTAB-Map, but it is learned per-frame depth, not a depth sensor | We are still on the **minimum** tier: drift + DA3 depth error are documented (SENSOR_HONESTY.md); never claim RGB-D-grade |
| C3 | architecture §5/§6/§9 "optional Depth Anything 3 Metric Large → VoxelLayer (geometry side-channel)" | Dev 2 now **depends** on DA3 depth for SLAM — a Dev 1 → Dev 2 topic dependency the architecture does not show | Proposed amendment below (C10). Dev 2 fails closed without depth (`depth_missing` / `depth_stale`) |
| C4 | architecture §10 "VO-lost hold" | Two meanings now: visual odometry lost (rgbd_odometry), and no map constraint for too long | VO lost → selector drops the sample (auto falls back to wheel; visual-only → `/odom` stops → `odom_stale`). No loop closure / proximity for `max_dead_reckon_m` in localize → `dead_reckon_distance` |
| C5 | dev.md task 1 "stereo (recommended) / RGB-D / mono (minimum)" configs | User decision 2026-09-28: RGB-D only | Only `rtabmap_rgbd.yaml` ships; `rtabmap_mono.yaml` deleted. No mono fallback — Dev 2 cannot run without depth |
| C6 | dev.md task 1 "tune visual feature tracking, bundle adjustment, keyframing" | — | Tuned: features (GFTT/BRIEF), depth caps (`Vis/MaxDepth`, `Kp/MaxDepth`), node spacing, neighbor-link refining, proximity detection, loop-closure rejection. Values untuned until sim runs |
| C7 | dev.md §3 TF owner = Dev 2; architecture silent | Two odometry sources could both claim `odom->base_link` | `odom_selector` is the **only** publisher (D6); `rgbd_odometry` runs with `publish_tf:=false`; Dev 5 diff-drive must set `publish_odom_tf=false` |
| C8 | dev.md task 6 "rosbag playback" | Profiles §4: `bag` is eval only | `bag_eval.launch.py` replays inputs only; TF and `/clock` regenerated |
| C9 | dev.md §3 "TF publish rate ≥ 15 Hz, jitter < 50 ms" | `odom->base_link` rate = selected odometry rate. Wheel: Dev 5's rate. **Visual: DA3 depth rate** (not measured; DA3-Large may be < 15 Hz) | Not faked (no re-stamping / extrapolation). `auto` with wheel alive meets it; `visual` may not — measure with `tf_rate_check`, report to Dev 3/4/5. `map->odom` is 20 Hz regardless |
| C10 | Nav2 global costmap may expect `/map` from SLAM | RGB-D mode gives RTAB-Map an occupancy grid again | Dev 2 publishes `/map` from DA3 depth; **optional** for Dev 3 (quality = DA3 quality) |
| C11 | dev.md hours / difficulty | Soft aim ~30h (§17) | Not a design constraint |

## Proposed architecture.md / dev.md amendment (needs owner approval — not applied)

architecture.md has highest authority, so Dev 2 does not edit it. Suggested text for the owner:

- **§5 context diagram / §6 tech stack, "Optional geometry" row:** "Depth Anything 3 Metric Large (Dev 1) publishes
  a metric point cloud (`/perception/depth_cloud`). **Required** input to RTAB-Map RGB-D (localization; Dev 2
  converts it to a depth image); optional geometry
  side-channel → VoxelLayer (costmaps)."
- **§6 "Vision recommended / minimum":** "Stereo·RGB-D / mono. Current hardware: mono + DA3 pseudo-depth (still the
  minimum tier: drift docs + VO-lost hold)."
- **§9 first line:** "Depth Anything 3 Metric Large → VoxelLayer is a geometry side-channel **for costmaps**, not a
  second perception port. The same depth also feeds RTAB-Map (§10)."
- **§10:** "Odometry: wheel (Dev 5) or visual (RTAB-Map rgbd_odometry on DA3 depth), selectable; `auto` prefers wheel."
- **dev.md §3 table:** new row `/perception/depth_cloud` · `sensor_msgs/PointCloud2` x/y/z m · **Dev 1** → **Dev 2**, Dev 3 ·
  "stamp = source image stamp, frame = camera optical frame, back-projected with the raw K at camera resolution".
