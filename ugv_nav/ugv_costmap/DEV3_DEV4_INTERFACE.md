# Dev 3 → Dev 4 interface

Status (2026-10-02): **live.** Dev 3's `semantic_costmap_node` publishes the semantic grid and Dev 4's Nav2
costmaps read it (`src/ugv_navigation/config/costmaps.yaml`, the navigation launch default). Dev 4's side of
the same contract is `src/ugv_navigation/DEV4_INTERFACES.md` §3. `architecture.md` wins on conflict.

Package: `ugv_nav/ugv_costmap/` (ROS package `ugv_costmap`) = `costmap_core/` (ROS-independent core, see
[`costmap_core/README.md`](costmap_core/README.md)) + `ugv_costmap/` (ROS node and ROS-free adapter) +
`config/semantic_costmap.yaml` + `launch/semantic_costmap.launch.py`. Dev 4's Nav2 package is
`src/ugv_navigation/`.

```
Dev 1 /segmentation/mask ─► Dev 3 semantic_costmap_node ─► /semantic_costmap/grid ─► Nav2 StaticLayer ─┐
Dev 1 /perception/depth_cloud ───────────────────────────────────────────────────► Nav2 VoxelLayer ───┤
                                                                                   Nav2 InflationLayer ┘
        Nav2 publishes /global_costmap/costmap (map) and /local_costmap/costmap (odom) ─► Smac2D + RPP
```

## 1. Dev 3 output (the wire)

| Item | Value |
|---|---|
| Topic / type | `/semantic_costmap/grid`, `nav_msgs/msg/OccupancyGrid` (`output_topic`) |
| QoS | reliable, transient local, keep last 1 |
| Frame / stamp | `base_link` (`robot_frame`); stamp = the mask's stamp |
| Extent | 6 m ahead (`max_range`) × ±4 m (`half_width`), 0.1 m cells (`resolution`), origin shifted by the camera mount offset |
| Rate | one grid per mask (mask pooled by `mask_downsample: 4` before projection) |
| Values | traversable `0` · hazard `100` · observed class 0 `unknown_occupancy` = `50` (never free, architecture §8.1) · outside the camera's view `-1` |
| Inflation | none in Dev 3 (`inflation_radius=0`): Nav2's `InflationLayer` inflates semantic and geometry lethal once |
| Fail-safe | mask stamp older than `max_mask_age_s` (0.5 s) or in the future, or no mask for 0.5 s → whole field of view `100`, rest `-1` (§8.4, §8.6) |

Nav2 side (Dev 4's file, both costmaps): StaticLayer `semantic_layer` on this topic, VoxelLayer on
`/perception/depth_cloud` with `combination_method: 1` (Max), so **geometry lethal always wins** over
semantic traversable and free geometry never lowers a semantic hazard (§9); `track_unknown_space: false`,
so `-1` (never observed) plans as free: an accepted §8.1 exception for never-observed cells only
(`mindmap.md` D22).

## 2. Dev 3 inputs

| Input | Source | Use |
|---|---|---|
| `/segmentation/mask` | Dev 1, `mono8`, `{0 unknown, 1 traversable, 2 hazard}`, stamp = image time, `frame_id` = optical frame | main input. Dev 1 publishes a mask only when it is valid, so the node judges it by its **own stamp age**. `/segmentation/port_meta` is not used: it carries no stamp and arrives after its mask, so it cannot be paired with the sample it describes |
| `/camera/camera_info` | Dev 5 camera driver (reliable, transient local) | intrinsics. The single CameraInfo topic for Dev 3 (`camera_info_topic`). A mask whose `frame_id` or size differs from it is refused. Dev 1's `/segmentation/camera_info` republish is not used |
| TF `base_link ← <mask frame>` | Dev 5 URDF (static camera mount) | camera height, pitch and offset. Roll or yaw > 2° is refused (the core models height + pitch only) |

Not used by Dev 3: `/segmentation/confidence`, `/ugv/perception_degraded` (Dev 5 consumes it).

## 3. Internal core costs (`costmap_core`)

| Core value | Meaning | On the wire |
|---|---|---|
| `0` | traversable (class 1) | `0` |
| `254` | hazard (class 2) or geometric occupancy | `100` |
| `255` | unknown: class 0, or a cell no mask pixel reached | `50` if the camera covers the cell, else `-1` |

Rules enforced in the core: geometry lethal always wins; several pixels in one cell resolve HAZARD > UNKNOWN >
TRAVERSABLE (the node's pooling keeps the same precedence); unknown is never turned into free; inflation never
lowers a cost. `ugv_costmap/adapter.py` `costs_to_occupancy` is the only conversion.

## 4. Still open

- **Footprint:** Dev 5's footprint YAMLs do not exist yet; Nav2 runs the placeholder in `costmaps.yaml` and
  logs a warning (`mindmap.md` D25). Replace before outdoor runs.
- **Rectification:** the mask is computed on `/camera/image_raw`; the core ignores distortion
  (`costmap_core/README.md` §8).
- **Multi-resolution inflation** (dev.md Dev 3 task 4): Nav2's single InflationLayer only.

## 5. Synthetic costmaps for Dev 4 tests

`costmap_core.pipeline.run_costmap_pipeline` turns synthetic inputs into core costs; Dev 4's closed-loop
fixture (`src/ugv_navigation/test/fixtures/test_only_closed_loop_costmaps.yaml`) publishes scenario grids on
`/test/scenario_map`. Sizes and positions there are test values.

```python
import numpy as np
FREE, LETHAL, UNKNOWN = 0, 254, 255
grid = np.full((50, 50), FREE, dtype=np.int64)   # [row=y, col=x]
grid[25, :] = LETHAL                             # wall
grid[25, 20:24] = FREE                           # opening
```

## 6. Evidence

`python3 -m pytest` in `ugv_nav/ugv_costmap/` (core + adapter). `colcon test --packages-select ugv_costmap`
in the Lyrical container runs the same suite.
