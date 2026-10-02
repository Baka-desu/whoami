# Dev 4 interfaces: hand-off to Dev 2, Dev 3 and Dev 5

What the Dev 4 planning/control stack (`ugv_navigation`, `navigation.launch.py`)
consumes and produces, and what it needs decided by the other devs. Companion to
Dev 3's `ugv_nav/ugv_costmap/DEV3_DEV4_INTERFACE.md`. `architecture.md` wins on conflict.

Labels: **IMPLEMENTED** (in this package, tested on ROS 2 Lyrical / Nav2 1.5.1) ·
**PROPOSED** (Dev 4 proposal, needs the other dev's agreement) · **OPEN** (not decided) ·
**DECIDED** (agreed, recorded in `mindmap.md`).

Evidence: `src/ugv_navigation/docker/test_in_lyrical.sh src/ugv_navigation` runs every test below.

## 1. Topics and actions

| Name | Type | Direction | Status | Contract |
|---|---|---|---|---|
| `/cmd_vel_nav2` | `geometry_msgs/msg/Twist` (unstamped) | Dev 4 → Dev 5 | IMPLEMENTED | Candidate only (§3.1). Published by `controller_server` (~20 Hz while following a path) and `behavior_server` (spin / backup recoveries). One zero twist when a goal ends. **Silent when idle and during the Wait recovery.** Bounds: \|v\| ≤ 0.4 m/s, \|ω\| ≤ 0.8 rad/s (placeholders, §5); v < 0 only during BackUp (0.10 m/s, 0.30 m) |
| `/cmd_vel` | — | — | IMPLEMENTED | **No Dev 4 node publishes it.** Checked live by `test_navigation_graph`, `test_closed_loop` and `nav_goal_testbench check` |
| `/ugv/nav2_heartbeat` | `std_msgs/msg/Bool` | Dev 4 → Dev 5 | IMPLEMENTED | 20 Hz, every tick. `true` only if planner, controller, behavior server and bt_navigator all answered lifecycle `get_state` = `active` within 0.5 s. `false` at startup, within one poll (0.2 s) of any server reporting non-active, and within 0.5 s + one poll (0.1 s) of one crashing or hanging. A single lost `get_state` reply is retried after 0.2 s, so it never makes a healthy server stale. If the heartbeat process itself is starved of CPU, it does not blame Nav2 for its own pause (age check resumes after one fresh poll round). Verified: 15/15 live runs with no false `false`, and `false` after a killed server. Same shape as Dev 2's `/ugv/pose_valid` |
| `/ugv/nav2_status` | `std_msgs/msg/String`, transient local | Dev 4 → any | IMPLEMENTED | `ok` or the reason, e.g. `controller_server stale (no reply for 0.61 s)`. Published on change |
| `/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | operator / mission → Dev 4 | IMPLEMENTED | Goals in `map`. Result `error_code` / `error_msg` from `compute_path`, `follow_path`, `spin`, `wait`, `backup` |
| TF `map → odom → base_link`, `/odom` | TF, `nav_msgs/msg/Odometry` | Dev 2 → Dev 4 | IMPLEMENTED (consumer) | Matches Dev 2's `odom_selector` output. Nav2 activation waits for the TF |
| `/semantic_costmap/grid` → global / local costmaps | `nav_msgs/msg/OccupancyGrid` → Nav2 StaticLayer | Dev 3 → Dev 4 | IMPLEMENTED (§3) | `config/costmaps.yaml` (launch default; `costmap_params_file:=` overrides) |

## 2. For Dev 5 (safety authority, bringup, platform)

**Watchdog.** Use `/ugv/nav2_heartbeat` for the §12 "Nav2 crash / no controller
heartbeat" row: hold on `false` **or** on message age > your timeout (the heartbeat
node itself can die). Do **not** use `/cmd_vel_nav2` age as Nav2 liveness: it is
silent whenever no goal is running and for 5 s during the Wait recovery. Keep a
separate freshness check on the candidate: a `/cmd_vel_nav2` older than ~0.5 s must be
treated as zero at Level 4 (never repeat the last candidate). That is a candidate
property, not a health fault.

**Bringup include** (IMPLEMENTED): `ugv_bringup/launch/bringup.launch.py` includes
`ugv_navigation navigation.launch.py` with only `robot:=<bringup robot arg>` (default empty). The costmaps
come from this package's `config/costmaps.yaml`; `costmap_params_file:=` and `use_sim_time:=` exist for other
callers (the closed-loop tests use them).

Launches only: `planner_server`, `controller_server`, `behavior_server`,
`bt_navigator`, `lifecycle_manager_navigation` (by default composed into one
`nav2_container` process; `use_composition:=false` for separate processes) and
`nav2_heartbeat` (always its own process, so it reports `false` if the container dies).
Node, topic and service names are the same in both modes. No camera, TF, map server,
velocity smoother, collision monitor or motor driver.

**Robot limits** (OPEN, Dev 5 values needed). Fill
`config/robots/{primary,secondary}/nav2_limits.yaml` per platform: max linear speed,
max angular speed, linear accel / decel, angular accel / decel, rotate-in-place speed,
cancel deceleration, recovery spin speeds. Current values are placeholders
(0.4 m/s, 0.8 rad/s, 0.5 / -1.0 m/s², ±2.0 rad/s²). A test checks each file
overrides exactly the `[ROBOT LIMIT]` keys and that the limits are consistent.

**Safety hold vs. navigation** (OPEN, decision needed). When you hold `/cmd_vel` at
zero (perception degraded, pose invalid), Nav2 does not know: after 10 s without
0.5 m of progress the controller fails, spin / wait / backup run (also held), and
after 6 retries the goal is ABORTED. Options:

| Option | Behaviour | Needs |
|---|---|---|
| A. Accept abort | Operator / mission re-sends the goal after the hold | Nothing |
| B. Hold-aware BT (Dev 4 recommends) | BT pauses (no progress check, no recoveries) while a hold is active, resumes the same goal after | Dev 5 publishes e.g. `/ugv/safety_hold` (`std_msgs/Bool`, 20 Hz); Dev 4 adds the BT condition |
| C. Longer progress timeout | Tolerates short holds only | Nothing; masks real "stuck" cases |

## 3. For Dev 3 (costmaps)

**Dev 3's grid in Nav2** (IMPLEMENTED, live in `config/costmaps.yaml`, the launch default). Nav2's
planner and controller cannot subscribe to an external costmap topic: they host `global_costmap` /
`local_costmap`, and those publish `/global_costmap/costmap` and `/local_costmap/costmap` themselves.
So Dev 3 publishes a layer input and Nav2 composes it:

| Piece | Live value |
|---|---|
| Publisher | `ugv_costmap` `semantic_costmap_node` (Dev 3, `ugv_nav/ugv_costmap/`) |
| Topic / type | `/semantic_costmap/grid`, `nav_msgs/OccupancyGrid`, reliable + transient local, depth 1 |
| Frame | `base_link` (a 6 m × 8 m wedge ahead of the camera, 0.1 m cells; stamp = mask stamp). StaticLayer transforms it into the `map` (global) and `odom` (local) costmaps |
| Values | traversable `0` · hazard `100` · observed class 0 (`unknown`) `50` (never free, cost ≈ 127) · outside the camera's view `-1`. Un-inflated: inflation happens once, in Nav2's `InflationLayer` |
| Fail-safe | mask stamp older than 0.5 s, or no mask for 0.5 s → the whole field of view `100` (§8.4, §8.6) |
| Nav2 layers (both costmaps) | `semantic_layer` StaticLayer (`map_topic: /semantic_costmap/grid`, `trinary_costmap: false`, `use_maximum: false`), `voxel_layer` VoxelLayer on `/perception/depth_cloud` with `combination_method: 1` (Max: geometry lethal always wins over semantic traversable, §9), `inflation_layer` (radius 0.6, `cost_scaling_factor` 3.0) |
| Unseen space | `track_unknown_space: false`: `-1` cells plan as free (DECIDED, mindmap D22, below) |

**Required costmap settings** (for Smac2D / RPP to behave as designed):

| Setting | Why |
|---|---|
| Inflation `inflation_radius` ≥ half the robot's largest cross-section | Smac2D collision checking; otherwise it logs *"inflation is not set sufficiently"* |
| Local inflation `cost_scaling_factor` = RPP `inflation_cost_scaling_factor` (3.0) | RPP recovers obstacle distance from cost; tell Dev 4 if you change it |
| Local costmap update ≥ 5 Hz; global ≥ 1 Hz | How fast new hazards reach RPP / the 2 Hz replanning |
| `footprint` or `robot_radius` in both costmaps | Smac2D 2D plans a circle (inflation); RPP collision checks the footprint |

`inflate_around_unknown` is not set in the product file: with `track_unknown_space: false` there are no
NO_INFORMATION cells to inflate around. The closed-loop fixture
(`test/fixtures/test_only_closed_loop_costmaps.yaml`) runs the stricter variant (`track_unknown_space: true`
plus `inflate_around_unknown: true`) to prove the planner keeps clear of unknown blocks.

**Unknown-space policy** (DECIDED, owner 2026-10-02, mindmap D22). The robot sees only its camera wedge, so
space it has never observed is planned as free (`track_unknown_space: false`); otherwise no goal beyond what
was already seen is reachable. What the camera has seen as class 0 is never free (`50`), and a stale or
missing mask turns the whole view lethal. This is an accepted exception to architecture §8.1 for
never-observed cells only.

**Package location** (DECIDED, mindmap D20). Dev 3's package is `ugv_nav/ugv_costmap/` (ROS package
`ugv_costmap`: `costmap_core` + `semantic_costmap_node`); Dev 4's Nav2 package is `src/ugv_navigation/`.
Two packages, one folder each, folder name = package name.

## 4. Cross-dev findings (not Dev 4's to fix)

**TF `base_link` parent (Dev 2 × Dev 5): RESOLVED.** An earlier Dev 5 URDF (#15) made `base_footprint` the
parent of `base_link` while Dev 2's `odom_selector` publishes `odom → base_link`, giving `base_link` two
parents. The live `ugv_robot_description/urdf.py` roots at `base_link`, so the contract `map → odom →
base_link` (dev.md §3) holds and Nav2 keeps `robot_base_frame: base_link`.

## 5. Per-robot configuration

`ros2 launch ugv_navigation navigation.launch.py robot:=<name>` selects, per robot:

| What | File | Owner | Applied to |
|---|---|---|---|
| Speed / acceleration limits | `config/robots/<name>/nav2_limits.yaml` (this package) | Dev 4 file, Dev 5 values | controller + behavior server |
| Footprint (`footprint`, `footprint_padding`) | Dev 5's `config/robots/footprint_<name>.yaml`, found by searching upward from this package (source tree, or an install space built inside the repo) | Dev 5 (selection is Dev 4's, per Dev 5) | both costmaps |

Parameter order: Dev 4 defaults < Dev 3 costmap params < footprint < robot limits.
`footprint_file:=<path>` overrides the footprint lookup; `robot_params_file:=<path>`
replaces the limits file (not together with `robot:=`). Without either, the footprint
is whatever `config/costmaps.yaml` sets: a Jackal-class PLACEHOLDER, and the launch logs a warning
saying so (mindmap D25: replace it with Dev 5's real footprints before any outdoor run, architecture §13
item 9). The closed-loop tests run every scenario with both of Dev 5's footprints (TEST-ONLY copies from
#15 until it is merged).

## 6. Evidence (Lyrical, Nav2 1.5.1)

| Test | Proves |
|---|---|
| `test_dev4_config` | Every plugin / parameter / BT node exists in the installed Nav2; contracts above |
| `test_navigation_graph` | Full stack activates; only controller + behavior server publish `/cmd_vel_nav2`; nothing publishes `/cmd_vel`; heartbeat `false` while configured, `true` when active, `false` after `controller_server` is killed |
| `test_closed_loop` | A→B in 5 scenarios (open, wall with opening, corridor, unknown block, hazard appearing mid-run) × Dev 5's 2 footprints, TEST-ONLY fake base: goal reached, footprint polygon clear of lethal and unknown cells, twists within limits. Benchmark CSV in the build dir; numbers in `README.md` |
| `test_heartbeat_core`, `test_testbench_core` | Heartbeat and testbench logic |
