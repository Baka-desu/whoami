# Dev 1 in the architecture

## Product sentence

Installable ROS 2 system: calibrated vision on a differential-drive UGV → A→B outdoors, GPS-denied. Vision is the **primary** sensor. Sim, bags, RUGD, tutorial ONNX are **eval / bring-up profiles**, not the product.

Dev 1’s slice of that sentence:

> Live camera frames become a **fresh, valid, 3-class mask** (and optional depth) that Nav2 costmaps and the safety mux can trust without knowing which model produced them.

## Dual fan-out (architecture §5)

The camera splits **in parallel**. Dev 1 owns the left branch plus optional depth. Dev 2 owns the right branch and **must not** consume our mask.

```
[Vision sensor]
       │
       ├──────────────► adapters → PERCEPTION PORT → Nav2 SemanticLayer   (Dev 1 → Dev 3)
       ├──────────────► RTAB-Map  → pose / map                            (Dev 2, ignores mask)
       └──────────────► Depth Anything → VoxelLayer                       (Dev 1 geometry, optional)
```

If Dev 1 is down, Dev 2 can still track (and then Dev 5 will hold because perception is degraded). If Dev 2 is down, Dev 1 still publishes the port (and Dev 5 holds on invalid pose). **No circular dependency.**

## What we publish (contract)

Architecture §8 is stricter than `dev.md`. We implement the architecture.

| Output | Type | Who consumes | Rule |
|---|---|---|---|
| `/segmentation/mask` | `sensor_msgs/Image` `mono8` | Dev 3, Dev 5 watchdog | pixels **strictly** `{0,1,2}` |
| `/segmentation/confidence` | `sensor_msgs/Image` `32FC1` | Dev 3 (optional), eval | port-normalized `[0,1]`, same H×W and header as mask |
| `/segmentation/port_meta` | custom (T01) | Dev 3, Dev 5 | `valid`, informational `age`, `scale`, same stamp/frame as mask |
| `/ugv/perception_degraded` | `std_msgs/Bool` | **Dev 5** (level-3 hold) | `true` on stale, invalid, gate storm, adapter fail |
| depth / cloud (T08) | `Image` and/or `PointCloud2` | Dev 3 VoxelLayer | **not** remapped to {0,1,2} |

`/cmd_vel` and `/cmd_vel_nav2` are **out of bounds**. We never publish them.

`header.stamp` on the mask **is the source image time**, not “now”. `header.frame_id` is the **optical / camera frame**. Resolution equals the source image unless a scale is stated in config (architecture §8.5).

## Canonical classes (v1)

| ID | Name | Cost intent (Dev 3 binds to this) |
|---|---|---|
| 0 | unknown | never free — inflate |
| 1 | traversable | free / low |
| 2 | hazard | lethal / inscribed |

No fourth class. Soft / ambiguous → `0`. That is a product decision, not a model decision.

## Adapter vs port

An **adapter** speaks model language (YOLOE prompts, ONNX class names, logits).

The **port** speaks only `{0,1,2}` + normalized confidence + freshness.

Mandatory chain, every frame (this **is** T07; T07 contains no model branch):

```
real image → adapter raw labels/scores          # T06 via T12 backend
          → remap YAML (or refuse to publish)   # T03
          → normalize scores to [0,1]           # T04
          → apply τ_trav, τ_haz, τ_min, κ
          → stamp/frame/age/valid               # copied from sensor / T05
          → publish port  OR  degraded=true and do not present as current
```

T07 must leave `header.stamp` and optical `frame_id` identical to the source frame.

## Fail-safe split (easy to get wrong)

Architecture §8.6: gate fail or stale/invalid → `/ugv/perception_degraded` **+ front ROI lethal/max-inflate** + safety hold.

| Piece | Owner |
|---|---|
| Detect fail / stale / invalid | **Dev 1** |
| Publish `/ugv/perception_degraded` | **Dev 1** |
| Stop treating the last mask as live | **Dev 1** |
| Front ROI lethal / max-inflate on the costmap | **Dev 3** (reacts to degraded) |
| Zero `/cmd_vel` | **Dev 5** |

We do not implement costmap inflation. We make the flag honest.

## Optional depth (architecture §6, §9)

Depth Anything is **not** a second perception port. It exists so VoxelLayer can mark occupied cells. Conflict rule is Dev 3’s: geometry lethal wins; our class-1 pixels must never clear it. Dev 1’s job is to give Dev 3 a real depth/cloud with the same stamp/frame discipline as the mask.

## Runtime profiles (what we actually run)

| Profile | Dev 1 source | Product? |
|---|---|---|
| `live_cam` | calibrated outdoor camera | **yes — default** |
| `bag` | recorded **outdoor** bag from that camera | eval |
| `rugd` | RUGD images (real outdoor dataset) | eval only, if we opt in |
| `sim` | Gazebo camera | Dev 5 integration; **not** how we qualify perception |

## Kill list that applies to us

- YOLOE as brain  
- Adapter publish without remap  
- Shared τ on raw multi-model confidence  
- Unknown as free  
- Stale mask as current  
- Tutorial ONNX as the only ontology  
- Mono USB marketed as outdoor-meter-ready (sensor honesty lives in Dev 2, but we still do not fake calibration)

## What we ignore from `dev.md`

See [CONFLICTS.md](CONFLICTS.md). Short version: hours, difficulty ranks, and any task wording that makes ONNX or Depth Anything part of the 3-class port are discarded.
