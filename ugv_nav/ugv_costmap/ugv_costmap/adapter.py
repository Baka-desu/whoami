"""ROS-free helpers between ROS messages and costmap_core. Every rule here is unit-tested without ROS.

The adapter adds nothing to the costmap rules. It only:
- pools the mask (the core projects pixel by pixel in Python; 640x480 is too slow for a live loop),
  keeping the core's own per-cell precedence HAZARD > UNKNOWN > TRAVERSABLE inside each block,
- scales the intrinsics to the pooled mask,
- turns the TF camera pose into the core's (height, pitch) ground geometry, refusing poses the core
  cannot represent (roll / yaw),
- turns core costs into nav_msgs/OccupancyGrid values for Nav2's StaticLayer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from costmap_core.class_to_cost import DEFAULT_COST_VALUES, SemanticClass
from costmap_core.mask_projection import SEMANTIC_CLASS_PRECEDENCE
from costmap_core.projection import CameraGroundGeometry, CameraIntrinsics

# OccupancyGrid values (nav_msgs): -1 unknown, 0 free, 100 lethal.
OCC_UNKNOWN = -1
OCC_FREE = 0
OCC_LETHAL = 100

# class id -> precedence rank, and back, from the core's own table.
_RANK_OF_CLASS = np.zeros(256, dtype=np.int8)
for _cls, _rank in SEMANTIC_CLASS_PRECEDENCE.items():
    _RANK_OF_CLASS[int(_cls)] = _rank
_CLASS_OF_RANK = np.zeros(max(SEMANTIC_CLASS_PRECEDENCE.values()) + 1, dtype=np.uint8)
for _cls, _rank in SEMANTIC_CLASS_PRECEDENCE.items():
    _CLASS_OF_RANK[_rank] = int(_cls)


class AdapterError(ValueError):
    """An input the adapter refuses (never coerced into a costmap)."""


def pool_mask(classes: np.ndarray, factor: int) -> np.ndarray:
    """Block-pool a class mask by `factor`; each block takes its highest-precedence class.

    So one hazard pixel in a block makes the block hazard, and unknown beats traversable, exactly as the
    core resolves several pixels landing in one cell.
    """
    if factor < 1:
        raise AdapterError(f"factor must be >= 1, got {factor}")
    h, w = classes.shape
    if h % factor or w % factor:
        raise AdapterError(f"mask {w}x{h} is not divisible by downsample factor {factor}")
    if factor == 1:
        return classes
    ranks = _RANK_OF_CLASS[classes].reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))
    return _CLASS_OF_RANK[ranks]


def mask_is_fresh(now_ns: int, stamp_ns: int, max_age_s: float) -> bool:
    """A mask is current only if its own stamp is 0..max_age_s old (architecture §8.4: the consumer rejects by age).

    Validity is bound to the sample: Dev 1 publishes a mask only when it is valid, so the mask stamp is all a
    consumer needs. A future stamp is refused like a stale one, the same rule as Dev 1's freshness check.
    """
    age_ns = now_ns - stamp_ns
    return 0 <= age_ns <= max_age_s * 1e9


def scale_intrinsics(fx: float, fy: float, cx: float, cy: float, factor: int) -> CameraIntrinsics:
    """Intrinsics of the pooled image: pooled pixel (u', v') is the centre of block k*u' .. k*u'+k-1."""
    half = (factor - 1) / 2.0
    return CameraIntrinsics(fx=fx / factor, fy=fy / factor, cx=(cx - half) / factor, cy=(cy - half) / factor)


@dataclass(frozen=True)
class CameraMount:
    """Camera optical frame pose in the robot frame, reduced to what the core can represent."""

    x: float  # forward offset of the camera above the ground (m)
    y: float  # left offset (m)
    geometry: CameraGroundGeometry  # height above ground, downward pitch


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not n > 0:
        raise AdapterError("zero quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def mount_from_tf(translation: tuple[float, float, float], rotation_xyzw: tuple[float, float, float, float],
                  max_roll_yaw_rad: float = math.radians(2.0)) -> CameraMount:
    """base_link <- camera optical frame transform -> CameraMount.

    The robot frame is x-forward, y-left, z-up with z=0 on the ground (base_link on the ground plane).
    The optical frame is x-right, y-down, z-forward. The core models only height + pitch, so a camera with
    roll or yaw beyond `max_roll_yaw_rad` is refused rather than projected wrongly.
    """
    r = _quat_to_matrix(*rotation_xyzw)
    forward = r[:, 2]  # optical z in the robot frame
    right = r[:, 0]    # optical x in the robot frame
    pitch = math.atan2(-forward[2], math.hypot(forward[0], forward[1]))  # positive = looking down
    yaw = math.atan2(forward[1], forward[0])
    # Roll: with zero roll the optical x axis is horizontal.
    roll = math.asin(max(-1.0, min(1.0, right[2])))
    if abs(yaw) > max_roll_yaw_rad or abs(roll) > max_roll_yaw_rad:
        raise AdapterError(
            f"camera yaw {math.degrees(yaw):.1f} deg / roll {math.degrees(roll):.1f} deg: the costmap core "
            "supports a forward-facing camera with pitch only"
        )
    x, y, z = translation
    return CameraMount(x=x, y=y, geometry=CameraGroundGeometry(camera_height=z, pitch_rad=pitch))


def costs_to_occupancy(final: np.ndarray, coverage: np.ndarray, unknown_occupancy: int) -> np.ndarray:
    """Core costs -> OccupancyGrid data (row-major int8, same row/col convention as the core grid).

    traversable -> 0, hazard -> 100, observed unknown -> `unknown_occupancy` (never free, architecture §8.1),
    cells the camera cannot see -> -1. The core uses one value for "observed unknown" and "never seen",
    so `coverage` (cells the camera can see at all) separates them.
    """
    if not 0 < unknown_occupancy < OCC_LETHAL:
        raise AdapterError("unknown_occupancy must be in 1..99: not free, not lethal")
    cv = DEFAULT_COST_VALUES
    out = np.full(final.shape, OCC_UNKNOWN, dtype=np.int8)
    out[final == cv.traversable_cost] = OCC_FREE
    out[final == cv.unknown_cost] = unknown_occupancy
    out[final == cv.hazard_cost] = OCC_LETHAL
    out[~coverage] = OCC_UNKNOWN
    return out


def front_roi_lethal(coverage: np.ndarray) -> np.ndarray:
    """Fail-safe grid (architecture §8.6): everything the camera covers is lethal, the rest unknown."""
    out = np.full(coverage.shape, OCC_UNKNOWN, dtype=np.int8)
    out[coverage] = OCC_LETHAL
    return out


def all_traversable(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, int(SemanticClass.TRAVERSABLE), dtype=np.uint8)
