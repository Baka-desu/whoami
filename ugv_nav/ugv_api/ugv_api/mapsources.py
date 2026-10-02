"""ROS message -> map store source, as pure functions on plain values. No rclpy, numpy and stdlib only.

ros_node.py copies the few fields it needs out of a message and calls one of these; the result is exactly the
`source` that app.py's encode callback for that layer consumes (see `_layer_encoders` there):

  cloud       cloud_source          {"fields", "point_step", "n_points", "is_bigendian", "data"}
  trajectory  trajectory_source     (N, 7) float32 x y z qx qy qz qw
  elevation   ElevationPairer       {"x", "y", "z", "confidence", "obstacle_h", "origin_xy", "resolution",
                                     "width", "height"}
  grid        grid_source           {"cells", "resolution", "origin_xy", "origin_yaw"}
  depth       depth_source          {"depth_m": (H, W) float32}
  live        backproject + transform_points -> {"xyz": (N, 3) float32}   (built by the caller)
  camera      jpeg_source           bytes

A source is immutable once it is in the store. What a message hands over is therefore either copied or a
read-only view of a buffer nobody writes again (rclpy builds a fresh message, and a fresh `data` buffer, for
every sample it delivers, so a view over `msg.data` is never overwritten by the next message).

Malformed input raises ValueError; the caller counts and logs it once and keeps the previous good source.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

import numpy as np

from ugv_api.mapcodec import POINTFIELD_FLOAT32

__all__ = [
    "ElevationPairer",
    "MapConfig",
    "backproject",
    "cloud_columns",
    "cloud_source",
    "depth_source",
    "grid_source",
    "intrinsics",
    "jpeg_source",
    "parse_stats",
    "quaternion_yaw",
    "stamp_seconds",
    "trajectory_source",
    "transform_points",
]


# ------------------------------------------------------------------------------------------------ config


def _int_at_least(name: str, value: int, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _finite_positive(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


def _topic(name: str, value: str) -> None:
    if not (isinstance(value, str) and value.startswith("/") and len(value) > 1 and " " not in value):
        raise ValueError(f"{name} must be an absolute topic name, got {value!r}")


@dataclass(frozen=True)
class MapConfig:
    """The `map:` block of config/api.yaml. The defaults here and in that file are the same values (a test
    compares them), so a gateway started without the file behaves like one started with it."""

    # --- what the HTTP layer encodes with (create_app tunables)
    cloud_point_budget: int = 500_000
    cloud_spacing_m: float = 0.05
    elevation_max_side: int = 512
    depth_stride: int = 2
    depth_max_range_m: float = 8.0
    # --- the live scan built from the depth image
    live_stride: int = 4
    live_range_min_m: float = 0.3
    live_range_max_m: float = 8.0
    # --- demand: heavy subscriptions live this long after the last GET /api/v1/map; stats go quiet after
    # stats_stale_s without a message
    idle_timeout_s: float = 10.0
    stats_stale_s: float = 5.0
    # --- inputs
    cloud_topic: str = "/rtabmap/cloud_map"
    trajectory_topic: str = "/rtabmap/mapPath"
    elevation_cloud_topic: str = "/ugv/elevation/cloud"
    elevation_obstacles_topic: str = "/ugv/elevation/obstacles"
    grid_topic: str = "/global_costmap/costmap"
    depth_topic: str = "/perception/depth/image"
    camera_topic: str = "/image_raw/compressed"
    map_stats_topic: str = "/ugv/map/stats"
    perception_stats_topic: str = "/ugv/perception/stats"

    def __post_init__(self) -> None:
        if not isinstance(self.cloud_point_budget, int) or self.cloud_point_budget < 0:
            raise ValueError(f"cloud_point_budget must be an integer >= 0, got {self.cloud_point_budget!r}")
        _finite_positive("cloud_spacing_m", self.cloud_spacing_m)
        _int_at_least("elevation_max_side", self.elevation_max_side, 1)
        _int_at_least("depth_stride", self.depth_stride, 1)
        _finite_positive("depth_max_range_m", self.depth_max_range_m)
        _int_at_least("live_stride", self.live_stride, 1)
        _finite_positive("live_range_max_m", self.live_range_max_m)
        if not (isinstance(self.live_range_min_m, (int, float)) and math.isfinite(self.live_range_min_m)
                and 0 <= self.live_range_min_m < self.live_range_max_m):
            raise ValueError(f"live_range_min_m ({self.live_range_min_m!r}) must be finite, >= 0 and below "
                             f"live_range_max_m ({self.live_range_max_m!r})")
        _finite_positive("idle_timeout_s", self.idle_timeout_s)
        _finite_positive("stats_stale_s", self.stats_stale_s)
        for f in dataclass_fields(self):
            if f.name.endswith("_topic"):
                _topic(f.name, getattr(self, f.name))

    def app_kwargs(self) -> dict[str, Any]:
        """The create_app keyword arguments this block sets."""
        return {
            "cloud_point_budget": self.cloud_point_budget,
            "cloud_spacing_m": self.cloud_spacing_m,
            "elevation_max_side": self.elevation_max_side,
            "depth_stride": self.depth_stride,
            "depth_max_range_m": self.depth_max_range_m,
        }


# ----------------------------------------------------------------------------------------------- small


def stamp_seconds(sec: int, nanosec: int, *, fallback_s: float) -> float:
    """A header stamp in seconds; an unset (zero) stamp is replaced by `fallback_s` (receipt time)."""
    value = int(sec) + int(nanosec) * 1e-9
    return value if value > 0 else float(fallback_s)


def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Rotation about z, in radians, of an (x, y, z, w) quaternion (roll and pitch are ignored)."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _bytes_view(data: Any) -> np.ndarray:
    """A read-only uint8 view of a bytes-like object (bytes, bytearray, array('B'), uint8 ndarray), no copy."""
    return _readonly(np.frombuffer(data, dtype=np.uint8))


# ------------------------------------------------------------------------------------------- point clouds


def cloud_source(*, fields: Sequence[Sequence[Any]], point_step: int, n_points: int, is_bigendian: bool,
                 data: Any) -> dict[str, Any]:
    """`cloud` source: the PointCloud2 layout plus a read-only view of its payload. The payload is not
    parsed here: the HTTP thread's `cloud_view` does that once per request, over the same memory."""
    return {
        "fields": [(str(name), int(offset), int(datatype), int(count)) for name, offset, datatype, count in fields],
        "point_step": int(point_step),
        "n_points": int(n_points),
        "is_bigendian": bool(is_bigendian),
        "data": _bytes_view(data),
    }


def cloud_columns(*, fields: Sequence[Sequence[Any]], point_step: int, n_points: int, is_bigendian: bool,
                  data: Any, names: Sequence[str]) -> dict[str, np.ndarray]:
    """The named FLOAT32 fields of a PointCloud2 as independent contiguous float32 arrays, one entry per point.
    Copies, so the message buffer is free to go."""
    if is_bigendian:
        raise ValueError("big-endian point clouds are not supported")
    point_step, n_points = int(point_step), int(n_points)
    if point_step <= 0 or n_points < 0:
        raise ValueError("point_step must be > 0 and n_points >= 0")
    by_name = {str(name): (int(offset), int(datatype)) for name, offset, datatype, _count in fields}
    base = np.frombuffer(data, dtype=np.uint8)
    if base.size < point_step * n_points:
        raise ValueError(f"data holds {base.size} bytes, fewer than point_step * n_points = {point_step * n_points}")
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name not in by_name:
            raise ValueError(f"point cloud has no '{name}' field")
        offset, datatype = by_name[name]
        if datatype != POINTFIELD_FLOAT32:
            raise ValueError(f"'{name}' field must be FLOAT32 (7), got datatype {datatype}")
        if offset < 0 or offset + 4 > point_step:
            raise ValueError(f"field '{name}' at offset {offset} does not fit in point_step {point_step}")
        if n_points == 0:
            out[name] = np.zeros(0, dtype=np.float32)
        else:
            column = np.ndarray((n_points,), dtype="<f4", buffer=base, offset=offset, strides=(point_step,))
            out[name] = np.array(column, dtype=np.float32)  # contiguous copy
    return out


# ---------------------------------------------------------------------------------------- path and grid


def trajectory_source(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """`trajectory` source: poses as an (N, 7) float32 array of x y z qx qy qz qw (empty path: 0 rows)."""
    if len(rows) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    out = np.asarray(rows, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != 7:
        raise ValueError(f"a pose row is x y z qx qy qz qw (7 values), got shape {out.shape}")
    return np.ascontiguousarray(out)


def _int8_cells(data: Any, width: int, height: int) -> np.ndarray:
    if width < 0 or height < 0:
        raise ValueError(f"grid size {width} x {height} is negative")
    if isinstance(data, np.ndarray):
        cells = data.astype(np.int8, copy=False).reshape(-1)
    elif isinstance(data, (bytes, bytearray, memoryview)) or hasattr(data, "typecode"):
        cells = np.frombuffer(data, dtype=np.int8)  # array('b') and friends: no copy
    else:
        cells = np.asarray(data, dtype=np.int8)
    if cells.size != width * height:
        raise ValueError(f"grid data holds {cells.size} cells, expected width * height = {width * height}")
    return _readonly(cells.reshape(height, width))


def grid_source(*, data: Any, width: int, height: int, resolution: float, origin_x: float, origin_y: float,
                origin_q: Sequence[float]) -> dict[str, Any]:
    """`grid` source from an OccupancyGrid: int8 cells (row = y), resolution, origin position and origin yaw
    (from the origin quaternion x y z w)."""
    return {
        "cells": _int8_cells(data, int(width), int(height)),
        "resolution": float(resolution),
        "origin_xy": (float(origin_x), float(origin_y)),
        "origin_yaw": quaternion_yaw(*(float(c) for c in origin_q)),
    }


class ElevationPairer:
    """Pairs the elevation cloud with its obstacle grid by header stamp.

    The two topics are published together and carry one `header.stamp`; the cloud holds the cells and the grid
    holds their geometry (resolution, size, origin). A source is built only from a cloud and a grid with equal
    stamps: the latest of each is kept, and the pair is emitted when the second half of it arrives. Not
    thread-safe: both callbacks run in one mutually exclusive callback group.
    """

    def __init__(self) -> None:
        self._cloud: tuple[int, dict[str, np.ndarray]] | None = None
        self._grid: tuple[int, dict[str, Any]] | None = None

    def reset(self) -> None:
        self._cloud = None
        self._grid = None

    def add_cloud(self, stamp_ns: int, columns: dict[str, np.ndarray]) -> dict[str, Any] | None:
        self._cloud = (int(stamp_ns), columns)
        return self._pair()

    def add_grid(self, stamp_ns: int, *, resolution: float, width: int, height: int, origin_x: float,
                 origin_y: float) -> dict[str, Any] | None:
        self._grid = (int(stamp_ns), {
            "origin_xy": (float(origin_x), float(origin_y)),
            "resolution": float(resolution),
            "width": int(width),
            "height": int(height),
        })
        return self._pair()

    def _pair(self) -> dict[str, Any] | None:
        if self._cloud is None or self._grid is None or self._cloud[0] != self._grid[0]:
            return None
        cols, geometry = self._cloud[1], self._grid[1]
        return {
            "x": cols["x"], "y": cols["y"], "z": cols["z"],
            "confidence": cols["confidence"], "obstacle_h": cols["obstacle_h"],
            **geometry,
        }


# ------------------------------------------------------------------------------- depth image, live scan


def depth_source(*, encoding: str, height: int, width: int, step: int, is_bigendian: bool,
                 data: Any) -> dict[str, np.ndarray]:
    """`depth` source from a 32FC1 image in metres (NaN = hole): a native-order (H, W) float32 copy."""
    if encoding != "32FC1":
        raise ValueError(f"depth image encoding must be 32FC1 (metres), got {encoding!r}")
    height, width, step = int(height), int(width), int(step)
    if height <= 0 or width <= 0:
        raise ValueError(f"depth image size {width} x {height} is empty")
    if step < width * 4:
        raise ValueError(f"row step {step} is shorter than {width} float32 values")
    base = np.frombuffer(data, dtype=np.uint8)
    if base.size < step * (height - 1) + width * 4:
        raise ValueError(f"depth data holds {base.size} bytes, too few for {width} x {height} with step {step}")
    view = np.ndarray((height, width), dtype=">f4" if is_bigendian else "<f4", buffer=base, strides=(step, 4))
    return {"depth_m": _readonly(np.array(view, dtype=np.float32))}


def intrinsics(k: Sequence[float], *, width: int, height: int, info_width: int = 0,
               info_height: int = 0) -> tuple[float, float, float, float] | None:
    """(fx, fy, cx, cy) from a row-major 3x3 K, or None for a missing or fake K (zero, negative or non-finite
    focal length or centre). When CameraInfo says which size it was calibrated for and the image is another,
    K is scaled to the image."""
    if len(k) != 9:
        return None
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        return None
    if info_width > 0 and info_height > 0 and (info_width, info_height) != (width, height):
        sx, sy = width / info_width, height / info_height
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
    return fx, fy, cx, cy


def backproject(depth_m: np.ndarray, intr: tuple[float, float, float, float], *, stride: int,
                min_range_m: float, max_range_m: float) -> np.ndarray:
    """Every `stride`-th pixel of a depth image (metres along the optical axis) as an (N, 3) float32 array of
    points in the camera optical frame (x right, y down, z forward), row-major. Pixels that are not finite
    or whose depth is outside [min_range_m, max_range_m] are dropped."""
    fx, fy, cx, cy = intr
    z = depth_m[::stride, ::stride]
    rows = np.arange(0, depth_m.shape[0], stride, dtype=np.float64)[:, None]
    cols = np.arange(0, depth_m.shape[1], stride, dtype=np.float64)[None, :]
    keep = np.isfinite(z) & (z >= min_range_m) & (z <= max_range_m)
    zk = z[keep].astype(np.float64)
    u = np.broadcast_to(cols, z.shape)[keep]
    v = np.broadcast_to(rows, z.shape)[keep]
    return np.stack([(u - cx) * zk / fx, (v - cy) * zk / fy, zk], axis=1).astype(np.float32)


def transform_points(xyz: np.ndarray, translation: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    """`R @ p + t` for every row of an (N, 3) array; (qx, qy, qz, qw) is normalised first. float32 out."""
    qx, qy, qz, qw = (float(c) for c in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not norm > 0 or not math.isfinite(norm):
        raise ValueError("a transform needs a non-zero, finite quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rot = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    t = np.array([float(c) for c in translation])
    return (xyz.astype(np.float64) @ rot.T + t).astype(np.float32)


# ----------------------------------------------------------------------------------------- camera, stats


def jpeg_source(data: Any) -> bytes:
    """`camera` source: the JPEG as immutable bytes. Anything that does not start like a JPEG is refused,
    because the route serves it as image/jpeg."""
    frame = bytes(data)
    if frame[:3] != b"\xff\xd8\xff":
        raise ValueError("camera frame is not a JPEG (no FF D8 FF start)")
    return frame


def parse_stats(text: str) -> dict[str, Any] | None:
    """A stats message (a JSON object in a std_msgs/String) as a dict; None for anything else."""
    try:
        value = json.loads(text)
    except (ValueError, RecursionError, TypeError):
        return None
    return value if isinstance(value, dict) else None
