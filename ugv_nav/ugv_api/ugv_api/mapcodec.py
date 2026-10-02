"""Binary encoders (and test-only decoders) for the 3D map layers the gateway sends to the web viewer.

Pure numpy + struct, no ROS imports. The byte layouts are the "Binary format v1" contract shared with the
viewer's TypeScript decoder (same spec, same golden files in test/fixtures/map/). Everything is
little-endian:

    prelude, 24 bytes, every layer:  char[4] magic | u16 format = 1 | u16 header_bytes | u32 epoch |
                                     u32 seq | f64 stamp_s
    UGVC cloud       header 64  u32 count, u32 source_count, f32 spacing_m, u32 flags (bit0 = rgb),
                                f32[3] bbox_min, f32[3] bbox_max; body f32 xyz[3n] then u8 rgb[3n]
    UGVE elevation   header 48  u32 width, u32 height, f32 resolution_m, f32 origin_x, f32 origin_y,
                                u32 known_cells; body f32 height[w*h], u8 obstacle[w*h], u8 confidence[w*h]
    UGVT trajectory  header 32  u32 count, f32 length_m; body f32 x,y,z,qx,qy,qz,qw per pose
    UGVG cost grid   header 48  u32 width, u32 height, f32 resolution_m, f32 origin_x, f32 origin_y,
                                f32 origin_yaw; body i8 cell[w*h]
    UGVD depth       header 40  u32 width, u32 height, f32 unit_m, f32 max_range_m; body u16 count[w*h]

2-D layers are row-major with row = y index and column = x index; the origin is the world position of the
corner of cell (0, 0). `epoch` (random per gateway process) and `seq` let the viewer tell a restarted
gateway from a stale frame, so both are always written, modulo 2**32.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy as np

FORMAT = 1

MAGIC_CLOUD = b"UGVC"
MAGIC_ELEVATION = b"UGVE"
MAGIC_TRAJECTORY = b"UGVT"
MAGIC_GRID = b"UGVG"
MAGIC_DEPTH = b"UGVD"

PRELUDE_BYTES = 24
CLOUD_HEADER_BYTES = 64
ELEVATION_HEADER_BYTES = 48
TRAJECTORY_HEADER_BYTES = 32
GRID_HEADER_BYTES = 48
DEPTH_HEADER_BYTES = 40

FLAG_RGB = 1  # UGVC flags bit0

OBSTACLE_UNIT_M = 0.05  # one obstacle byte = 5 cm
DEPTH_UNIT_M = 0.001  # one depth count = 1 mm
_UNIT_TOLERANCE = 1e-3  # in obstacle units (0.05 mm): float noise must not push 0.15 m up to 4 units

# sensor_msgs/msg/PointField datatype codes
POINTFIELD_UINT32 = 6
POINTFIELD_FLOAT32 = 7

_PRELUDE = struct.Struct("<4sHHIId")
_CLOUD = struct.Struct("<IIfI3f3f")
_ELEVATION = struct.Struct("<IIfffI")
_TRAJECTORY = struct.Struct("<If")
_GRID = struct.Struct("<IIffff")
_DEPTH = struct.Struct("<IIff")

assert _PRELUDE.size == PRELUDE_BYTES
assert PRELUDE_BYTES + _CLOUD.size == CLOUD_HEADER_BYTES
assert PRELUDE_BYTES + _ELEVATION.size == ELEVATION_HEADER_BYTES
assert PRELUDE_BYTES + _TRAJECTORY.size == TRAJECTORY_HEADER_BYTES
assert PRELUDE_BYTES + _GRID.size == GRID_HEADER_BYTES
assert PRELUDE_BYTES + _DEPTH.size == DEPTH_HEADER_BYTES

__all__ = [
    "cloud_view",
    "encode_cloud",
    "grid_from_cells",
    "encode_elevation",
    "encode_trajectory",
    "encode_grid",
    "encode_depth",
    "decode_cloud",
    "decode_elevation",
    "decode_trajectory",
    "decode_grid",
    "decode_depth",
]


# ------------------------------------------------------------------------------------------ shared


def _prelude(magic: bytes, header_bytes: int, epoch: int, seq: int, stamp_s: float) -> bytes:
    return _PRELUDE.pack(magic, FORMAT, header_bytes, int(epoch) & 0xFFFFFFFF, int(seq) & 0xFFFFFFFF, float(stamp_s))


def _positive_finite(value: float, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _origin(origin_xy: Sequence[float]) -> tuple[float, float]:
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    if not (math.isfinite(ox) and math.isfinite(oy)):
        raise ValueError("origin_xy must be finite")
    return ox, oy


def _plane(a: np.ndarray | Sequence, dtype, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (rows, columns) array, got shape {arr.shape}")
    return arr


def _decode_prelude(b: bytes, magic: bytes, header_bytes: int, layer: str) -> dict:
    """Validate magic / format / header_bytes and return the prelude fields. Length is checked by the caller."""
    if len(b) < header_bytes:
        raise ValueError(f"{layer}: {len(b)} bytes is shorter than the {header_bytes}-byte header")
    got_magic, fmt, got_header, epoch, seq, stamp_s = _PRELUDE.unpack_from(b, 0)
    if got_magic != magic:
        raise ValueError(f"{layer}: bad magic {got_magic!r}, expected {magic!r}")
    if fmt != FORMAT:
        raise ValueError(f"{layer}: unsupported format {fmt}, expected {FORMAT}")
    if got_header != header_bytes:
        raise ValueError(f"{layer}: header_bytes {got_header}, expected {header_bytes}")
    return {"epoch": epoch, "seq": seq, "stamp_s": stamp_s}


def _require_length(b: bytes, expected: int, layer: str) -> None:
    if len(b) != expected:
        raise ValueError(f"{layer}: {len(b)} bytes, expected exactly {expected}")


# ------------------------------------------------------------------------------------------- cloud


def cloud_view(
    *,
    fields: Sequence[tuple[str, int, int, int]],
    point_step: int,
    n_points: int,
    is_bigendian: bool,
    data: bytes,
) -> tuple[np.ndarray, np.ndarray | None]:
    """(xyz (N, 3) float32, rgb (N, 3) uint8 or None) over a PointCloud2 payload, without copying `data`.

    `fields` are (name, offset, datatype, count) tuples with sensor_msgs/PointField datatype codes. The
    returned arrays are views of `data` whenever the layout allows (x, y, z adjacent in that order; rgb
    always); otherwise xyz is gathered into a new array. rgb comes from a packed `rgb` (preferred) or
    `rgba` field: a 32-bit word holding 0x??RRGGBB (float32 or uint32), the alpha byte is ignored.
    Raises ValueError for a big-endian cloud, a missing or non-FLOAT32 x/y/z, or a buffer that is too short.
    """
    if is_bigendian:
        raise ValueError("big-endian point clouds are not supported")
    point_step, n_points = int(point_step), int(n_points)
    if point_step <= 0 or n_points < 0:
        raise ValueError("point_step must be > 0 and n_points >= 0")
    by_name = {str(name): (int(offset), int(datatype), int(count)) for name, offset, datatype, count in fields}
    offsets = []
    for axis in "xyz":
        if axis not in by_name:
            raise ValueError(f"point cloud has no '{axis}' field")
        offset, datatype, _ = by_name[axis]
        if datatype != POINTFIELD_FLOAT32:
            raise ValueError(f"'{axis}' field must be FLOAT32 (7), got datatype {datatype}")
        offsets.append(offset)
    rgb_offset = None
    for name in ("rgb", "rgba"):
        if name in by_name and by_name[name][1] in (POINTFIELD_UINT32, POINTFIELD_FLOAT32):
            rgb_offset = by_name[name][0]
            break
    for offset in (*offsets, *(() if rgb_offset is None else (rgb_offset,))):
        if offset < 0 or offset + 4 > point_step:
            raise ValueError(f"field at offset {offset} does not fit in point_step {point_step}")

    base = np.frombuffer(data, dtype=np.uint8)
    if base.size < point_step * n_points:
        raise ValueError(f"data holds {base.size} bytes, fewer than point_step * n_points = {point_step * n_points}")

    ox, oy, oz = offsets
    if n_points == 0:
        xyz = np.zeros((0, 3), dtype=np.float32)
    elif oy == ox + 4 and oz == ox + 8:
        xyz = np.ndarray((n_points, 3), dtype="<f4", buffer=base, offset=ox, strides=(point_step, 4))
    else:
        xyz = np.empty((n_points, 3), dtype=np.float32)
        for col, offset in enumerate(offsets):
            xyz[:, col] = np.ndarray((n_points,), dtype="<f4", buffer=base, offset=offset, strides=(point_step,))

    if rgb_offset is None:
        return xyz, None
    if n_points == 0:
        return xyz, np.zeros((0, 3), dtype=np.uint8)
    # little-endian word 0xAARRGGBB is stored B, G, R, A: columns 2, 1, 0 are R, G, B
    bgra = np.ndarray((n_points, 4), dtype=np.uint8, buffer=base, offset=rgb_offset, strides=(point_step, 1))
    return xyz, bgra[:, 2::-1]


_HASH_MUL = (np.uint64(0x9E3779B185EBCA87), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0x165667B19E3779F9))
_SPLITMIX = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB))


def _voxel_hash(xyz: np.ndarray, spacing_m: float) -> np.ndarray:
    """uint64 hash of each point's voxel index at `spacing_m`: equal for points in the same voxel."""
    idx = np.floor(xyz.astype(np.float64) / spacing_m)
    idx = np.clip(idx, -(2.0**62), 2.0**62).astype(np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        h = idx[:, 0] * _HASH_MUL[0] + idx[:, 1] * _HASH_MUL[1] + idx[:, 2] * _HASH_MUL[2]
        h = h + _SPLITMIX[0]
        h = (h ^ (h >> np.uint64(30))) * _SPLITMIX[1]
        h = (h ^ (h >> np.uint64(27))) * _SPLITMIX[2]
        h = h ^ (h >> np.uint64(31))
    return h


def _select_by_voxel_hash(xyz: np.ndarray, rgb: np.ndarray | None, spacing_m: float, budget: int) -> np.ndarray:
    """Boolean mask of the `budget` points kept: those in the voxels with the lowest hash values. When the
    last voxel only partly fits, its points are ranked by their own values, so the chosen set depends only
    on the point set, never on the order the points arrived in."""
    h = _voxel_hash(xyz, spacing_m)
    last = np.partition(h, budget - 1)[budget - 1]  # hash of the last voxel that is kept, maybe in part
    keep = h < last
    edge = np.flatnonzero(h == last)
    room = budget - int(keep.sum())
    if room < len(edge):
        keys = [xyz[edge, 2], xyz[edge, 1], xyz[edge, 0]]
        if rgb is not None:
            keys = [rgb[edge, 2], rgb[edge, 1], rgb[edge, 0], *keys]
        edge = edge[np.lexsort(keys)[:room]]
    keep[edge] = True
    return keep


def encode_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    *,
    epoch: int,
    seq: int,
    stamp_s: float,
    budget: int,
    spacing_m: float,
) -> bytes:
    """UGVC. Drops non-finite points; if more than `budget` finite points remain, keeps `budget` of them by
    voxel hash (voxels of `spacing_m`: the lowest-hash voxels first), so the same point set always yields
    the same selection whatever its order. Kept points stay in input order. `source_count` is the finite
    count before that cut; bbox is over the points written."""
    pts = np.asarray(xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {pts.shape}")
    col = None
    if rgb is not None:
        col = np.asarray(rgb)
        if col.dtype != np.uint8 or col.shape != pts.shape:
            raise ValueError(f"rgb must be uint8 with the shape of xyz {pts.shape}, got {col.dtype} {col.shape}")
    if int(budget) < 0:
        raise ValueError("budget must be >= 0")
    spacing = _positive_finite(spacing_m, "spacing_m")

    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        pts = pts[finite]
        col = None if col is None else col[finite]
    source_count = len(pts)
    if source_count > budget:
        if budget == 0:
            keep = np.zeros(source_count, dtype=bool)
        else:
            keep = _select_by_voxel_hash(pts, col, spacing, int(budget))
        pts = pts[keep]
        col = None if col is None else col[keep]
    count = len(pts)
    bbox_min = pts.min(axis=0) if count else np.zeros(3, np.float32)
    bbox_max = pts.max(axis=0) if count else np.zeros(3, np.float32)

    out = [
        _prelude(MAGIC_CLOUD, CLOUD_HEADER_BYTES, epoch, seq, stamp_s),
        _CLOUD.pack(count, source_count, spacing, FLAG_RGB if col is not None else 0, *bbox_min, *bbox_max),
        np.ascontiguousarray(pts, dtype="<f4").tobytes(),
    ]
    if col is not None:
        out.append(np.ascontiguousarray(col, dtype=np.uint8).tobytes())
    return b"".join(out)


def decode_cloud(b: bytes) -> dict:
    d = _decode_prelude(b, MAGIC_CLOUD, CLOUD_HEADER_BYTES, "cloud")
    count, source_count, spacing_m, flags, *bbox = _CLOUD.unpack_from(b, PRELUDE_BYTES)
    has_rgb = bool(flags & FLAG_RGB)
    _require_length(b, CLOUD_HEADER_BYTES + (15 if has_rgb else 12) * count, "cloud")
    xyz_end = CLOUD_HEADER_BYTES + 12 * count
    d.update(
        count=count,
        source_count=source_count,
        spacing_m=spacing_m,
        flags=flags,
        has_rgb=has_rgb,
        bbox_min=np.array(bbox[:3], dtype=np.float32),
        bbox_max=np.array(bbox[3:], dtype=np.float32),
        xyz=np.frombuffer(b, dtype="<f4", count=3 * count, offset=CLOUD_HEADER_BYTES).reshape(count, 3),
        rgb=np.frombuffer(b, dtype=np.uint8, count=3 * count, offset=xyz_end).reshape(count, 3) if has_rgb else None,
    )
    return d


# ---------------------------------------------------------------------------------------- elevation


def grid_from_cells(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    confidence: np.ndarray,
    obstacle_h: np.ndarray,
    *,
    origin_xy: Sequence[float],
    resolution: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse per-cell samples (the /ugv/elevation/cloud fields, at cell centres) -> dense
    (height, obstacle_h, confidence) float32 arrays of shape (height, width).

    Cell index = floor((coordinate - origin) / resolution); samples outside the grid or with a non-finite
    position or height are ignored. Cells without a sample stay NaN in the height plane and 0 in the other
    two. If two samples land in one cell the later one wins.
    """
    res = _positive_finite(resolution, "resolution")
    ox, oy = _origin(origin_xy)
    width, height = int(width), int(height)
    if width < 0 or height < 0:
        raise ValueError("width and height must be >= 0")
    cols = [np.asarray(a, dtype=np.float64).ravel() for a in (x, y, z, confidence, obstacle_h)]
    if len({len(c) for c in cols}) != 1:
        raise ValueError("x, y, z, confidence and obstacle_h must have the same length")
    xs, ys, zs, conf, obst = cols
    with np.errstate(invalid="ignore"):
        ix = np.floor((xs - ox) / res)
        iy = np.floor((ys - oy) / res)
        ok = np.isfinite(ix) & np.isfinite(iy) & np.isfinite(zs) & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    h = np.full((height, width), np.nan, dtype=np.float32)
    o = np.zeros((height, width), dtype=np.float32)
    c = np.zeros((height, width), dtype=np.float32)
    row, col = iy[ok].astype(np.intp), ix[ok].astype(np.intp)
    h[row, col] = zs[ok]
    o[row, col] = obst[ok]
    c[row, col] = conf[ok]
    return h, o, c


def _block_reduce(
    h: np.ndarray, o: np.ndarray, c: np.ndarray, f: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f x f blocks: max height, max obstacle and min confidence over the known cells; a block with no known
    cell stays unknown. Edge blocks are padded with unknown cells."""
    rows, cols = h.shape
    pad = ((0, -rows % f), (0, -cols % f))
    h = np.pad(h, pad, constant_values=np.nan)
    o = np.pad(o, pad, constant_values=0.0)
    c = np.pad(c, pad, constant_values=0.0)
    known = np.isfinite(h)
    shape = (h.shape[0] // f, f, h.shape[1] // f, f)
    any_known = known.reshape(shape).any(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        hb = np.where(known, h, -np.inf).reshape(shape).max(axis=(1, 3))
        ob = np.where(known, o, -np.inf).reshape(shape).max(axis=(1, 3))
        cb = np.where(known, c, np.inf).reshape(shape).min(axis=(1, 3))
    return (
        np.where(any_known, hb, np.nan).astype(np.float32),
        np.where(any_known, ob, 0.0).astype(np.float32),
        np.where(any_known, cb, 0.0).astype(np.float32),
    )


def encode_elevation(
    height: np.ndarray,
    obstacle_h: np.ndarray,
    confidence: np.ndarray,
    *,
    epoch: int,
    seq: int,
    stamp_s: float,
    origin_xy: Sequence[float],
    resolution: float,
    max_side: int,
) -> bytes:
    """UGVE from dense (H, W) arrays (row = y index, column = x index): `height` float32 metres (NaN =
    unknown), `obstacle_h` float32 metres (0 = none), `confidence` 0..1.

    The grid is first cropped to the bounding box of known cells (origin_xy shifts by the cropped rows and
    columns), then, if a side still exceeds `max_side`, block-reduced by the smallest integer factor that
    fits (resolution multiplies by it). On the wire obstacle is uint8 in 5 cm units, rounded up so a real
    obstacle never becomes 0 and saturating at 255; confidence is uint8 0..255; unknown cells carry 0 in
    both. An all-unknown or empty grid is written with width = height = 0.
    """
    h = _plane(height, np.float32, "height")
    o = _plane(obstacle_h, np.float32, "obstacle_h")
    c = _plane(confidence, np.float32, "confidence")
    if not (h.shape == o.shape == c.shape):
        raise ValueError(f"height, obstacle_h and confidence must share a shape, got {h.shape}, {o.shape}, {c.shape}")
    max_side = int(max_side)
    if max_side < 1:
        raise ValueError("max_side must be >= 1")
    res = _positive_finite(resolution, "resolution")
    ox, oy = _origin(origin_xy)

    known = np.isfinite(h)
    o = np.where(np.isnan(o), np.float32(0.0), o)
    c = np.clip(np.nan_to_num(c, nan=0.0), 0.0, 1.0)

    if not known.any():
        h = np.zeros((0, 0), dtype=np.float32)
        o = np.zeros((0, 0), dtype=np.float32)
        c = np.zeros((0, 0), dtype=np.float32)
    else:
        used_rows = np.flatnonzero(known.any(axis=1))
        used_cols = np.flatnonzero(known.any(axis=0))
        r0, r1, c0, c1 = used_rows[0], used_rows[-1] + 1, used_cols[0], used_cols[-1] + 1
        h, o, c = h[r0:r1, c0:c1], o[r0:r1, c0:c1], c[r0:r1, c0:c1]
        ox += float(c0) * res
        oy += float(r0) * res
        factor = -(-max(h.shape) // max_side)  # ceil(longest side / max_side)
        if factor > 1:
            h, o, c = _block_reduce(h, o, c, factor)
            res *= factor
        known = np.isfinite(h)
        # one canonical NaN, and unknown cells carry zeros so equal inputs give equal bytes
        h = np.where(known, h, np.float32(np.nan)).astype(np.float32)
        o = np.where(known, o, 0.0)
        c = np.where(known, c, 0.0)

    obstacle = np.zeros(o.shape, dtype=np.uint8)
    real = o > 0
    units = np.ceil(o[real].astype(np.float64) / OBSTACLE_UNIT_M - _UNIT_TOLERANCE)
    obstacle[real] = np.clip(units, 1, 255).astype(np.uint8)
    conf = np.rint(c.astype(np.float64) * 255.0).astype(np.uint8)

    rows, cols = h.shape
    return b"".join(
        (
            _prelude(MAGIC_ELEVATION, ELEVATION_HEADER_BYTES, epoch, seq, stamp_s),
            _ELEVATION.pack(cols, rows, res, ox, oy, int(np.isfinite(h).sum())),
            np.ascontiguousarray(h, dtype="<f4").tobytes(),
            np.ascontiguousarray(obstacle).tobytes(),
            np.ascontiguousarray(conf).tobytes(),
        )
    )


def decode_elevation(b: bytes) -> dict:
    d = _decode_prelude(b, MAGIC_ELEVATION, ELEVATION_HEADER_BYTES, "elevation")
    width, height, resolution_m, origin_x, origin_y, known_cells = _ELEVATION.unpack_from(b, PRELUDE_BYTES)
    n = width * height
    _require_length(b, ELEVATION_HEADER_BYTES + 6 * n, "elevation")
    shape = (height, width)
    d.update(
        width=width,
        height=height,
        resolution_m=resolution_m,
        origin_x=origin_x,
        origin_y=origin_y,
        known_cells=known_cells,
        heights=np.frombuffer(b, dtype="<f4", count=n, offset=ELEVATION_HEADER_BYTES).reshape(shape),
        obstacle=np.frombuffer(b, dtype=np.uint8, count=n, offset=ELEVATION_HEADER_BYTES + 4 * n).reshape(shape),
        confidence=np.frombuffer(b, dtype=np.uint8, count=n, offset=ELEVATION_HEADER_BYTES + 5 * n).reshape(shape),
    )
    return d


# --------------------------------------------------------------------------------------- trajectory


def encode_trajectory(poses: np.ndarray, *, epoch: int, seq: int, stamp_s: float) -> bytes:
    """UGVT from an (N, 7) float32 array of x, y, z, qx, qy, qz, qw. Poses with a non-finite value are
    dropped. `length_m` is the 3-D length of the polyline through the poses written."""
    p = np.asarray(poses, dtype=np.float32)
    if p.size == 0:
        p = p.reshape(0, 7)
    if p.ndim != 2 or p.shape[1] != 7:
        raise ValueError(f"poses must have shape (N, 7), got {p.shape}")
    p = p[np.isfinite(p).all(axis=1)]
    steps = np.diff(p[:, :3].astype(np.float64), axis=0)
    length = float(np.sqrt((steps * steps).sum(axis=1)).sum())
    return b"".join(
        (
            _prelude(MAGIC_TRAJECTORY, TRAJECTORY_HEADER_BYTES, epoch, seq, stamp_s),
            _TRAJECTORY.pack(len(p), length),
            np.ascontiguousarray(p, dtype="<f4").tobytes(),
        )
    )


def decode_trajectory(b: bytes) -> dict:
    d = _decode_prelude(b, MAGIC_TRAJECTORY, TRAJECTORY_HEADER_BYTES, "trajectory")
    count, length_m = _TRAJECTORY.unpack_from(b, PRELUDE_BYTES)
    _require_length(b, TRAJECTORY_HEADER_BYTES + 28 * count, "trajectory")
    d.update(
        count=count,
        length_m=length_m,
        poses=np.frombuffer(b, dtype="<f4", count=7 * count, offset=TRAJECTORY_HEADER_BYTES).reshape(count, 7),
    )
    return d


# --------------------------------------------------------------------------------------- cost grid


def encode_grid(
    cells: np.ndarray,
    *,
    epoch: int,
    seq: int,
    stamp_s: float,
    resolution: float,
    origin_xy: Sequence[float],
    origin_yaw: float,
) -> bytes:
    """UGVG from an integer (H, W) array in OccupancyGrid convention (-1 unknown, 0..100 cost); values
    outside -1..100 are clamped into it."""
    a = np.asarray(cells)
    if a.ndim != 2 or a.dtype.kind not in "iu":
        raise ValueError(f"cells must be a 2-D integer array, got {a.dtype} with shape {a.shape}")
    a = np.minimum(a, 100) if a.dtype.kind == "u" else np.clip(a, -1, 100)
    a = a.astype(np.int8)
    res = _positive_finite(resolution, "resolution")
    ox, oy = _origin(origin_xy)
    rows, cols = a.shape
    return b"".join(
        (
            _prelude(MAGIC_GRID, GRID_HEADER_BYTES, epoch, seq, stamp_s),
            _GRID.pack(cols, rows, res, ox, oy, float(origin_yaw)),
            np.ascontiguousarray(a).tobytes(),
        )
    )


def decode_grid(b: bytes) -> dict:
    d = _decode_prelude(b, MAGIC_GRID, GRID_HEADER_BYTES, "grid")
    width, height, resolution_m, origin_x, origin_y, origin_yaw = _GRID.unpack_from(b, PRELUDE_BYTES)
    _require_length(b, GRID_HEADER_BYTES + width * height, "grid")
    d.update(
        width=width,
        height=height,
        resolution_m=resolution_m,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_yaw=origin_yaw,
        cells=np.frombuffer(b, dtype=np.int8, count=width * height, offset=GRID_HEADER_BYTES).reshape(height, width),
    )
    return d


# ------------------------------------------------------------------------------------------- depth


def encode_depth(
    depth_m: np.ndarray, *, epoch: int, seq: int, stamp_s: float, stride: int, max_range_m: float
) -> bytes:
    """UGVD from an (H, W) depth image in metres, decimated to every `stride`-th row and column. NaN, values
    <= 0 and values > `max_range_m` become 0 (hole); the rest become round(metres / 0.001) clipped to
    1..65535 so a real measurement is never mistaken for a hole."""
    d = np.asarray(depth_m)
    if d.ndim != 2:
        raise ValueError(f"depth_m must be a 2-D (rows, columns) array, got shape {d.shape}")
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    max_range = _positive_finite(max_range_m, "max_range_m")
    d = d[::stride, ::stride].astype(np.float64)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d) & (d > 0.0) & (d <= max_range)
    counts = np.zeros(d.shape, dtype=np.uint16)
    counts[valid] = np.clip(np.rint(d[valid] / DEPTH_UNIT_M), 1, 65535).astype(np.uint16)
    rows, cols = counts.shape
    return b"".join(
        (
            _prelude(MAGIC_DEPTH, DEPTH_HEADER_BYTES, epoch, seq, stamp_s),
            _DEPTH.pack(cols, rows, DEPTH_UNIT_M, max_range),
            np.ascontiguousarray(counts, dtype="<u2").tobytes(),
        )
    )


def decode_depth(b: bytes) -> dict:
    d = _decode_prelude(b, MAGIC_DEPTH, DEPTH_HEADER_BYTES, "depth")
    width, height, unit_m, max_range_m = _DEPTH.unpack_from(b, PRELUDE_BYTES)
    _require_length(b, DEPTH_HEADER_BYTES + 2 * width * height, "depth")
    d.update(
        width=width,
        height=height,
        unit_m=unit_m,
        max_range_m=max_range_m,
        counts=np.frombuffer(b, dtype="<u2", count=width * height, offset=DEPTH_HEADER_BYTES).reshape(height, width),
    )
    return d
