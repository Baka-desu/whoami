"""Test-only reader for the gateway's binary map layers.

The shipped `ugv_api.mapcodec` only encodes: the web viewer's TypeScript decoder (`ui/src/map/codec.ts`) is the one
real consumer, and its tests decode the same golden files byte for byte. This reader lets the Python tests look at
what an encode produced. It checks the prelude and the exact length, nothing else, and returns views into `b`.
"""

from __future__ import annotations

import numpy as np

from ugv_api import mapcodec as mc


def _prelude(b: bytes, magic: bytes, header_bytes: int) -> dict:
    got, fmt, header, epoch, seq, stamp_s = mc._PRELUDE.unpack_from(b, 0)
    assert (got, fmt, header) == (magic, mc.FORMAT, header_bytes), (got, fmt, header)
    return {"epoch": epoch, "seq": seq, "stamp_s": stamp_s}


def cloud(b: bytes) -> dict:
    d = _prelude(b, mc.MAGIC_CLOUD, mc.CLOUD_HEADER_BYTES)
    count, source_count, spacing_m, flags, *bbox = mc._CLOUD.unpack_from(b, mc.PRELUDE_BYTES)
    has_rgb = bool(flags & mc.FLAG_RGB)
    assert len(b) == mc.CLOUD_HEADER_BYTES + (15 if has_rgb else 12) * count
    xyz_end = mc.CLOUD_HEADER_BYTES + 12 * count
    d.update(
        count=count, source_count=source_count, spacing_m=spacing_m, flags=flags, has_rgb=has_rgb,
        bbox_min=np.array(bbox[:3], dtype=np.float32), bbox_max=np.array(bbox[3:], dtype=np.float32),
        xyz=np.frombuffer(b, dtype="<f4", count=3 * count, offset=mc.CLOUD_HEADER_BYTES).reshape(count, 3),
        rgb=np.frombuffer(b, dtype=np.uint8, count=3 * count, offset=xyz_end).reshape(count, 3) if has_rgb else None,
    )
    return d


def trajectory(b: bytes) -> dict:
    d = _prelude(b, mc.MAGIC_TRAJECTORY, mc.TRAJECTORY_HEADER_BYTES)
    count, length_m = mc._TRAJECTORY.unpack_from(b, mc.PRELUDE_BYTES)
    assert len(b) == mc.TRAJECTORY_HEADER_BYTES + 28 * count
    poses = np.frombuffer(b, dtype="<f4", count=7 * count, offset=mc.TRAJECTORY_HEADER_BYTES).reshape(count, 7)
    d.update(count=count, length_m=length_m, poses=poses)
    return d


def grid(b: bytes) -> dict:
    d = _prelude(b, mc.MAGIC_GRID, mc.GRID_HEADER_BYTES)
    width, height, resolution_m, origin_x, origin_y, origin_yaw = mc._GRID.unpack_from(b, mc.PRELUDE_BYTES)
    assert len(b) == mc.GRID_HEADER_BYTES + width * height
    d.update(
        width=width, height=height, resolution_m=resolution_m, origin_x=origin_x, origin_y=origin_y,
        origin_yaw=origin_yaw,
        cells=np.frombuffer(b, dtype=np.int8, count=width * height, offset=mc.GRID_HEADER_BYTES).reshape(height, width),
    )
    return d
