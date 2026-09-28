"""Unorganized PointCloud2. Holes are already omitted."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def points_to_cloud(points: np.ndarray, stamp_ns: int, frame_id: str) -> PointCloud2:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(xyz.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * msg.width
    msg.is_dense = False
    msg.data = xyz.tobytes()
    return msg
