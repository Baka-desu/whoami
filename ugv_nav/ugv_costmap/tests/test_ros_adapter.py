"""ugv_costmap.adapter: the ROS-free glue between ROS messages and costmap_core."""

import math

import numpy as np
import pytest

from costmap_core.projection import CameraIntrinsics, project_pixel_to_ground
from ugv_costmap.adapter import (
    OCC_FREE, OCC_LETHAL, OCC_UNKNOWN, AdapterError, costs_to_occupancy, front_roi_lethal, mount_from_tf,
    pool_mask, scale_intrinsics,
)

UNKNOWN, TRAV, HAZARD = 0, 1, 2


def _quat_optical_pitched_down(pitch):
    """Rotation robot <- optical for a forward camera pitched down by `pitch` (no roll / yaw)."""
    # Columns: optical x (right) = -y_robot; optical y (down); optical z (forward), rotated by pitch about y.
    c, s = math.cos(pitch), math.sin(pitch)
    r = np.array([
        [0.0, -s, c],
        [-1.0, 0.0, 0.0],
        [0.0, -c, -s],
    ])
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        return ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w)
    raise AssertionError("test rotation too close to 180 deg")


def test_pool_keeps_core_precedence_hazard_over_unknown_over_traversable():
    m = np.full((4, 4), TRAV, dtype=np.uint8)
    m[0, 0] = UNKNOWN          # top-left block: unknown beats traversable
    m[0, 2] = UNKNOWN
    m[1, 3] = HAZARD           # top-right block: hazard beats unknown
    out = pool_mask(m, 2)
    assert out.tolist() == [[UNKNOWN, HAZARD], [TRAV, TRAV]]


def test_pool_refuses_indivisible_mask():
    with pytest.raises(AdapterError):
        pool_mask(np.zeros((5, 4), dtype=np.uint8), 2)


def test_scaled_intrinsics_project_block_centre_like_the_full_image():
    full = CameraIntrinsics(fx=500.0, fy=510.0, cx=319.5, cy=239.5)
    pooled = scale_intrinsics(full.fx, full.fy, full.cx, full.cy, 4)
    mount = mount_from_tf((0.0, 0.0, 0.5), _quat_optical_pitched_down(math.radians(20)))
    # Pooled pixel (u', v') covers full pixels 4u'..4u'+3: its centre is 4u'+1.5.
    for u, v in [(10, 100), (80, 110), (150, 70)]:
        a = project_pixel_to_ground(u, v, pooled, mount.geometry)
        b = project_pixel_to_ground(4 * u + 1.5, 4 * v + 1.5, full, mount.geometry)
        assert a.x == pytest.approx(b.x) and a.y == pytest.approx(b.y)


def test_mount_from_tf_reads_height_pitch_and_offset():
    mount = mount_from_tf((0.2, -0.05, 0.45), _quat_optical_pitched_down(math.radians(15)))
    assert mount.geometry.camera_height == pytest.approx(0.45)
    assert mount.geometry.pitch_rad == pytest.approx(math.radians(15))
    assert (mount.x, mount.y) == pytest.approx((0.2, -0.05))


def test_mount_from_tf_refuses_yawed_camera():
    yaw = math.radians(30)
    # Yaw the pitched camera about robot z.
    qx, qy, qz, qw = _quat_optical_pitched_down(0.0)
    hz, hw = math.sin(yaw / 2), math.cos(yaw / 2)
    q = (hw * qx - hz * qy, hw * qy + hz * qx, hw * qz + hz * qw, hw * qw - hz * qz)
    with pytest.raises(AdapterError):
        mount_from_tf((0.0, 0.0, 0.5), q)


def test_costs_to_occupancy_unknown_is_never_free_and_unseen_is_unknown():
    final = np.array([[0, 254, 255, 255]], dtype=np.int64)
    coverage = np.array([[True, True, True, False]])
    assert costs_to_occupancy(final, coverage, 50).tolist() == [[OCC_FREE, OCC_LETHAL, 50, OCC_UNKNOWN]]


@pytest.mark.parametrize("bad", [0, 100, -1])
def test_unknown_occupancy_must_be_neither_free_nor_lethal(bad):
    with pytest.raises(AdapterError):
        costs_to_occupancy(np.zeros((1, 1), dtype=np.int64), np.ones((1, 1), bool), bad)


def test_failsafe_marks_field_of_view_lethal():
    cov = np.array([[True, False]])
    assert front_roi_lethal(cov).tolist() == [[OCC_LETHAL, OCC_UNKNOWN]]
