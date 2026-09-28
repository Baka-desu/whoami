"""T08 meter chain. No weights, no GPU."""

from __future__ import annotations

import numpy as np

from ugv_perception.depth.geometry import (
    SKY_THRESHOLD,
    backproject,
    focal_model,
    geometry_is_fresh,
    hole_safe_resize,
    k_model,
    meters_from_raw,
    model_hw,
    preprocess_nchw,
)


def _k(h: int, w: int, fx: float = 200.0, fy: float = 200.0) -> np.ndarray:
    return np.array(
        [[fx, 0.0, (w - 1) / 2.0], [0.0, fy, (h - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_plane_center_is_two_meters() -> None:
    h, w = 28, 28
    k_cam = _k(h, w)
    km, (mh, mw) = k_model(k_cam, (h, w))
    focal = focal_model(km)
    raw_z = 2.0 * 300.0 / focal
    raw = np.full((mh, mw), raw_z, dtype=np.float64)
    sky = np.zeros((mh, mw), dtype=np.float64)
    depth = hole_safe_resize(meters_from_raw(raw, focal), sky, (h, w))
    xyz = backproject(depth, k_cam)
    center = xyz[np.argmin(np.abs(xyz[:, 0]) + np.abs(xyz[:, 1]))]
    half_pixel = 0.5 * 2.0 / float(k_cam[0, 0])
    assert abs(float(center[0])) <= half_pixel + 1e-4
    assert abs(float(center[1])) <= half_pixel + 1e-4
    assert abs(float(center[2]) - 2.0) < 1e-3


def test_bilinear_through_hole_stays_a_hole() -> None:
    depth = np.array([[2.0, np.nan, 3.0]], dtype=np.float64)
    sky = np.zeros((1, 3), dtype=np.float64)
    out = hole_safe_resize(depth, sky, (1, 3))
    assert np.isnan(out[0, 1])
    assert abs(float(out[0, 0]) - 2.0) < 1e-6
    assert abs(float(out[0, 2]) - 3.0) < 1e-6


def test_sky_threshold_boundary() -> None:
    depth = np.ones((1, 2), dtype=np.float64)
    sky = np.array([[SKY_THRESHOLD, SKY_THRESHOLD - 0.01]], dtype=np.float64)
    out = hole_safe_resize(depth, sky, (1, 2))
    assert np.isnan(out[0, 0])
    assert np.isfinite(out[0, 1])


def test_nan_raw_is_absent() -> None:
    depth = np.array([[np.nan]], dtype=np.float64)
    sky = np.zeros((1, 1), dtype=np.float64)
    xyz = backproject(hole_safe_resize(depth, sky, (1, 1)), _k(1, 1))
    assert xyz.shape == (0, 3)


def test_stale_cloud_is_not_fresh() -> None:
    stamp = 1_000_000_000
    assert geometry_is_fresh(stamp + 400_000_000, stamp) is True
    assert geometry_is_fresh(stamp + 600_000_000, stamp) is False
    assert geometry_is_fresh(stamp - 1, stamp) is False


def test_preprocess_longest_side_and_patch() -> None:
    mh, mw = model_hw(100, 200)
    assert max(mh, mw) <= 504 + 14
    assert mh % 14 == 0 and mw % 14 == 0


def test_trail_still_maps_to_landscape_336x504() -> None:
    assert model_hw(408, 612) == (336, 504)
    rgb = np.zeros((408, 612, 3), dtype=np.uint8)
    blob, sized = preprocess_nchw(rgb)
    assert sized == (336, 504)
    assert blob.shape == (1, 3, 336, 504)


def test_focal_model_scales_meters() -> None:
    raw = np.array([[1.0]])
    k_lo = _k(408, 612, fx=300.0, fy=300.0)
    k_hi = _k(408, 612, fx=600.0, fy=600.0)
    f_lo = focal_model(k_model(k_lo, (408, 612))[0])
    f_hi = focal_model(k_model(k_hi, (408, 612))[0])
    z_lo = float(meters_from_raw(raw, f_lo)[0, 0])
    z_hi = float(meters_from_raw(raw, f_hi)[0, 0])
    assert abs(z_hi / z_lo - 2.0) < 1e-6


def test_export_wrapper_stops_before_sky_fill() -> None:
    text = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts"
        / "export_da3metric_openvino.py"
    ).read_text()
    assert "class Da3HeadExport" in text
    assert "_process_depth_head" in text
    assert "_process_mono_sky_estimation(" not in text.split("class Da3HeadExport", 1)[1]
