"""DA3 depth vs ground-truth depth metrics (sensor honesty numbers). Arrays only, no ROS."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ugv_localization.depth import DepthErrorAccumulator

_BINS = (0.0, 2.0, 4.0, 8.0)


def test_m1_perfect_depth() -> None:
    gt = np.array([[1.0, 3.0], [5.0, 9.0]], dtype=np.float32)
    acc = DepthErrorAccumulator(bins_m=_BINS)
    acc.add(gt.copy(), gt)
    rep = acc.report()
    assert rep["frames"] == 1 and rep["pixels"] == 4
    assert rep["abs_rel"] == pytest.approx(0.0)
    assert rep["rmse_m"] == pytest.approx(0.0)
    assert rep["delta1"] == pytest.approx(1.0)
    assert rep["median_scale_gt_over_est"] == pytest.approx(1.0)
    assert rep["coverage"] == pytest.approx(1.0)


def test_m2_uniform_scale_error_is_visible() -> None:
    gt = np.full((3, 3), 4.0, dtype=np.float32)
    est = gt * 1.1  # DA3 reads 10 % long
    acc = DepthErrorAccumulator(bins_m=_BINS)
    acc.add(est, gt)
    rep = acc.report()
    assert rep["abs_rel"] == pytest.approx(0.1, rel=1e-5)
    assert rep["rmse_m"] == pytest.approx(0.4, rel=1e-5)
    assert rep["median_scale_gt_over_est"] == pytest.approx(1 / 1.1, rel=1e-5)
    assert rep["delta1"] == pytest.approx(1.0)  # 1.1 < 1.25


def test_m3_holes_and_invalid_gt_excluded_and_counted() -> None:
    gt = np.array([2.0, 2.0, math.nan, 0.0], dtype=np.float32)
    est = np.array([2.0, math.nan, 2.0, 2.0], dtype=np.float32)
    acc = DepthErrorAccumulator(bins_m=_BINS)
    acc.add(est, gt)
    rep = acc.report()
    assert rep["pixels"] == 1  # only index 0 has both
    assert rep["coverage"] == pytest.approx(0.5)  # est valid on 1 of 2 valid-gt pixels


def test_m4_range_bins() -> None:
    gt = np.array([1.0, 3.0, 6.0, 12.0], dtype=np.float32)
    est = np.array([1.0, 3.3, 7.2, 18.0], dtype=np.float32)  # error grows with range
    acc = DepthErrorAccumulator(bins_m=_BINS)
    acc.add(est, gt)
    bins = acc.report()["abs_rel_by_range"]
    assert bins["0.0-2.0"] == pytest.approx(0.0)
    assert bins["2.0-4.0"] == pytest.approx(0.1, rel=1e-5)
    assert bins["4.0-8.0"] == pytest.approx(0.2, rel=1e-5)
    assert bins["8.0-inf"] == pytest.approx(0.5, rel=1e-5)


def test_m5_accumulates_over_frames() -> None:
    acc = DepthErrorAccumulator(bins_m=_BINS)
    acc.add(np.array([2.0], np.float32), np.array([2.0], np.float32))
    acc.add(np.array([3.0], np.float32), np.array([2.0], np.float32))
    rep = acc.report()
    assert rep["frames"] == 2 and rep["pixels"] == 2
    assert rep["abs_rel"] == pytest.approx(0.25)


def test_m6_empty_and_shape_errors() -> None:
    acc = DepthErrorAccumulator(bins_m=_BINS)
    rep = acc.report()
    assert rep["pixels"] == 0 and rep["abs_rel"] is None  # never invent numbers
    with pytest.raises(ValueError, match="shape"):
        acc.add(np.zeros((2, 2), np.float32), np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match="bins"):
        DepthErrorAccumulator(bins_m=(2.0, 1.0))
