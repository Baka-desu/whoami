"""Depth accuracy: DA3 metric depth vs ground-truth depth (Dev 5 sim depth camera).

Feeds SENSOR_HONESTY.md and the Vis/MaxDepth / Grid/RangeMax caps in rtabmap_rgbd.yaml.
A pixel counts only if both depths are finite and > 0. Metrics (standard monocular-depth set):
  abs_rel  mean |est - gt| / gt          rmse_m  sqrt(mean (est - gt)^2)
  delta1   fraction with max(est/gt, gt/est) < 1.25
  median_scale_gt_over_est  median over frames of the per-frame median gt/est — a scale bias
           here biases every RGB-D constraint against wheel odometry
  coverage fraction of valid-gt pixels where est is valid (holes / sky masking)
No pixel → metric is None, never 0.0.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class DepthErrorAccumulator:
    def __init__(self, bins_m: Sequence[float] = (0.0, 2.0, 4.0, 8.0)) -> None:
        edges = [float(b) for b in bins_m]
        if not edges or edges[0] < 0.0 or any(b >= a for a, b in zip(edges[1:], edges)):
            raise ValueError(f"bins_m must be increasing and >= 0, got {list(bins_m)}")
        self._edges = edges + [math.inf]
        self._frames = 0
        self._n = 0
        self._gt_valid = 0
        self._abs_rel = 0.0
        self._sq = 0.0
        self._delta1 = 0
        self._bin_abs_rel = [0.0] * len(edges)
        self._bin_n = [0] * len(edges)
        self._frame_scales: list[float] = []

    def add(self, est: np.ndarray, gt: np.ndarray) -> None:
        if np.shape(est) != np.shape(gt):
            raise ValueError(f"est/gt shape mismatch: {np.shape(est)} vs {np.shape(gt)}")
        est = np.asarray(est, dtype=np.float64).ravel()
        gt = np.asarray(gt, dtype=np.float64).ravel()
        gt_ok = np.isfinite(gt) & (gt > 0.0)
        both = gt_ok & np.isfinite(est) & (est > 0.0)
        self._frames += 1
        self._gt_valid += int(np.count_nonzero(gt_ok))
        e, g = est[both], gt[both]
        if e.size == 0:
            return
        rel = np.abs(e - g) / g
        self._n += int(e.size)
        self._abs_rel += float(rel.sum())
        self._sq += float(np.square(e - g).sum())
        self._delta1 += int(np.count_nonzero(np.maximum(e / g, g / e) < 1.25))
        self._frame_scales.append(float(np.median(g / e)))
        for i, (lo, hi) in enumerate(zip(self._edges, self._edges[1:])):
            sel = (g >= lo) & (g < hi)
            self._bin_n[i] += int(np.count_nonzero(sel))
            self._bin_abs_rel[i] += float(rel[sel].sum())

    def report(self) -> dict:
        n = self._n

        def mean(total: float, count: int) -> float | None:
            return total / count if count else None

        return {
            "frames": self._frames,
            "pixels": n,
            "coverage": mean(float(n), self._gt_valid),
            "abs_rel": mean(self._abs_rel, n),
            "rmse_m": math.sqrt(self._sq / n) if n else None,
            "delta1": mean(float(self._delta1), n),
            "median_scale_gt_over_est": float(np.median(self._frame_scales)) if self._frame_scales else None,
            "abs_rel_by_range": {
                f"{lo}-{hi}": mean(total, count)
                for lo, hi, total, count in zip(self._edges, self._edges[1:], self._bin_abs_rel, self._bin_n)
            },
        }
