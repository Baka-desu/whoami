"""RUGD SegFormer decode. Scripted logits. No weights, no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import AdapterError
from ugv_perception.adapter.rugd import (
    CLASS_NAMES,
    RugdSegformerAdapter,
    _resize_maps,
    decode_rugd_logits,
    preprocess_rgb,
)
from ugv_perception.compose import compose_tick, load_compose_configs
from ugv_perception.remap.load import load_remap

_ROOT = Path(__file__).resolve().parents[3]


def _frame(h: int = 4, w: int = 6) -> ImageFrame:
    return ImageFrame(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        stamp_ns=2_000_000_000,
        frame_id="camera_optical",
    )


def test_decode_tree_argmax_on_camera_hw() -> None:
    frame = _frame()
    logits = np.full((1, 25, 2, 2), -20.0, dtype=np.float32)
    logits[0, 4] = 8.0
    raw = decode_rugd_logits(logits, frame)
    assert raw.adapter_id == "rugd"
    assert raw.hw == (4, 6)
    assert set(int(x) for x in np.unique(raw.label_ids).tolist()) == {4}
    assert raw.id_to_name[4] == "tree"
    assert float(raw.raw_scores.min()) > 0.5


def test_decode_rejects_nan() -> None:
    logits = np.zeros((25, 2, 2), dtype=np.float32)
    logits[0, 0, 0] = np.nan
    try:
        decode_rugd_logits(logits, _frame())
    except AdapterError:
        return
    raise AssertionError("NaN logits must raise")


def _ref_bilinear_plane(src: np.ndarray, h: int, w: int) -> np.ndarray:
    mh, mw = src.shape
    src_f = src.astype(np.float32, copy=False)
    ys = np.clip((np.arange(h, dtype=np.float32) + 0.5) * (mh / h) - 0.5, 0.0, mh - 1)
    xs = np.clip((np.arange(w, dtype=np.float32) + 0.5) * (mw / w) - 0.5, 0.0, mw - 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    y0 = np.floor(yy).astype(np.intp)
    x0 = np.floor(xx).astype(np.intp)
    y1 = np.minimum(y0 + 1, mh - 1)
    x1 = np.minimum(x0 + 1, mw - 1)
    wy = yy - y0.astype(np.float32)
    wx = xx - x0.astype(np.float32)
    return (
        src_f[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + src_f[y0, x1] * (1.0 - wy) * wx
        + src_f[y1, x0] * wy * (1.0 - wx)
        + src_f[y1, x1] * wy * wx
    )


def test_resize_maps_matches_reference_bilinear() -> None:
    rng = np.random.default_rng(0)
    src = rng.standard_normal((5, 8, 10)).astype(np.float32)
    stacked = _resize_maps(src, 16, 20)
    assert stacked.shape == (5, 16, 20)
    for c in range(5):
        np.testing.assert_allclose(
            stacked[c], _ref_bilinear_plane(src[c], 16, 20), rtol=1e-5, atol=1e-5
        )


def test_preprocess_nchw_shape() -> None:
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    blob = preprocess_rgb(rgb, input_hw=(8, 8), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    assert blob.shape == (1, 3, 8, 8)
    assert blob.dtype == np.float32


def test_ontology_maps_dirt_and_tree() -> None:
    table = load_remap(_ROOT / "config" / "ontologies" / "rugd.yaml")
    assert table.adapter_id == "rugd"
    assert table.name_to_class["dirt"] == 1
    assert table.name_to_class["gravel"] == 1
    assert table.name_to_class["tree"] == 2
    assert table.name_to_class["sky"] == 0
    assert set(CLASS_NAMES) == set(table.name_to_class)


def test_infer_falls_back_to_numpy_when_run_seg_fails() -> None:
    logits = np.full((1, 25, 2, 2), -20.0, dtype=np.float32)
    logits[0, 4] = 8.0

    class _Backend:
        def __init__(self) -> None:
            self.runs = 0
            self.seg_post_disabled = False
            self._stash = None

        def run_seg(self, blob, out_hw):
            self.runs += 1
            self.seg_post_disabled = True
            self._stash = logits
            raise AdapterError("post missing")

        def run(self, blob):
            self.runs += 1
            if self._stash is not None:
                out = self._stash
                self._stash = None
                return out
            return logits

    backend = _Backend()
    adapter = RugdSegformerAdapter(
        backend, input_hw=(2, 2), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
    )
    raw = adapter.infer(_frame(4, 6))
    assert set(int(x) for x in np.unique(raw.label_ids).tolist()) == {4}
    assert backend.runs == 2
    raw2 = adapter.infer(_frame(4, 6))
    assert set(int(x) for x in np.unique(raw2.label_ids).tolist()) == {4}
    assert backend.runs == 3


def test_nan_gpu_decode_does_not_retry_infer() -> None:
    class _Backend:
        def run_seg(self, blob, out_hw):
            raise AdapterError("RUGD logits are not finite")

        def run(self, blob):
            raise AssertionError("must not retry infer on NaN logits")

    adapter = RugdSegformerAdapter(
        _Backend(), input_hw=(2, 2), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
    )
    try:
        adapter.infer(_frame(4, 6))
    except AdapterError as exc:
        assert "not finite" in str(exc)
        return
    raise AssertionError("NaN logits must raise")


def test_infer_uses_run_decoded_when_no_run_seg() -> None:
    class _Backend:
        def run_decoded(self, blob, out_hw):
            labels = np.full(out_hw, 4, dtype=np.int32)
            scores = np.full(out_hw, 0.9, dtype=np.float32)
            return labels, scores

        def run(self, blob):
            raise AssertionError("must not fall back to run when run_decoded works")

    adapter = RugdSegformerAdapter(
        _Backend(), input_hw=(2, 2), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
    )
    raw = adapter.infer(_frame(4, 6))
    assert raw.hw == (4, 6)
    assert set(int(x) for x in np.unique(raw.label_ids).tolist()) == {4}


def test_compose_tick_uses_rugd_table() -> None:
    class _Scripted:
        def infer(self, frame: ImageFrame):
            logits = np.full((1, 25, 4, 4), -20.0, dtype=np.float32)
            logits[0, 1, :, :2] = 6.0
            logits[0, 4, :, 2:] = 6.0
            return decode_rugd_logits(logits, frame)

    table, gates, fresh = load_compose_configs(
        remap_path=_ROOT / "config" / "ontologies" / "rugd.yaml",
        gates_path=_ROOT / "config" / "perception" / "rugd.yaml",
        freshness_path=_ROOT / "config" / "perception" / "port.yaml",
    )
    frame = _frame()
    out = compose_tick(
        frame=frame,
        now_ns=frame.stamp_ns + 1_000_000,
        adapter=_Scripted(),
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert out.mask is not None
    pix = out.mask.classes
    assert int(pix[0, 0]) == 1
    assert int(pix[0, -1]) == 2
