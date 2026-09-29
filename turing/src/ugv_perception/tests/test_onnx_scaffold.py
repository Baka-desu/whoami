"""T09 tutorial ONNX scaffold. No invented IR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.onnx_tutorial import (
    ADAPTER_ID,
    CLASS_NAMES,
    OnnxTutorialAdapter,
    decode_onnx_logits,
)
from ugv_perception.adapter.output import AdapterError
from ugv_perception.compose import compose_tick, load_compose_configs
from ugv_perception.remap.load import load_remap

_ROOT = Path(__file__).resolve().parents[3]


def _frame(h: int = 4, w: int = 6) -> ImageFrame:
    return ImageFrame(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        stamp_ns=2_000_000_000,
        frame_id="camera_optical",
    )


def test_softmax_puts_scores_in_unit_interval() -> None:
    logits = np.zeros((1, 3, 2, 2), dtype=np.float32)
    logits[0, 0] = -4.2
    logits[0, 1] = 0.7
    logits[0, 2] = 3.8
    raw = decode_onnx_logits(logits, _frame())
    assert raw.adapter_id == ADAPTER_ID
    assert raw.hw == (4, 6)
    assert set(int(x) for x in np.unique(raw.label_ids).tolist()) == {2}
    assert float(raw.raw_scores.min()) >= 0.0
    assert float(raw.raw_scores.max()) <= 1.0
    assert raw.id_to_name[0] == "background"
    assert raw.id_to_name[2] == "grass"


def test_nan_logits_raise() -> None:
    logits = np.zeros((3, 2, 2), dtype=np.float32)
    logits[0, 0, 0] = np.nan
    with pytest.raises(AdapterError):
        decode_onnx_logits(logits, _frame())


def test_onnx_ontology_is_eval_scaffold() -> None:
    table = load_remap(_ROOT / "config" / "ontologies" / "onnx.yaml")
    assert table.adapter_id == "onnx"
    assert table.name_to_class["background"] == 0
    assert table.name_to_class["sidewalk"] == 1
    assert table.name_to_class["grass"] == 1
    assert set(CLASS_NAMES) == set(table.name_to_class)


def test_compose_tick_pixels_are_port_ids() -> None:
    class _Scripted:
        def infer(self, frame: ImageFrame):
            logits = np.zeros((1, 3, 4, 4), dtype=np.float32)
            logits[0, 0, :, :2] = 6.0
            logits[0, 1, :, 2:] = 6.0
            return decode_onnx_logits(logits, frame)

    table, gates, fresh = load_compose_configs(
        remap_path=_ROOT / "config" / "ontologies" / "onnx.yaml",
        gates_path=_ROOT / "config" / "perception" / "onnx.yaml",
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
    pix = set(int(x) for x in np.unique(out.mask.classes).tolist())
    assert pix <= {0, 1, 2}
    assert 2 not in pix


def test_missing_ir_does_not_fall_back_to_rugd() -> None:
    from ugv_perception.backend.onnx_live import build_onnx_adapter

    with pytest.raises(FileNotFoundError, match="does not fall back"):
        build_onnx_adapter(_ROOT)


def test_default_node_source_does_not_import_onnx_adapter() -> None:
    text = (_ROOT / "src" / "ugv_perception" / "node" / "adapter_node.py").read_text()
    head, _, main = text.partition("def main")
    assert "onnx_tutorial" not in head
    assert "onnx_live" not in head
    assert 'selected == "onnx"' in main


def test_onnx_adapter_raises_on_backend_failure() -> None:
    def boom(_frame: ImageFrame):
        raise RuntimeError("gpu down")

    with pytest.raises(AdapterError):
        OnnxTutorialAdapter(boom).infer(_frame())
