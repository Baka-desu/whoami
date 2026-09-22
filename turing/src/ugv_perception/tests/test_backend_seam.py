"""T12 seam/contract tests — no OpenVINO, no GPU, no IR, no camera."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from ugv_perception.adapter.output import AdapterError
from ugv_perception.backend import build_backend, instances_from_engine
from ugv_perception.backend.cuda_pytorch import CudaPytorchBackend
from ugv_perception.backend.openvino_gpu import OpenVinoGpuBackend

_PROMPTS = ("dirt_path", "person")
_HW = (4, 4)


def test_b1_ids() -> None:
    assert OpenVinoGpuBackend.id == "openvino_gpu"
    assert CudaPytorchBackend.id == "cuda_pytorch"


def test_b3_unknown_and_forbidden_ids() -> None:
    with pytest.raises(ValueError):
        build_backend("nope", "weights/x.xml", _PROMPTS)
    for bad in ("ultralytics", "openvino_xpu", "xpu"):
        with pytest.raises(ValueError):
            build_backend(bad, "weights/x.xml", _PROMPTS)


def test_b4_factory_import_does_not_load_openvino() -> None:
    before = "openvino" in sys.modules
    import importlib

    importlib.reload(sys.modules["ugv_perception.backend.factory"])
    if not before:
        assert "openvino" not in sys.modules


def test_b5_compose_still_has_no_engine_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "compose"
    for py in root.glob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                for name in ("openvino", "torch", "ultralytics"):
                    assert name not in stripped


def test_b6_prompt_id_is_index_plus_one() -> None:
    mask = np.ones(_HW, dtype=bool)
    out = instances_from_engine(
        rgb_hw=_HW,
        prompts=_PROMPTS,
        class_indices=[0],
        scores=[0.9],
        masks=[mask],
    )
    assert len(out) == 1
    assert out[0].prompt_id == 1
    assert type(out[0].score) is float


def test_b6_class_out_of_range_raises() -> None:
    mask = np.ones(_HW, dtype=bool)
    with pytest.raises(ValueError):
        instances_from_engine(
            rgb_hw=_HW,
            prompts=_PROMPTS,
            class_indices=[2],
            scores=[0.9],
            masks=[mask],
        )


def test_b7_numpy_float_becomes_python_float() -> None:
    mask = np.ones(_HW, dtype=bool)
    out = instances_from_engine(
        rgb_hw=_HW,
        prompts=_PROMPTS,
        class_indices=[0],
        scores=[np.float32(0.9)],
        masks=[mask],
    )
    assert type(out[0].score) is float
    assert abs(out[0].score - 0.9) < 1e-6


def test_b7_score_out_of_range_raises() -> None:
    mask = np.ones(_HW, dtype=bool)
    with pytest.raises(ValueError):
        instances_from_engine(
            rgb_hw=_HW,
            prompts=_PROMPTS,
            class_indices=[0],
            scores=[1.1],
            masks=[mask],
        )


def test_b8_uint8_mask_raises() -> None:
    with pytest.raises(TypeError):
        instances_from_engine(
            rgb_hw=_HW,
            prompts=_PROMPTS,
            class_indices=[0],
            scores=[0.9],
            masks=[np.ones(_HW, dtype=np.uint8)],
        )


def test_b8_bool_nearest_2x2_to_4x4() -> None:
    small = np.array([[True, False], [False, True]], dtype=bool)
    out = instances_from_engine(
        rgb_hw=(4, 4),
        prompts=_PROMPTS,
        class_indices=[0],
        scores=[0.5],
        masks=[small],
    )
    assert out[0].mask.shape == (4, 4)
    assert out[0].mask.dtype == np.bool_
    assert bool(out[0].mask[0, 0]) is True
    assert bool(out[0].mask[0, 3]) is False
    assert bool(out[0].mask[3, 0]) is False
    assert bool(out[0].mask[3, 3]) is True


def test_b8_float_bilinear_then_threshold() -> None:
    small = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = instances_from_engine(
        rgb_hw=(4, 4),
        prompts=_PROMPTS,
        class_indices=[1],
        scores=[0.7],
        masks=[small],
    )
    assert out[0].prompt_id == 2
    assert out[0].mask.shape == (4, 4)
    assert out[0].mask.dtype == np.bool_
    assert bool(out[0].mask[0, 0]) is True
    assert bool(out[0].mask[3, 3]) is True


def test_b9_empty() -> None:
    assert (
        instances_from_engine(
            rgb_hw=_HW,
            prompts=_PROMPTS,
            class_indices=[],
            scores=[],
            masks=[],
        )
        == ()
    )


def test_b12_backend_does_not_import_port_stack() -> None:
    root = Path(__file__).resolve().parents[1] / "backend"
    banned = ("compose", "remap", "confidence", "freshness", "adapter.pack")
    for py in root.glob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            if line.startswith("import ") or line.startswith("from "):
                assert "import openvino" not in line
                assert not line.startswith("from openvino")
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                for name in banned:
                    assert name not in stripped, f"{py.name}: {stripped}"


def test_b13_cuda_pytorch_factory_raises_on_this_box() -> None:
    with pytest.raises(RuntimeError, match="Intel Arc"):
        build_backend("cuda_pytorch", "weights/x.pt", _PROMPTS)


def test_b14_seam_does_not_need_openvino() -> None:
    out = instances_from_engine(
        rgb_hw=_HW,
        prompts=_PROMPTS,
        class_indices=[],
        scores=[],
        masks=[],
    )
    assert out == ()


def test_yolo_seg_decode_empty() -> None:
    from ugv_perception.backend.openvino_gpu import decode_yolo_seg

    nc, nm, n = 11, 32, 8
    pred = np.zeros((1, 4 + nc + nm, n), dtype=np.float32)
    proto = np.zeros((1, nm, 160, 160), dtype=np.float32)
    cls_i, scores, masks = decode_yolo_seg(pred, proto, nc, (640, 640))
    assert cls_i == []
    assert scores == []
    assert masks == []


def test_yolo_seg_decode_one_detection() -> None:
    from ugv_perception.backend.openvino_gpu import decode_yolo_seg

    nc, nm = 11, 32
    pred = np.zeros((1, 4 + nc + nm, 4), dtype=np.float32)
    pred[0, 0, 0] = 320.0
    pred[0, 1, 0] = 320.0
    pred[0, 2, 0] = 160.0
    pred[0, 3, 0] = 160.0
    pred[0, 4 + 4, 0] = 0.9  # class 4
    pred[0, 4 + nc, 0] = 1.0  # first proto coeff
    proto = np.zeros((1, nm, 160, 160), dtype=np.float32)
    proto[0, 0, 40:120, 40:120] = 8.0
    cls_i, scores, masks = decode_yolo_seg(pred, proto, nc, (640, 640))
    assert cls_i == [4]
    assert len(scores) == 1
    assert type(scores[0]) is float
    assert abs(scores[0] - 0.9) < 1e-6
    assert masks[0].shape == (160, 160)
    assert masks[0].dtype == np.float32
    assert bool(masks[0][80, 80] >= 0.5)
    assert bool(masks[0][0, 0] < 0.5)


def test_yolo_seg_decode_nms_keeps_higher_score() -> None:
    from ugv_perception.backend.openvino_gpu import decode_yolo_seg

    nc, nm = 11, 32
    pred = np.zeros((1, 4 + nc + nm, 2), dtype=np.float32)
    for i, score in ((0, 0.9), (1, 0.4)):
        pred[0, 0, i] = 320.0
        pred[0, 1, i] = 320.0
        pred[0, 2, i] = 80.0
        pred[0, 3, i] = 80.0
        pred[0, 4, i] = score
        pred[0, 4 + nc, i] = 1.0
    proto = np.zeros((1, nm, 40, 40), dtype=np.float32)
    proto[0, 0] = 4.0
    cls_i, scores, masks = decode_yolo_seg(pred, proto, nc, (640, 640))
    assert cls_i == [0]
    assert len(scores) == 1
    assert abs(scores[0] - 0.9) < 1e-6
    assert len(masks) == 1


def test_yolo_seg_decode_score_out_of_range_raises() -> None:
    from ugv_perception.backend.openvino_gpu import decode_yolo_seg

    nc, nm = 2, 4
    pred = np.zeros((1, 4 + nc + nm, 1), dtype=np.float32)
    pred[0, 0, 0] = 16.0
    pred[0, 1, 0] = 16.0
    pred[0, 2, 0] = 8.0
    pred[0, 3, 0] = 8.0
    pred[0, 4, 0] = 1.2
    proto = np.zeros((1, nm, 8, 8), dtype=np.float32)
    with pytest.raises(AdapterError):
        decode_yolo_seg(pred, proto, nc, (32, 32))


def test_openvino_gpu_run_on_ir() -> None:
    """Engine smoke: compile YOLOE-26s IR on GPU. Not outdoor product proof."""
    from ugv_perception.adapter.prompts import load_prompts
    from ugv_perception.adapter.output import Instance

    root = Path(__file__).resolve().parents[3]
    ir = root / "weights" / "yoloe-26s-seg.xml"
    if not ir.is_file():
        pytest.skip("YOLOE-26s IR missing")
    prompts = load_prompts(
        root / "config" / "perception" / "yoloe_prompts.yaml",
        root / "config" / "ontologies" / "yoloe.yaml",
    )
    try:
        backend = build_backend("openvino_gpu", str(ir), prompts)
    except Exception as exc:
        pytest.skip(f"OpenVINO GPU compile unavailable: {exc}")
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = backend.run(rgb)
    assert type(out) is tuple
    for inst in out:
        assert type(inst) is Instance
        assert type(inst.prompt_id) is int
        assert 1 <= inst.prompt_id <= len(prompts)
        assert type(inst.score) is float
        assert 0.0 <= inst.score <= 1.0
        assert inst.mask.dtype == np.bool_
        assert inst.mask.shape == (480, 640)
