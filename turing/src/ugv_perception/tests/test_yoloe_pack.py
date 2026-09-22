"""T06 pack tests — scripted instances, no camera, no OpenVINO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.adapter import (
    ADAPTER_ID,
    UNLABELED_ID,
    UNLABELED_NAME,
    AdapterError,
    ImageFrame,
    Instance,
    YoloeAdapter,
    load_adapter_config,
    load_prompts,
    pack,
)

_ROOT = Path(__file__).resolve().parents[3]
_ONTO = _ROOT / "config" / "ontologies" / "yoloe.yaml"
_PROMPTS = _ROOT / "config" / "perception" / "yoloe_prompts.yaml"
_ADAPTER = _ROOT / "config" / "adapters" / "yoloe.yaml"

_HW = (4, 4)
_PROMPTS_T = (
    "dirt_path",
    "gravel",
    "grass",
    "sky",
    "person",
    "vehicle",
    "rock",
    "water",
    "fence",
    "tree",
    "vegetation",
)


def _frame() -> ImageFrame:
    return ImageFrame(
        rgb=np.zeros((_HW[0], _HW[1], 3), dtype=np.uint8),
        stamp_ns=1_000,
        frame_id="camera_optical",
    )


def _mask(on: list[tuple[int, int]] | None = None) -> np.ndarray:
    m = np.zeros(_HW, dtype=bool)
    if on:
        for y, x in on:
            m[y, x] = True
    return m


def _inst(prompt_id: int, score: float, cells: list[tuple[int, int]]) -> Instance:
    return Instance(prompt_id=prompt_id, score=score, mask=_mask(cells))


def test_y1_adapter_id() -> None:
    out = pack(_frame(), (), _PROMPTS_T)
    assert out.adapter_id == ADAPTER_ID
    assert type(out.adapter_id) is str


def test_y2_stamp_frame_copied_not_now() -> None:
    frame = _frame()
    out = pack(frame, (), _PROMPTS_T)
    assert out.stamp_ns == frame.stamp_ns
    assert out.frame_id == frame.frame_id
    assert type(out.stamp_ns) is int


def test_y2_numpy_stamp_rejected() -> None:
    frame = ImageFrame(
        rgb=np.zeros((_HW[0], _HW[1], 3), dtype=np.uint8),
        stamp_ns=np.int64(1000),  # type: ignore[arg-type]
        frame_id="camera_optical",
    )
    with pytest.raises(TypeError):
        pack(frame, (), _PROMPTS_T)


def test_y3_shapes_and_dtypes() -> None:
    out = pack(_frame(), (), _PROMPTS_T)
    assert out.label_ids.dtype == np.int32
    assert out.raw_scores.dtype == np.float32
    assert out.label_ids.shape == _HW
    assert out.raw_scores.shape == _HW
    assert out.hw == _HW


def test_y4_id_to_name_python_int_keys() -> None:
    out = pack(_frame(), (), _PROMPTS_T)
    assert all(type(k) is int for k in out.id_to_name)
    assert 0 in out.id_to_name
    assert set(range(1, len(_PROMPTS_T) + 1)) <= set(out.id_to_name)


def test_y5_unknown_prompt_load_error(tmp_path: Path) -> None:
    p = tmp_path / "yoloe_prompts.yaml"
    p.write_text("adapter_id: yoloe\nprompts:\n  - not_a_class\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ontology"):
        load_prompts(p, _ONTO)


def test_y5_product_prompts_load() -> None:
    prompts = load_prompts(_PROMPTS, _ONTO)
    assert prompts == _PROMPTS_T


def test_y6_unlabeled_not_in_prompts_and_id_zero(tmp_path: Path) -> None:
    p = tmp_path / "yoloe_prompts.yaml"
    p.write_text("adapter_id: yoloe\nprompts:\n  - unlabeled\n  - person\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unlabeled"):
        load_prompts(p, _ONTO)
    inst = _inst(0, 0.9, [(0, 0)])
    with pytest.raises(ValueError):
        pack(_frame(), (inst,), _PROMPTS_T)
    out = pack(_frame(), (), _PROMPTS_T)
    assert out.id_to_name[UNLABELED_ID] == UNLABELED_NAME
    assert UNLABELED_NAME not in (n for i, n in out.id_to_name.items() if i != 0)


def test_y7_higher_score_wins_score_copied() -> None:
    low = _inst(5, 0.4, [(1, 1), (1, 2)])  # person
    high = _inst(1, 0.8, [(1, 1)])  # dirt_path
    out = pack(_frame(), (low, high), _PROMPTS_T)
    assert out.label_ids[1, 1] == 1
    assert out.raw_scores[1, 1] == np.float32(0.8)
    assert out.label_ids[1, 2] == 5
    assert out.raw_scores[1, 2] == np.float32(0.4)


def test_y8_uncovered_unlabeled_zero() -> None:
    inst = _inst(5, 0.9, [(0, 0)])
    out = pack(_frame(), (inst,), _PROMPTS_T)
    assert out.label_ids[0, 0] == 5
    assert out.label_ids[3, 3] == UNLABELED_ID
    assert out.raw_scores[3, 3] == 0.0


def test_y9_score_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        pack(_frame(), (_inst(5, 1.1, [(0, 0)]),), _PROMPTS_T)
    with pytest.raises(ValueError):
        pack(_frame(), (_inst(5, -0.1, [(0, 0)]),), _PROMPTS_T)


def test_y9_score_not_python_float_raises() -> None:
    with pytest.raises(TypeError):
        pack(_frame(), (_inst(5, np.float32(0.9), [(0, 0)]),), _PROMPTS_T)  # type: ignore[arg-type]


def test_y10_packed_scores_not_transformed() -> None:
    inst = _inst(5, 0.42, [(2, 2)])
    out = pack(_frame(), (inst,), _PROMPTS_T)
    assert out.raw_scores[2, 2] == np.float32(0.42)
    assert not np.allclose(out.raw_scores[2, 2], 1.0)


def test_y11_backend_failure_raises() -> None:
    class Boom:
        id = "openvino_gpu"

        def load(self, weights_path: str, **engine_args: object) -> None:
            return None

        def run(self, rgb: np.ndarray) -> tuple[Instance, ...]:
            raise RuntimeError("gpu down")

    adapter = YoloeAdapter(Boom(), _PROMPTS_T)
    with pytest.raises(AdapterError):
        adapter.infer(_frame())


def test_y12_adapter_sources_do_not_import_neighbors() -> None:
    root = Path(__file__).resolve().parents[1] / "adapter"
    banned = ("remap", "confidence", "freshness", "rclpy", "openvino")
    for py in root.glob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                for name in banned:
                    assert name not in stripped, f"{py.name}: {stripped}"


def test_y13_pack_does_not_branch_on_backend_id() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("adapter", "pack.py").read_text()
    assert "backend.id" not in text
    assert "openvino_gpu" not in text


def test_y14_no_ros_publishers() -> None:
    root = Path(__file__).resolve().parents[1] / "adapter"
    for py in root.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "Publisher" not in text
        assert "/segmentation/" not in text
        assert "cmd_vel" not in text


def test_mask_must_be_bool_2d_matching_hw() -> None:
    """Implementation contract: mask dtype/shape before pack (not a new Y-id)."""
    bad_dtype = Instance(prompt_id=5, score=0.9, mask=np.ones(_HW, dtype=np.uint8))
    with pytest.raises(TypeError):
        pack(_frame(), (bad_dtype,), _PROMPTS_T)
    bad_shape = Instance(prompt_id=5, score=0.9, mask=np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError):
        pack(_frame(), (bad_shape,), _PROMPTS_T)
    bad_ndim = Instance(prompt_id=5, score=0.9, mask=np.zeros(4, dtype=bool))
    with pytest.raises(ValueError):
        pack(_frame(), (bad_ndim,), _PROMPTS_T)


def test_infer_with_test_backend_only() -> None:
    inst = _inst(5, 0.9, [(0, 0)])

    class Fixture:
        id = "test_only"

        def load(self, weights_path: str, **engine_args: object) -> None:
            return None

        def run(self, rgb: np.ndarray) -> tuple[Instance, ...]:
            return (inst,)

    out = YoloeAdapter(Fixture(), _PROMPTS_T).infer(_frame())
    assert out.label_ids[0, 0] == 5
    assert out.stamp_ns == 1_000


def test_adapter_config_loads() -> None:
    cfg = load_adapter_config(_ADAPTER)
    assert cfg["backend"] == "openvino_gpu"
    assert cfg["weights"].endswith(".xml")
    xml = _ROOT / cfg["weights"]
    assert xml.is_file(), f"pinned IR missing: {xml}"
    assert xml.with_suffix(".bin").is_file()


@pytest.mark.skip(reason="IR on disk; outdoor Image+CameraInfo still Dev 5 — no dummy outdoor RGB")
def test_infer_on_real_frame_blocked() -> None:
    raise AssertionError("must stay skipped")
