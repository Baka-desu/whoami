"""T03 remap tests — name tables only, no camera, no model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.port import UNKNOWN, assert_canonical
from ugv_perception.remap import apply, load_remap

_YOLOE_YAML = Path(__file__).resolve().parents[3] / "config" / "ontologies" / "yoloe.yaml"


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _ids(*rows: list[int]) -> np.ndarray:
    return np.array(rows, dtype=np.int32)


def _scores(hw: tuple[int, int], value: float = 0.9) -> np.ndarray:
    return np.full(hw, value, dtype=np.float32)


# --- load ---


def test_r1_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_remap(tmp_path / "missing.yaml")


def test_r2_empty_adapter_id_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: \"\"\nmap:\n  person: 2\ndefault: 0\n",
    )
    with pytest.raises(ValueError):
        load_remap(p)


def test_c1_stem_mismatch_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: onnx\nmap:\n  person: 2\ndefault: 0\n",
    )
    with pytest.raises(ValueError, match="C1"):
        load_remap(p)


def test_r5_cautious_key_fails(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: yoloe\nmap:\n  cautious: 1\ndefault: 0\n",
    )
    with pytest.raises(ValueError):
        load_remap(p)


def test_r5_class_three_fails(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: yoloe\nmap:\n  person: 3\ndefault: 0\n",
    )
    with pytest.raises(ValueError):
        load_remap(p)


def test_r5_bool_value_fails(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: yoloe\nmap:\n  person: true\ndefault: 0\n",
    )
    with pytest.raises(TypeError):
        load_remap(p)


def test_r5_float_value_fails(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: yoloe\nmap:\n  person: 1.0\ndefault: 0\n",
    )
    with pytest.raises(TypeError):
        load_remap(p)


def test_r5_string_value_fails(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: yoloe\nmap:\n  person: \"1\"\ndefault: 0\n",
    )
    with pytest.raises(TypeError):
        load_remap(p)


def test_r6_default_one_fails(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: yoloe\nmap:\n  person: 2\ndefault: 1\n",
    )
    with pytest.raises(ValueError):
        load_remap(p)


def test_r6_default_zero_passes() -> None:
    table = load_remap(_YOLOE_YAML)
    assert table.default == UNKNOWN
    assert table.adapter_id == "yoloe"


def test_r7_duplicate_keys_fail(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "yoloe.yaml",
        "adapter_id: yoloe\nmap:\n  person: 2\n  person: 1\ndefault: 0\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_remap(p)


def test_empty_map_legal(tmp_path: Path) -> None:
    p = _write(tmp_path / "yoloe.yaml", "adapter_id: yoloe\nmap: {}\ndefault: 0\n")
    table = load_remap(p)
    labels = _ids([0, 0], [0, 0])
    names = {0: "anything"}
    scores = _scores(labels.shape)
    classes, _ = apply(table, adapter_id="yoloe", label_ids=labels, id_to_name=names, raw_scores=scores)
    assert np.all(classes == 0)


def test_second_adapter_without_file_cannot_load(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_remap(tmp_path / "onnx.yaml")


# --- apply ---


def test_r8_wrong_adapter_id_raises() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([0, 1], [1, 0])
    names = {0: "person", 1: "dirt_path"}
    with pytest.raises(ValueError):
        apply(
            table,
            adapter_id="onnx",
            label_ids=labels,
            id_to_name=names,
            raw_scores=_scores(labels.shape),
        )


def test_r8_adapter_id_int_raises() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([0, 1], [1, 0])
    names = {0: "person", 1: "dirt_path"}
    with pytest.raises(TypeError):
        apply(
            table,
            adapter_id=1,  # type: ignore[arg-type]
            label_ids=labels,
            id_to_name=names,
            raw_scores=_scores(labels.shape),
        )


def test_r9_float_labels_fail() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = np.array([[0, 1], [1, 0]], dtype=np.float32)
    with pytest.raises(TypeError):
        apply(
            table,
            adapter_id="yoloe",
            label_ids=labels,
            id_to_name={0: "person", 1: "dirt_path"},
            raw_scores=_scores((2, 2)),
        )


def test_r9_1d_labels_fail() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = np.array([0, 1], dtype=np.int32)
    with pytest.raises(ValueError):
        apply(
            table,
            adapter_id="yoloe",
            label_ids=labels,
            id_to_name={0: "person", 1: "dirt_path"},
            raw_scores=np.zeros(2, dtype=np.float32),
        )


def test_r9_negative_id_fails() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([0, -1], [1, 0])
    with pytest.raises(ValueError):
        apply(
            table,
            adapter_id="yoloe",
            label_ids=labels,
            id_to_name={0: "person", 1: "dirt_path"},
            raw_scores=_scores((2, 2)),
        )


def test_r10_numpy_int_key_fails() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([1, 1], [1, 1])
    names = {np.int64(1): "person"}
    with pytest.raises(TypeError):
        apply(
            table,
            adapter_id="yoloe",
            label_ids=labels,
            id_to_name=names,  # type: ignore[arg-type]
            raw_scores=_scores(labels.shape),
        )


def test_r11_nameless_id_raises_not_zero() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([0, 9], [1, 0])
    names = {0: "person", 1: "dirt_path"}
    with pytest.raises(ValueError, match="id_to_name"):
        apply(
            table,
            adapter_id="yoloe",
            label_ids=labels,
            id_to_name=names,
            raw_scores=_scores(labels.shape),
        )


def test_r12_unmapped_name_defaults_to_unknown() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([0, 1], [2, 0])
    names = {0: "person", 1: "dirt_path", 2: "mystery"}
    classes, _ = apply(
        table,
        adapter_id="yoloe",
        label_ids=labels,
        id_to_name=names,
        raw_scores=_scores(labels.shape),
    )
    assert classes[0, 0] == 2
    assert classes[0, 1] == 1
    assert classes[1, 0] == 0
    assert classes.dtype == np.uint8


def test_r13_scores_identity_and_unchanged() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([0, 1], [1, 0])
    names = {0: "person", 1: "dirt_path"}
    scores = _scores(labels.shape, 0.4)
    before = scores.copy()
    classes, out = apply(
        table,
        adapter_id="yoloe",
        label_ids=labels,
        id_to_name=names,
        raw_scores=scores,
    )
    assert out is scores
    assert np.array_equal(scores, before)
    assert classes.shape == labels.shape


def test_r13_shape_mismatch_raises() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([0, 1], [1, 0])
    with pytest.raises(ValueError):
        apply(
            table,
            adapter_id="yoloe",
            label_ids=labels,
            id_to_name={0: "person", 1: "dirt_path"},
            raw_scores=_scores((3, 3)),
        )


def test_r14_output_canonical_unknown_stays_zero() -> None:
    table = load_remap(_YOLOE_YAML)
    labels = _ids([0, 0], [0, 0])
    names = {0: "sky"}
    classes, _ = apply(
        table,
        adapter_id="yoloe",
        label_ids=labels,
        id_to_name=names,
        raw_scores=_scores(labels.shape),
    )
    assert_canonical(classes)
    assert np.all(classes == 0)
    assert classes.dtype == np.uint8


def test_r15_apply_has_no_python_pixel_loop() -> None:
    src = Path(__file__).resolve().parents[1] / "remap" / "apply.py"
    text = src.read_text(encoding="utf-8")
    assert "for y in range" not in text
    assert "for x in range" not in text
    assert ".item()" not in text


def test_port_does_not_import_remap() -> None:
    port_dir = Path(__file__).resolve().parents[1] / "port"
    for py in port_dir.glob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "remap" not in stripped
