"""T04 confidence tests — numeric tables, no camera, no model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.confidence import apply, load_gates
from ugv_perception.port import assert_canonical

_PRODUCT = Path(__file__).resolve().parents[3] / "config" / "perception" / "yoloe.yaml"

_BASE = """
adapter_id: yoloe
normalizer: identity
tau_min: 0.25
tau_trav: 0.5
tau_haz: 0.35
kappa: 0.1
min_known_fraction: 0.05
"""


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _classes(*rows: list[int]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8)


def _scores(hw: tuple[int, int] | np.ndarray, value: float = 0.9) -> np.ndarray:
    if isinstance(hw, np.ndarray):
        hw = hw.shape
    return np.full(hw, value, dtype=np.float32)


def _profile(tmp_path: Path, extra: str = "") -> object:
    return load_gates(_write(tmp_path / "yoloe.yaml", _BASE + extra))


# --- load ---


def test_g1_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gates(tmp_path / "missing.yaml")


def test_g3_minmax_load_error(tmp_path: Path) -> None:
    body = _BASE.replace("identity", "minmax")
    with pytest.raises(ValueError, match="minmax"):
        load_gates(_write(tmp_path / "yoloe.yaml", body))


def test_g4_tau_bool_typeerror(tmp_path: Path) -> None:
    body = _BASE.replace("tau_min: 0.25", "tau_min: true")
    with pytest.raises(TypeError):
        load_gates(_write(tmp_path / "yoloe.yaml", body))


def test_g6_tau_int_typeerror(tmp_path: Path) -> None:
    body = _BASE.replace("tau_min: 0.25", "tau_min: 1")
    with pytest.raises(TypeError):
        load_gates(_write(tmp_path / "yoloe.yaml", body))


def test_g4_tau_out_of_range(tmp_path: Path) -> None:
    body = _BASE.replace("tau_min: 0.25", "tau_min: 1.5")
    with pytest.raises(ValueError):
        load_gates(_write(tmp_path / "yoloe.yaml", body))


def test_g5_sigmoid_missing_params(tmp_path: Path) -> None:
    body = _BASE.replace("identity", "sigmoid")
    with pytest.raises(ValueError):
        load_gates(_write(tmp_path / "yoloe.yaml", body))


def test_g5_sigmoid_scale_zero(tmp_path: Path) -> None:
    body = _BASE.replace("identity", "sigmoid") + "sigmoid_center: 0.0\nsigmoid_scale: 0.0\n"
    with pytest.raises(ValueError):
        load_gates(_write(tmp_path / "yoloe.yaml", body))


def test_g5_identity_rejects_sigmoid_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_gates(_write(tmp_path / "yoloe.yaml", _BASE + "sigmoid_center: 0.0\n"))


def test_c1_stem_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C1"):
        load_gates(_write(tmp_path / "onnx.yaml", _BASE))


def test_product_yaml_loads() -> None:
    profile = load_gates(_PRODUCT)
    assert profile.adapter_id == "yoloe"
    assert profile.normalizer == "identity"


# --- apply ---


def test_g7_wrong_adapter_id(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    with pytest.raises(ValueError):
        apply(
            profile,
            adapter_id="onnx",
            classes=classes,
            raw_scores=_scores(classes),
        )


def test_g10_identity_above_one_errors(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    scores = _scores(classes, 1.0001)
    with pytest.raises(ValueError):
        apply(profile, adapter_id="yoloe", classes=classes, raw_scores=scores)


def test_g10_identity_negative_errors(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    scores = _scores(classes, -0.1)
    with pytest.raises(ValueError):
        apply(profile, adapter_id="yoloe", classes=classes, raw_scores=scores)


def test_g10_identity_nan_errors(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    scores = _scores(classes, 0.5)
    scores[0, 0] = np.nan
    with pytest.raises(ValueError):
        apply(profile, adapter_id="yoloe", classes=classes, raw_scores=scores)


def test_g10_constant_stays_not_one(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    scores = _scores(classes, 0.42)
    out_c, conf, _, _ = apply(
        profile, adapter_id="yoloe", classes=classes, raw_scores=scores
    )
    assert conf is scores
    assert np.allclose(conf, 0.42)
    assert not np.allclose(conf, 1.0)
    assert np.all(out_c == 0)  # 0.42 < tau_trav 0.5 and tau_min 0.25? 0.42 > 0.25, class 1 < 0.5 → 0


def test_g12_low_score_class1_becomes_unknown(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    scores = _scores(classes, 0.1)
    out_c, _, _, _ = apply(profile, adapter_id="yoloe", classes=classes, raw_scores=scores)
    assert np.all(out_c == 0)


def test_g12_high_trav_survives(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    scores = _scores(classes, 0.9)
    out_c, conf, frac, coll = apply(
        profile, adapter_id="yoloe", classes=classes, raw_scores=scores
    )
    assert np.all(out_c == 1)
    assert np.all((conf >= 0.0) & (conf <= 1.0))
    assert frac == 1.0
    assert coll is False


def test_g12_hazard_near_tau(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([2, 2], [2, 2])
    keep, _, _, _ = apply(
        profile, adapter_id="yoloe", classes=classes, raw_scores=_scores(classes, 0.4)
    )
    drop, _, _, _ = apply(
        profile, adapter_id="yoloe", classes=classes, raw_scores=_scores(classes, 0.2)
    )
    assert np.all(keep == 2)
    assert np.all(drop == 0)


def test_g13_unknown_stays_unknown(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([0, 0], [0, 0])
    scores = _scores(classes, 0.99)
    out_c, _, _, _ = apply(profile, adapter_id="yoloe", classes=classes, raw_scores=scores)
    assert np.all(out_c == 0)


def test_g14_kappa_only_on_survivors(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 0], [1, 1])
    top = np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32)
    runner = np.array([[0.85, 0.0], [0.1, 0.85]], dtype=np.float32)
    out_c, _, _, _ = apply(
        profile,
        adapter_id="yoloe",
        classes=classes,
        raw_scores=top,
        runner_up=runner,
    )
    assert out_c[0, 0] == 0
    assert out_c[0, 1] == 0
    assert out_c[1, 0] == 1
    assert out_c[1, 1] == 0


def test_g14_no_runner_up_skips_kappa(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    out_c, _, _, _ = apply(
        profile, adapter_id="yoloe", classes=classes, raw_scores=_scores(classes, 0.9)
    )
    assert np.all(out_c == 1)


def test_g14_sigmoid_compares_normalized_not_raw(tmp_path: Path) -> None:
    body = (
        "adapter_id: yoloe\n"
        "normalizer: sigmoid\n"
        "tau_min: 0.0\n"
        "tau_trav: 0.0\n"
        "tau_haz: 0.0\n"
        "kappa: 0.05\n"
        "min_known_fraction: 0.0\n"
        "sigmoid_center: 0.0\n"
        "sigmoid_scale: 1.0\n"
    )
    profile = load_gates(_write(tmp_path / "yoloe.yaml", body))
    classes = _classes([1, 1], [1, 1])
    top = np.full((2, 2), 10.0, dtype=np.float32)
    runner = np.full((2, 2), 9.5, dtype=np.float32)
    out_c, conf, _, _ = apply(
        profile,
        adapter_id="yoloe",
        classes=classes,
        raw_scores=top,
        runner_up=runner,
    )
    assert conf is not top
    assert np.all(out_c == 0)
    assert not np.shares_memory(conf, top)


def test_g15_known_fraction_after_kappa(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    top = _scores(classes, 0.9)
    runner = _scores(classes, 0.85)
    _, _, frac, coll = apply(
        profile,
        adapter_id="yoloe",
        classes=classes,
        raw_scores=top,
        runner_up=runner,
    )
    assert frac == 0.0
    assert coll is True
    assert type(coll) is bool
    assert type(frac) is float


def test_g16_no_degraded_no_mutate(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 1], [1, 1])
    scores = _scores(classes, 0.1)
    before_c = classes.copy()
    before_s = scores.copy()
    out = apply(profile, adapter_id="yoloe", classes=classes, raw_scores=scores)
    assert len(out) == 4
    classes_out, conf, _frac, coll = out
    assert classes_out is not classes
    assert np.array_equal(classes, before_c)
    assert np.array_equal(scores, before_s)
    assert conf is scores
    assert coll is True
    assert not hasattr(out, "degraded")


def test_g17_output_canonical(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    classes = _classes([1, 2], [0, 1])
    scores = np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32)
    out_c, conf, _, _ = apply(
        profile, adapter_id="yoloe", classes=classes, raw_scores=scores
    )
    assert_canonical(out_c)
    assert conf.dtype == np.float32
    assert np.all((conf >= 0.0) & (conf <= 1.0))


def test_two_profiles_not_shared(tmp_path: Path) -> None:
    a = load_gates(_write(tmp_path / "yoloe.yaml", _BASE))
    other = (
        "adapter_id: onnx\n"
        "normalizer: identity\n"
        "tau_min: 0.9\n"
        "tau_trav: 0.9\n"
        "tau_haz: 0.9\n"
        "kappa: 0.0\n"
        "min_known_fraction: 0.0\n"
    )
    b = load_gates(_write(tmp_path / "onnx.yaml", other))
    assert a.tau_trav != b.tau_trav
    assert a is not b


def test_apply_does_not_hardcode_example_tau() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("confidence", "apply.py").read_text()
    assert "0.25" not in text
    assert "0.35" not in text
    assert "0.05" not in text


def test_port_and_remap_do_not_import_confidence() -> None:
    root = Path(__file__).resolve().parents[1]
    for folder in ("port", "remap"):
        for py in (root / folder).glob("*.py"):
            for line in py.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    assert "confidence" not in stripped
