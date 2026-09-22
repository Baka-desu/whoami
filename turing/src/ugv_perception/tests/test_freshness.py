"""T05 freshness tests — clocks and bools, no camera, no model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.freshness import (
    FreshnessResult,
    combine_degraded,
    decide_publish,
    evaluate,
    load_freshness,
)

_PRODUCT = Path(__file__).resolve().parents[3] / "config" / "perception" / "port.yaml"
_NS = 1_000_000_000
_NOW = 2_000_000_000


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _profile(tmp_path: Path, age: str = "0.5"):
    return load_freshness(
        _write(tmp_path / "port.yaml", f"perception_max_age: {age}\n")
    )


def test_f1_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_freshness(tmp_path / "missing.yaml")


def test_f2_int_max_age_typeerror(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        load_freshness(_write(tmp_path / "port.yaml", "perception_max_age: 1\n"))


def test_f2_zero_max_age_valueerror(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_freshness(_write(tmp_path / "port.yaml", "perception_max_age: 0.0\n"))


def test_f2_bool_max_age_typeerror(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        load_freshness(_write(tmp_path / "port.yaml", "perception_max_age: true\n"))


def test_product_yaml_loads() -> None:
    profile = load_freshness(_PRODUCT)
    assert type(profile.perception_max_age) is float
    assert profile.perception_max_age > 0.0


def test_f3_numpy_stamp_typeerror(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    with pytest.raises(TypeError):
        evaluate(profile, np.int64(_NOW - _NS // 10), _NOW)


def test_f3_float_stamp_typeerror(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    with pytest.raises(TypeError):
        evaluate(profile, 1.5, _NOW)


def test_f3_true_stamp_typeerror(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    with pytest.raises(TypeError):
        evaluate(profile, True, _NOW)


def test_f5_recent_stamp_is_fresh(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    stamp = _NOW - _NS // 10
    result = evaluate(profile, stamp, _NOW)
    assert result.is_fresh is True
    assert result.time_degraded is False
    assert type(result.time_degraded) is bool
    assert abs(result.age_s - 0.1) < 1e-12


def test_f5_f7_stale_stamp_time_degraded(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    stamp = _NOW - int(0.6 * _NS)
    result = evaluate(profile, stamp, _NOW)
    assert result.is_fresh is False
    assert result.time_degraded is True
    assert result.time_degraded is (not result.is_fresh)


def test_f6_future_stamp_time_degraded(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    result = evaluate(profile, _NOW + _NS, _NOW)
    assert result.age_s < 0.0
    assert result.is_fresh is False
    assert result.time_degraded is True


def test_f8_truthy_adapter_error_typeerror() -> None:
    with pytest.raises(TypeError):
        combine_degraded(False, False, 1)  # type: ignore[arg-type]


def test_f8_truthy_collapse_typeerror() -> None:
    with pytest.raises(TypeError):
        combine_degraded(False, 1, False)  # type: ignore[arg-type]


def test_f9_f11_collapse_fresh_clocks_still_publishes(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    result = evaluate(profile, _NOW - _NS // 10, _NOW)
    decision = decide_publish(result, True, False)
    assert decision.degraded is False
    assert decision.valid is True
    assert decision.publish_mask is True
    assert decision.publish_mask is decision.valid


def test_f9_f11_adapter_error_no_mask(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    result = evaluate(profile, _NOW - _NS // 10, _NOW)
    decision = decide_publish(result, False, True)
    assert decision.degraded is True
    assert decision.publish_mask is False


def test_f11_valid_publishes_mask(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    result = evaluate(profile, _NOW - _NS // 10, _NOW)
    decision = decide_publish(result, False, False)
    assert decision.valid is True
    assert decision.degraded is False
    assert decision.publish_mask is True


def test_f9_or_of_causes(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    stale = evaluate(profile, _NOW - int(0.6 * _NS), _NOW)
    assert combine_degraded(stale.time_degraded, False, False) is True
    fresh = evaluate(profile, _NOW - _NS // 10, _NOW)
    assert combine_degraded(fresh.time_degraded, False, False) is False


def test_f12_evaluate_does_not_return_stamp(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    result = evaluate(profile, _NOW - _NS // 10, _NOW)
    assert type(result) is FreshnessResult
    assert not hasattr(result, "stamp_ns")
    assert set(result.__slots__) == {"age_s", "is_fresh", "time_degraded"}


def test_f13_decide_publish_has_no_port_meta_degraded() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("freshness", "evaluate.py").read_text()
    assert "make_mask" not in text
    assert "PortMeta" not in text
    assert "rclpy" not in text


def test_f14_freshness_does_not_import_neighbors() -> None:
    root = Path(__file__).resolve().parents[1] / "freshness"
    banned = ("confidence", "remap", "rclpy", "ultralytics", "openvino")
    for py in root.glob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                for name in banned:
                    assert name not in stripped


def test_evaluate_does_not_hardcode_example_max_age() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("freshness", "evaluate.py").read_text()
    assert "0.50" not in text
    assert "0.5" not in text


def test_port_does_not_import_freshness() -> None:
    port_dir = Path(__file__).resolve().parents[1] / "port"
    for py in port_dir.glob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "freshness" not in stripped
