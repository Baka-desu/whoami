"""Error / edge branches of every kernel: bad types, bad shapes, bad YAML, degenerate geometry. No ROS."""

from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ugv_localization.camera import CalibrationError, calibration_from_camera_info, load_calibration
from ugv_localization.common import (
    age_s,
    load_yaml_mapping,
    require_exact_keys,
    require_finite,
    require_positive_float,
    require_stamp,
)
from ugv_localization.depth import DepthErrorAccumulator, depth_values
from ugv_localization.drift import associate, ate, rpe_translation, umeyama
from ugv_localization.modes import Mode
from ugv_localization.odom import (
    OdomGate,
    OdomSelector,
    SelectorProfile,
    Source,
    SourcePolicy,
    TfEdge,
    load_odom_select_config,
    odom_gate_profile_from_mapping,
)
from ugv_localization.tools.check_rtabmap_params import main as check_main
from ugv_localization.tools.drift_report import read_csv
from ugv_localization.validity import PoseValidityMonitor

_PKG = Path(__file__).resolve().parents[1]
_K = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
_P = [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
_GATE = {"odom_frame": "odom", "base_frame": "base_link", "max_future_s": 0.05, "quat_norm_tol": 0.01}
_PROFILE = SelectorProfile(0.25, 0.5, 1.0, 1000.0)


# --- common.checks / yamlio ---------------------------------------------------------------


def test_e1_require_stamp_and_numbers() -> None:
    with pytest.raises(ValueError, match="> 0"):
        require_stamp(0, name="t")
    with pytest.raises(TypeError):
        require_stamp(True, name="t")
    assert require_finite(3, name="x") == 3.0
    with pytest.raises(TypeError):
        require_finite(True, name="x")
    with pytest.raises(TypeError):
        require_finite("1.0", name="x")
    with pytest.raises(ValueError, match="finite"):
        require_finite(math.inf, name="x")
    with pytest.raises(ValueError, match="finite"):
        require_positive_float(math.nan, name="x")
    assert age_s(1_000_000_000, 3_500_000_000) == pytest.approx(2.5)


def test_e2_yaml_mapping_and_keys(tmp_path: Path) -> None:
    p = tmp_path / "l.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml_mapping(p)
    with pytest.raises(FileNotFoundError):
        load_yaml_mapping(tmp_path / "missing.yaml")
    with pytest.raises(KeyError, match="missing"):
        require_exact_keys({"a": 1}, {"a", "b"}, where="w")


# --- odom gate / selector -----------------------------------------------------------------


def test_e3_gate_profile_mapping_errors() -> None:
    with pytest.raises(ValueError, match="mapping"):
        odom_gate_profile_from_mapping(["odom"], where="w")
    with pytest.raises(TypeError, match="odom_frame"):
        odom_gate_profile_from_mapping({**_GATE, "odom_frame": ""}, where="w")
    with pytest.raises(TypeError):
        OdomGate("profile")  # type: ignore[arg-type]


def test_e4_select_config_errors(tmp_path: Path) -> None:
    text = (_PKG / "config" / "odom_select.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({**data, "selector": [1, 2]}), encoding="utf-8")
    with pytest.raises(ValueError, match="selector must be a mapping"):
        load_odom_select_config(p)
    p.write_text(yaml.safe_dump({**data, "visual_gate": {**_GATE, "extra": 1}}), encoding="utf-8")
    with pytest.raises(KeyError, match="extra"):
        load_odom_select_config(p)
    sel = dict(data["selector"])
    sel.pop("visual_timeout_s")
    p.write_text(yaml.safe_dump({**data, "selector": sel}), encoding="utf-8")
    with pytest.raises(KeyError, match="visual_timeout_s"):
        load_odom_select_config(p)


def test_e5_selector_type_checks_and_active() -> None:
    with pytest.raises(TypeError, match="policy"):
        OdomSelector(_PROFILE, "auto")  # type: ignore[arg-type]
    s = OdomSelector(_PROFILE, SourcePolicy.AUTO)
    assert s.active is None
    edge = TfEdge(100, "odom", "base_link", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    s.on_sample(Source.WHEEL, edge, (0.01,) * 36, now_ns=100)
    assert s.active is Source.WHEEL
    with pytest.raises(TypeError):
        s.on_sample(Source.WHEEL, edge, (0.01,) * 36, now_ns=1.0)  # type: ignore[arg-type]


def test_e6_selector_3d_rotation_continuity() -> None:
    # Pitched / rolled poses (non-planar) must also re-anchor without a jump.
    s = OdomSelector(_PROFILE, SourcePolicy.AUTO)
    q_roll = (math.sin(0.2), 0.0, 0.0, math.cos(0.2))
    s.on_sample(Source.WHEEL, TfEdge(1_000_000_000, "odom", "base_link", (1.0, 2.0, 0.3), q_roll), (0.01,) * 36, 1_000_000_000)
    q_pitch = (0.0, math.sin(0.3), 0.0, math.cos(0.3))
    t = 1_400_000_000
    res = s.on_sample(Source.VISUAL, TfEdge(t, "odom", "base_link", (-3.0, 5.0, 1.0), q_pitch), (0.01,) * 36, t)
    out = res.output
    assert out is not None
    assert out.translation == pytest.approx((1.0, 2.0, 0.3), abs=1e-9)
    assert out.rotation == pytest.approx(q_roll, abs=1e-9)
    assert math.isclose(sum(c * c for c in out.rotation), 1.0, rel_tol=1e-12)


# --- validity ------------------------------------------------------------------------------


def test_e7_validity_profile_type_check() -> None:
    with pytest.raises(TypeError, match="ValidityProfile"):
        PoseValidityMonitor("profile", Mode.MAPPING)  # type: ignore[arg-type]


# --- depth --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(width=0, height=1, step=4, stride=1),
        dict(width=2, height=0, step=8, stride=1),
        dict(width=2, height=1, step=4, stride=1),  # step < 4 * width
        dict(width=2, height=1, step=8, stride=0),
    ],
)
def test_e8_depth_values_bad_geometry(kw: dict) -> None:
    with pytest.raises(ValueError, match="geometry"):
        depth_values(b"\x00" * 64, is_bigendian=False, **kw)


def test_e9_depth_metrics_frame_without_overlap_counts_frame_only() -> None:
    acc = DepthErrorAccumulator()
    acc.add(np.array([np.nan, 2.0]), np.array([1.0, np.nan]))
    rep = acc.report()
    assert rep["frames"] == 1 and rep["pixels"] == 0 and rep["coverage"] == 0.0
    assert rep["median_scale_gt_over_est"] is None


# --- drift ---------------------------------------------------------------------------------


def test_e10_drift_input_validation() -> None:
    good = np.zeros((4, 3))
    with pytest.raises(ValueError, match=r"\(N,3\)"):
        ate(np.zeros((4, 2)), good)
    with pytest.raises(ValueError, match="non-finite"):
        ate(np.full((4, 3), np.nan), good)
    with pytest.raises(ValueError, match=">= 3"):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="same length"):
        ate(np.zeros((3, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="same length"):
        rpe_translation(np.zeros((3, 3)), np.zeros((4, 3)), 1.0)
    with pytest.raises(ValueError, match="sorted"):
        associate([1, 2], [5, 3], max_dt_ns=10)


def test_e11_umeyama_never_returns_a_reflection() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(10, 3))
    dst = src * np.array([1.0, 1.0, -1.0])  # mirror image: best orthogonal fit is a reflection
    rot, _, _ = umeyama(src, dst)
    assert np.linalg.det(rot) == pytest.approx(1.0)


# --- camera calibration --------------------------------------------------------------------


def _cal(**over):
    kw = dict(camera_name="cam", width=640, height=480, distortion_model="plumb_bob", d=[0.0] * 5, k=_K, r=_R, p=_P)
    kw.update(over)
    return calibration_from_camera_info(**kw)


def test_e12_calibration_from_camera_info_errors() -> None:
    with pytest.raises(CalibrationError, match="camera_name"):
        _cal(camera_name="")
    with pytest.raises(CalibrationError, match="list of numbers"):
        _cal(k=["a"] * 9)
    with pytest.raises(CalibrationError, match="finite"):
        _cal(d=[math.nan] * 5)
    with pytest.raises(CalibrationError, match="9 values"):
        _cal(r=_R[:8])
    with pytest.raises(CalibrationError, match="12 values"):
        _cal(p=_P[:11])


def test_e13_load_calibration_errors(tmp_path: Path) -> None:
    base = {
        "image_width": 640,
        "image_height": 480,
        "camera_name": "cam",
        "camera_matrix": {"rows": 3, "cols": 3, "data": _K},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.0] * 5},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": _R},
        "projection_matrix": {"rows": 3, "cols": 4, "data": _P},
    }
    p = tmp_path / "c.yaml"
    for over, match in [
        ({"distortion_model": 5}, "distortion_model must be a string"),
        ({"camera_matrix": [1, 2]}, "rows, cols, data"),
        ({"camera_matrix": {"rows": 2, "cols": 3, "data": _K[:6]}}, "3x3"),
    ]:
        p.write_text(yaml.safe_dump({**base, **over}), encoding="utf-8")
        with pytest.raises(CalibrationError, match=match):
            load_calibration(p)


# --- tools ---------------------------------------------------------------------------------


def test_e14_check_params_default_configs_and_bad_node_check(tmp_path: Path, capsys) -> None:
    dump = tmp_path / "d.txt"
    names = set()
    for cfg in ("rtabmap_rgbd.yaml", "rgbd_odometry.yaml"):
        params = yaml.safe_load((_PKG / "config" / cfg).read_text(encoding="utf-8"))["/**"]["ros__parameters"]
        names |= {k for k in params if "/" in k}
    names |= {"Mem/IncrementalMemory", "Mem/InitWMWithAllNodes"}
    dump.write_text("".join(f'Param: {n} = "x"   [..]\n' for n in sorted(names)), encoding="utf-8")
    assert check_main(["--dump", str(dump)]) == 0  # default = installed/source rgbd configs
    assert "OK:" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="CONFIG=DUMP"):
        check_main(["--dump", str(dump), "--node-check", "no_equals_sign"])


def test_e15_drift_csv_header_checked(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    p.write_text("t,x,y,z\n1,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_csv(p)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
@pytest.mark.parametrize("module", ["ugv_localization.tools.check_rtabmap_params", "ugv_localization.tools.drift_report"])
def test_e16_tools_run_as_scripts(module: str, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(module, run_name="__main__")
    assert exc.value.code == 0
