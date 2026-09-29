"""Offline tools: rtabmap param-name checker + drift_report CLI. No ROS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ugv_localization.modes import Mode, plan_mode
from ugv_localization.tools.check_rtabmap_params import (
    config_library_keys,
    config_node_keys,
    main as check_main,
    parse_node_param_list,
    parse_params_dump,
    unknown_keys,
)
from ugv_localization.tools.drift_report import main as drift_main

_PKG = Path(__file__).resolve().parents[1]
_PRODUCT_CFGS = [_PKG / "config" / n for n in ("rtabmap_rgbd.yaml", "rgbd_odometry.yaml", "rgbd_sync.yaml", "cloud_to_depth.yaml")]

# Format of `ros2 run rtabmap_slam rtabmap --params` (one line per library parameter).
_DUMP = """\
Param: Kp/MaxFeatures = "500"                 [Maximum features extracted from the images.]
Param: Mem/IncrementalMemory = "true"         [SLAM mode, otherwise it is Localization mode.]
Param: Reg/Strategy = "0"                     [0=Vis, 1=Icp, 2=VisIcp]
some unrelated banner line
"""


def test_k1_parse_dump() -> None:
    assert parse_params_dump(_DUMP) == {"Kp/MaxFeatures", "Mem/IncrementalMemory", "Reg/Strategy"}


def test_k2_unrecognized_dump_fails_loud() -> None:
    with pytest.raises(ValueError, match="no parameters"):
        parse_params_dump("rtabmap: command not found\n")


def test_k3_config_keys_only_library_params(tmp_path: Path) -> None:
    cfg = tmp_path / "r.yaml"
    cfg.write_text(
        "/**:\n  ros__parameters:\n    frame_id: base_link\n    Kp/MaxFeatures: '500'\n    Reg/Strategy: '0'\n",
        encoding="utf-8",
    )
    assert config_library_keys(cfg) == {"Kp/MaxFeatures", "Reg/Strategy"}


def test_k4_unknown_keys() -> None:
    assert unknown_keys({"Kp/MaxFeatures", "Made/Up"}, {"Kp/MaxFeatures"}) == ["Made/Up"]


@pytest.mark.parametrize("cfg", _PRODUCT_CFGS, ids=lambda p: p.name)
def test_k5_product_config_library_values_are_strings(cfg: Path) -> None:
    # rtabmap_ros declares library params as strings; YAML bools/ints would type-clash at load.
    import yaml

    params = yaml.safe_load(cfg.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    bad = {k: v for k, v in params.items() if "/" in k and not isinstance(v, str)}
    assert bad == {}


@pytest.mark.parametrize("cfg", _PRODUCT_CFGS, ids=lambda p: p.name)
def test_k6_product_config_does_not_hardcode_mode_params(cfg: Path) -> None:
    # Mode params come only from modes.plan_mode (single source of truth).
    keys = config_library_keys(cfg)
    mode_keys = set(plan_mode(Mode.MAPPING, str(_PKG / "x.db"), fresh=False).rtabmap_params())
    assert keys.isdisjoint(mode_keys)


def test_k6b_rgbd_invariants() -> None:
    # RGB-D only (mindmap D7): no mono-only knobs, depth drives the grid, odom never owns TF.
    import yaml

    def params(name: str) -> dict:
        return yaml.safe_load((_PKG / "config" / name).read_text(encoding="utf-8"))["/**"]["ros__parameters"]

    slam, vo, sync = params("rtabmap_rgbd.yaml"), params("rgbd_odometry.yaml"), params("rgbd_sync.yaml")
    c2d = params("cloud_to_depth.yaml")
    # Dev 1 cloud → depth image: exact stamps, full resolution, never invent depth
    assert c2d["approx"] is False and c2d["decimation"] == 1 and c2d["fill_holes_size"] == 0
    assert slam["subscribe_rgbd"] is True and slam["subscribe_rgb"] is False
    assert "Mem/StereoFromMotion" not in slam and slam["Grid/Sensor"] == "1"
    assert slam["Vis/MaxDepth"] == vo["Vis/MaxDepth"]
    assert vo["publish_tf"] is False  # odom_selector is the only odom->base_link publisher
    assert sync["approx_sync"] is False  # DA3 depth carries the source image stamp
    assert not (_PKG / "config" / "rtabmap_mono.yaml").exists()


def test_k7_cli_exit_codes(tmp_path: Path, capsys) -> None:
    dump = tmp_path / "dump.txt"
    dump.write_text(_DUMP + 'Param: Mem/InitWMWithAllNodes = "false"   [...]\n', encoding="utf-8")
    cfg = tmp_path / "r.yaml"
    cfg.write_text("/**:\n  ros__parameters:\n    Kp/MaxFeatures: '500'\n", encoding="utf-8")
    assert check_main(["--dump", str(dump), "--config", str(cfg)]) == 0
    cfg.write_text("/**:\n  ros__parameters:\n    Made/Up: '1'\n", encoding="utf-8")
    assert check_main(["--dump", str(dump), "--config", str(cfg)]) == 1
    assert "Made/Up" in capsys.readouterr().out


def test_k8_cli_checks_mode_params_too(tmp_path: Path) -> None:
    dump = tmp_path / "dump.txt"
    dump.write_text(_DUMP, encoding="utf-8")  # lacks Mem/InitWMWithAllNodes
    cfg = tmp_path / "r.yaml"
    cfg.write_text("/**:\n  ros__parameters:\n    Kp/MaxFeatures: '500'\n", encoding="utf-8")
    assert check_main(["--dump", str(dump), "--config", str(cfg)]) == 1


def test_r1_drift_report_cli(tmp_path: Path) -> None:
    rows = "\n".join(f"{(i + 1) * 100_000_000},{i * 0.5},0.0,0.0" for i in range(40))
    est = tmp_path / "est.csv"
    gt = tmp_path / "gt.csv"
    est.write_text("t_ns,x,y,z\n" + rows + "\n", encoding="utf-8")
    gt.write_text("t_ns,x,y,z\n" + rows + "\n", encoding="utf-8")
    out = tmp_path / "report.json"
    assert drift_main(["--est", str(est), "--gt", str(gt), "--out", str(out), "--segments", "5"]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["matched"] == 40
    assert rep["ate_rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert rep["gt_path_length_m"] == pytest.approx(19.5)


_NODE_LIST = """\
/rtabmap/rtabmap:
  approx_sync
  frame_id
  subscribe_rgbd
  Kp/MaxFeatures
"""


def test_k9_parse_node_param_list() -> None:
    assert parse_node_param_list(_NODE_LIST) == {"approx_sync", "frame_id", "subscribe_rgbd"}
    with pytest.raises(ValueError, match="node parameters"):
        parse_node_param_list("/rtabmap/rtabmap:\n")  # node header only (node not running)


def test_k10_cli_node_check(tmp_path: Path, capsys) -> None:
    dump = tmp_path / "dump.txt"
    dump.write_text(_DUMP + 'Param: Mem/InitWMWithAllNodes = "false"   [...]\n', encoding="utf-8")
    nodes = tmp_path / "nodes.txt"
    nodes.write_text(_NODE_LIST, encoding="utf-8")
    cfg = tmp_path / "r.yaml"
    cfg.write_text("/**:\n  ros__parameters:\n    frame_id: base_link\n    Kp/MaxFeatures: '500'\n", encoding="utf-8")
    assert config_node_keys(cfg) == {"frame_id"}
    args = ["--dump", str(dump), "--config", str(cfg), "--node-check", f"{cfg}={nodes}"]
    assert check_main(args) == 0
    cfg.write_text("/**:\n  ros__parameters:\n    queue_sizee: 5\n", encoding="utf-8")
    assert check_main(args) == 1
    assert "queue_sizee" in capsys.readouterr().out
