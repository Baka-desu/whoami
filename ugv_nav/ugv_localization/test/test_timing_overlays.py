"""Laptop timing profiles are overlays on the product profiles (PR #40 review: no duplicated YAML). No ROS."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from ugv_localization.common import load_yaml_mapping, merge_overlay
from ugv_localization.odom import load_odom_select_config
from ugv_localization.validity import load_validity_profile

_CFG = Path(__file__).resolve().parents[1] / "config"


def test_overlay_overrides_nested_keys_and_leaves_the_rest() -> None:
    base = {"a": 1.0, "b": {"x": 1.0, "y": 2.0}}
    out = merge_overlay(base, {"b": {"y": 5.0}}, where="o")
    assert out == {"a": 1.0, "b": {"x": 1.0, "y": 5.0}}
    assert base["b"]["y"] == 2.0, "the base mapping is not modified"


def test_overlay_refuses_a_key_the_base_does_not_have() -> None:
    with pytest.raises(KeyError, match="typo_s"):
        merge_overlay({"a": 1.0}, {"typo_s": 2.0}, where="o")
    with pytest.raises(KeyError, match="zz"):
        merge_overlay({"b": {"x": 1.0}}, {"b": {"zz": 2.0}}, where="o")
    with pytest.raises(ValueError, match="mapping"):
        merge_overlay({"b": {"x": 1.0}}, {"b": 3.0}, where="o")


def test_laptop_pose_validity_overlay_raises_only_the_age_limits() -> None:
    base = asdict(load_validity_profile(_CFG / "pose_validity.yaml"))
    laptop = asdict(load_validity_profile(_CFG / "pose_validity.yaml", _CFG / "pose_validity_laptop.yaml"))
    changed = {k: (base[k], laptop[k]) for k in base if base[k] != laptop[k]}
    assert changed == {
        "localization_max_age_s": (0.5, 1.5),
        "odom_max_age_s": (0.5, 1.5),
        "depth_max_age_s": (0.5, 1.5),
        "slam_max_age_s": (2.0, 4.0),
    }


def test_laptop_odom_select_overlay_raises_only_the_visual_timeout() -> None:
    base = load_odom_select_config(_CFG / "odom_select.yaml")
    laptop = load_odom_select_config(_CFG / "odom_select.yaml", _CFG / "odom_select_laptop.yaml")
    assert laptop.wheel_gate == base.wheel_gate and laptop.visual_gate == base.visual_gate
    b, l = asdict(base.selector), asdict(laptop.selector)
    assert {k: (b[k], l[k]) for k in b if b[k] != l[k]} == {"visual_timeout_s": (0.5, 1.5)}


def test_laptop_rgbd_sync_overlay_holds_only_known_keys() -> None:
    """rgbd_sync reads ROS parameters, so launch_ros layers [base, overlay]; the overlay must only touch base keys."""
    base = yaml.safe_load((_CFG / "rgbd_sync.yaml").read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    over = load_yaml_mapping(_CFG / "rgbd_sync_laptop.yaml")["/**"]["ros__parameters"]
    assert set(over) <= set(base)
    merged = {**base, **over}
    assert {k: (base[k], merged[k]) for k in base if base[k] != merged[k]} == {
        "topic_queue_size": (10, 30),
        "qos": (0, 1),
        "qos_camera_info": (2, 1),
    }
