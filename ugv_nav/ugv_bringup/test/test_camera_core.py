"""Camera driver helpers (pure Python, no ROS / OpenCV). The calibration here is a test fixture."""

from __future__ import annotations

import pytest
import yaml

from ugv_bringup.camera_core import CaptureError, RatePacer, camera_info_fields, check_capture_size, parse_device
from ugv_localization.camera import CalibrationError, calibration_to_yaml_dict, load_calibration
from ugv_localization.camera.calib import CameraCalibration


def fixture_calibration(**kw) -> CameraCalibration:
    base = dict(
        camera_name="test_cam", width=640, height=480,
        k=(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0),
        distortion_model="plumb_bob", d=(0.0,) * 5,
        r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        p=(500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    base.update(kw)
    return CameraCalibration(**base)


def test_camera_info_fields_carry_the_calibration_through(tmp_path):
    path = tmp_path / "cam.yaml"
    path.write_text(yaml.safe_dump(calibration_to_yaml_dict(fixture_calibration())), encoding="utf-8")
    f = camera_info_fields(load_calibration(path))
    assert (f.width, f.height, f.distortion_model) == (640, 480, "plumb_bob")
    assert f.k[0] == 500.0 and f.k[2] == 320.0 and len(f.k) == 9 and len(f.p) == 12 and len(f.d) == 5


def test_a_zero_k_calibration_cannot_be_loaded_so_it_can_never_be_published(tmp_path):
    bad = calibration_to_yaml_dict(fixture_calibration())
    bad["camera_matrix"]["data"] = [0.0] * 9
    path = tmp_path / "zero.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(path)


def test_missing_calibration_file_is_refused(tmp_path):
    with pytest.raises(Exception):
        load_calibration(tmp_path / "nope.yaml")


def test_capture_size_must_match_calibration():
    check_capture_size(640, 480, 640, 480)
    with pytest.raises(CaptureError, match="640x480"):
        check_capture_size(640, 480, 1280, 720)


@pytest.mark.parametrize("value,expected", [("0", 0), ("2", 2), ("/dev/video0", "/dev/video0"), (" /dev/video1 ", "/dev/video1")])
def test_parse_device(value, expected):
    assert parse_device(value) == expected


def test_parse_device_rejects_empty():
    with pytest.raises(CaptureError):
        parse_device("  ")


def test_pacer_limits_a_30fps_stream_to_about_5hz():
    pacer = RatePacer(5.0, slack_s=0.5 / 30.0)
    hits = [i for i in range(300) if pacer.due(i / 30.0)]  # 10 s of 30 fps
    assert 48 <= len(hits) <= 52
    assert hits[0] == 0


def test_pacer_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        RatePacer(0.0)
