import math
import xml.etree.ElementTree as ET

import pytest

from ugv_robot_description.urdf import MountError, camera_mount, robot_urdf


def test_mount_requires_every_value():
    with pytest.raises(MountError, match="camera_z is required"):
        camera_mount("0.1", "0", "", "15")


@pytest.mark.parametrize("z,p", [("0", "15"), ("-0.2", "15"), ("0.5", "90"), ("0.5", "nan")])
def test_mount_refuses_impossible_values(z, p):
    with pytest.raises(MountError):
        camera_mount("0", "0", z, p)


def test_urdf_chains_base_to_optical_frame():
    x, y, z, pitch = camera_mount("0.2", "-0.05", "0.45", "15")
    root = ET.fromstring(robot_urdf(x, y, z, pitch))
    joints = {j.get("name"): j for j in root.iter("joint")}
    mount = joints["camera_mount"].find("origin")
    assert [float(v) for v in mount.get("xyz").split()] == pytest.approx([0.2, -0.05, 0.45])
    assert float(mount.get("rpy").split()[1]) == pytest.approx(math.radians(15), abs=1e-6)
    assert joints["camera_optical"].find("child").get("link") == "camera_optical_frame"
