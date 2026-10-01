import os
from glob import glob

from setuptools import find_packages, setup

# Every path below is relative to this package; make that true whatever directory the build starts from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

package_name = "ugv_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev5",
    maintainer_email="dev@neogenmedia.com",
    description="Dev 5 bringup: the camera driver (Image + CameraInfo for Dev 1 / Dev 2, compressed stream for the UI).",
    license="Proprietary",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "camera_driver = ugv_bringup.nodes.camera_driver:main",
        ],
    },
)
