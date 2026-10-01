import os
from glob import glob

from setuptools import find_packages, setup

os.chdir(os.path.dirname(os.path.abspath(__file__)))

package_name = "ugv_robot_description"

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
    description="Dev 5 robot description: camera mount TF for the costmap and RTAB-Map.",
    license="Proprietary",
    extras_require={"test": ["pytest"]},
)
