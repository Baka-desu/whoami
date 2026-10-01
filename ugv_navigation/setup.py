import os
from glob import glob

from setuptools import setup

# Every path below is relative to this package; make that true whatever directory the build starts from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

package_name = "ugv_costmap"

setup(
    name=package_name,
    version="0.1.0",
    packages=["costmap_core", "ugv_costmap"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev3",
    maintainer_email="dev@neogenmedia.com",
    description="Dev 3 costmap core and its ROS 2 semantic-grid adapter.",
    license="Proprietary",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "semantic_costmap_node = ugv_costmap.semantic_costmap_node:main",
        ],
    },
)
