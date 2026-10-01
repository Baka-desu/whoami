from glob import glob

from setuptools import find_packages, setup

package_name = "ugv_safety"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        # Architecture §7: safety config lives in ugv_nav/config/safety/ (Dev 5 owned).
        (f"share/{package_name}/config/safety", glob("../config/safety/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev5",
    maintainer_email="dev@neogenmedia.com",
    description="Dev 5 safety authority: the sole publisher of the base /cmd_vel (architecture.md §3.1).",
    license="Proprietary",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "safety_arbiter = ugv_safety.nodes.safety_arbiter:main",
        ],
    },
)
