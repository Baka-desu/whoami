from glob import glob

from setuptools import find_packages, setup

package_name = "ugv_api"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jeswin-christie",
    maintainer_email="dev@neogenmedia.com",
    description="Dev 5 operator gateway: REST /api/v1 + SSE over the operator items of architecture.md.",
    license="Proprietary",
    extras_require={"test": ["pytest"]},  # colcon picks pytest from here (tests_require is gone)
    entry_points={
        "console_scripts": [
            "api_gateway = ugv_api.main:main",
        ],
    },
)
