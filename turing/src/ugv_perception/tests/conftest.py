"""ROS Lyrical on sys.path only when that install exists."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROS = Path("/opt/ros/lyrical")
_ROS_PY = _ROS / "lib" / "python3.12" / "site-packages"
if _ROS.is_dir() and _ROS_PY.is_dir():
    if str(_ROS_PY) not in sys.path:
        sys.path.insert(0, str(_ROS_PY))
    os.environ.setdefault("AMENT_PREFIX_PATH", str(_ROS))
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    _lib = f"{_ROS / 'lib64'}:{_ROS / 'lib'}"
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(_ROS / "lib") not in _ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = _lib + (":" + _ld if _ld else "")
    os.environ.setdefault("ROS_DOMAIN_ID", "91")
