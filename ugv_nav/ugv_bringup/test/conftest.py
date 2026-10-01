"""Make the package (and Dev 2's calibration loader it depends on) importable outside colcon."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
for root in (_HERE, _HERE.parent / "ugv_localization"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
