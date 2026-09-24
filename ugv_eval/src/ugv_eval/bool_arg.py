"""Pure --set true/false parsing for CLI tools. No rclpy import, so it is
testable on machines without a ROS install (see tests/test_bool_arg.py)."""

from __future__ import annotations

import argparse

_TRUE_WORDS = {"true", "1", "on", "yes"}
_FALSE_WORDS = {"false", "0", "off", "no"}


def parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in _TRUE_WORDS:
        return True
    if lowered in _FALSE_WORDS:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")
