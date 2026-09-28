"""Load the tutorial ONNX IR. Missing file fails; does not fall back to RUGD."""

from __future__ import annotations

from pathlib import Path


def build_onnx_adapter(root: Path):
    xml = root / "weights" / "onnx-tutorial.xml"
    if not xml.is_file():
        raise FileNotFoundError(
            f"tutorial ONNX IR missing: {xml}. adapter:=onnx does not fall back to rugd"
        )
    raise FileNotFoundError(
        "tutorial ONNX IR has no recorded input/mean/std yet; inspect the artifact first"
    )
