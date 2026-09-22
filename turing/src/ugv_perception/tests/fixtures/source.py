"""Allocated RGB ImageFrame for T10. Not a camera. Not outdoor proof."""

from __future__ import annotations

import numpy as np

from ugv_perception.adapter.frame import ImageFrame

_DEFAULT_FRAME_ID = "camera_optical"


class FixtureSource:
    """Caller-set stamp/frame. RGB exists for dtype/shape only."""

    def frame(
        self,
        *,
        stamp_ns: int,
        frame_id: str = _DEFAULT_FRAME_ID,
        hw: tuple[int, int] = (2, 2),
    ) -> ImageFrame:
        if type(stamp_ns) is not int or stamp_ns <= 0:
            raise TypeError("stamp_ns must be a Python int > 0")
        if type(frame_id) is not str or frame_id == "":
            raise ValueError("frame_id must be a non-empty str")
        h, w = hw
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return ImageFrame(rgb=rgb, stamp_ns=stamp_ns, frame_id=frame_id)
