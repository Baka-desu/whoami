"""ImageFrame DTO. T02 will fill this later. Not a camera driver."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ImageFrame:
    rgb: NDArray[np.uint8]
    stamp_ns: int
    frame_id: str
