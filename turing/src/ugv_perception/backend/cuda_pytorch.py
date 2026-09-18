"""CUDA PyTorch backend. Not runnable on this Arc box."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ugv_perception.adapter.output import AdapterError, Instance


class CudaPytorchBackend:
    id = "cuda_pytorch"

    def load(self, weights_path: str, **engine_args: object) -> None:
        raise AdapterError("cuda_pytorch is not available on this Intel Arc box")

    def run(self, rgb: NDArray[np.uint8]) -> tuple[Instance, ...]:
        raise AdapterError("cuda_pytorch is not available on this Intel Arc box")
