"""YoloeAdapter: backend.run + pack. Does not implement OpenVINO (T12)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import yaml

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import ADAPTER_ID, AdapterError, Instance, RawSemOutput
from ugv_perception.adapter.pack import pack


class InferenceBackend(Protocol):
    id: str

    def load(self, weights_path: str, **engine_args: object) -> None: ...

    def run(self, rgb: NDArray[np.uint8]) -> tuple[Instance, ...]: ...


def load_adapter_config(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adapter YAML missing: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if type(data) is not dict:
        raise TypeError("adapter YAML root must be a mapping")
    adapter_id = data.get("adapter_id")
    backend = data.get("backend")
    weights = data.get("weights")
    if adapter_id != ADAPTER_ID:
        raise ValueError("adapter_id must be yoloe")
    if type(backend) is not str or backend == "":
        raise ValueError("backend must be a non-empty str")
    if type(weights) is not str or weights == "":
        raise ValueError("weights must be a local path str")
    if path.stem != ADAPTER_ID:
        raise ValueError("repo convention: adapters/yoloe.yaml stem must be yoloe")
    return {"adapter_id": adapter_id, "backend": backend, "weights": weights}


class YoloeAdapter:
    def __init__(self, backend: InferenceBackend, prompts: tuple[str, ...]) -> None:
        if type(prompts) is not tuple or len(prompts) == 0:
            raise TypeError("prompts must be a non-empty tuple")
        self.backend = backend
        self.prompts = prompts

    def infer(self, frame: ImageFrame) -> RawSemOutput:
        try:
            instances = self.backend.run(frame.rgb)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("YOLOE backend failed") from exc
        return pack(frame, instances, self.prompts)
