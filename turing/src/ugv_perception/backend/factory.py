"""Build a loaded InferenceBackend. Do not import openvino at module level."""

from __future__ import annotations

from ugv_perception.adapter.yoloe import InferenceBackend

_FORBIDDEN = frozenset({"ultralytics", "openvino_xpu", "xpu"})


def build_backend(
    backend_id: str,
    weights_path: str,
    prompts: tuple[str, ...],
) -> InferenceBackend:
    if type(backend_id) is not str or backend_id == "":
        raise TypeError("backend_id must be a non-empty str")
    if type(weights_path) is not str or weights_path == "":
        raise TypeError("weights_path must be a non-empty str")
    if type(prompts) is not tuple or len(prompts) == 0:
        raise TypeError("prompts must be a non-empty tuple")
    if backend_id in _FORBIDDEN:
        raise ValueError(f"backend {backend_id!r} is not a product runtime")
    if backend_id == "cuda_pytorch":
        raise RuntimeError("cuda_pytorch is not available on this Intel Arc box")
    if backend_id == "openvino_gpu":
        from ugv_perception.backend.openvino_gpu import OpenVinoGpuBackend

        backend = OpenVinoGpuBackend()
        backend.load(weights_path, prompts=prompts)
        return backend
    raise ValueError(f"unknown backend_id {backend_id!r}")
