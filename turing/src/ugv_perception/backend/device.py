"""Pick Intel OpenVINO GPU, CUDA PyTorch, or OpenVINO CPU. Lazy vendor imports."""

from __future__ import annotations

from pathlib import Path


def intel_openvino_gpu_available() -> bool:
    try:
        import openvino as ov
    except ImportError:
        return False
    try:
        devices = [str(d) for d in ov.Core().available_devices]
    except Exception:
        return False
    return any(item == "GPU" or item.startswith("GPU.") for item in devices)


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_openvino_ir(xml: Path, *, fp16: bool = False) -> object:
    from ugv_perception.backend.openvino_gpu import OpenVinoGpuTensorBackend

    backend = OpenVinoGpuTensorBackend()
    backend.fp16 = fp16
    backend.load(str(xml), input_hw=None)
    return backend


def _load_cuda_safetensors(folder: Path, kind: str, *, fp16: bool = False) -> object:
    from ugv_perception.backend.cuda_pytorch import CudaPytorchTensorBackend

    backend = CudaPytorchTensorBackend()
    backend.da3_half = fp16
    backend.load(str(folder), kind=kind)
    return backend


def pick_tensor_backend(
    *,
    ir_xml: Path,
    safetensors_dir: Path,
    kind: str,
    fp16: bool = False,
) -> object | None:
    """Intel GPU IR, then CUDA safetensors, then OpenVINO CPU IR. None if nothing fits.

    Precision is FP32 on every backend. `fp16` is an explicit opt-in for DA3 only, meant for a target where FP32
    was measured to miss the latency budget; RUGD segmentation never runs in FP16.
    """
    if fp16 and kind != "da3":
        raise ValueError("fp16 is a DA3-only opt-in; RUGD always runs FP32")
    xml = Path(ir_xml)
    folder = Path(safetensors_dir)
    if intel_openvino_gpu_available() and xml.is_file():
        return _load_openvino_ir(xml, fp16=fp16)
    if cuda_available() and (folder / "model.safetensors").is_file():
        return _load_cuda_safetensors(folder, kind, fp16=fp16)
    if xml.is_file():
        return _load_openvino_ir(xml, fp16=fp16)
    return None
