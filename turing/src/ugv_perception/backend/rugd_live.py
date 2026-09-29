"""Wire the live RUGD adapter to Intel OpenVINO or NVIDIA CUDA."""

from __future__ import annotations

from pathlib import Path

from ugv_perception.adapter.output import AdapterError
from ugv_perception.adapter.rugd import RugdSegformerAdapter, load_rugd_config
from ugv_perception.backend.device import pick_tensor_backend


def build_live_adapter(root: Path) -> RugdSegformerAdapter:
    cfg = load_rugd_config(root / "config" / "adapters" / "rugd.yaml")
    xml = root / str(cfg["weights"])
    backend = pick_tensor_backend(
        ir_xml=xml,
        safetensors_dir=xml.with_suffix(""),
        kind="rugd",
    )
    if backend is None:
        raise AdapterError("RUGD weights missing (OpenVINO IR or safetensors)")
    hw = cfg["input_hw"]
    mean = cfg["mean"]
    std = cfg["std"]
    if type(hw) is not tuple or type(mean) is not tuple or type(std) is not tuple:
        raise TypeError("rugd config fields have the wrong type")
    return RugdSegformerAdapter(backend, input_hw=hw, mean=mean, std=std)
