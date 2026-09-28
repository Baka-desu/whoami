#!/usr/bin/env python3
"""Export DA3METRIC-LARGE head tensors to OpenVINO IR.

The wrapper returns depth_raw and sky from the depth head only.
It does not call the stock forward, which fills sky with a far depth.
You run this after fetch_da3metric_large.sh. Tests do not run it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "weights" / "da3metric-large"
IR_XML = ROOT / "weights" / "da3metric-large.xml"


def _export_module(net: object):
    import torch

    class Da3HeadExport(torch.nn.Module):
        def __init__(self, inner: object) -> None:
            super().__init__()
            self.net = inner

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            image = pixel_values.unsqueeze(1)
            feats, _aux = self.net.backbone(
                image,
                cam_token=None,
                export_feat_layers=[],
                ref_view_strategy="saddle_balanced",
            )
            height, width = int(image.shape[-2]), int(image.shape[-1])
            output = self.net._process_depth_head(feats, height, width)
            return output.depth[:, 0], output.sky[:, 0]

    return Da3HeadExport(net)


def _load_net():
    """Build the metric net without importing the API (that pulls gsplat)."""
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    if not (SRC / "model.safetensors").is_file():
        raise SystemExit(f"missing checkpoint: {SRC / 'model.safetensors'}")
    net = create_object(load_config(MODEL_REGISTRY["da3metric-large"]))
    state = load_file(str(SRC / "model.safetensors"))
    if any(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected checkpoint keys: {unexpected[:8]}")
    if missing:
        print(f"warning: {len(missing)} missing keys, first: {missing[:4]}")
    net.eval()
    return net


def export_ir(height: int, width: int) -> None:
    import torch
    import openvino as ov

    if height % 14 != 0 or width % 14 != 0:
        raise SystemExit("height and width must both be multiples of 14")
    if max(height, width) != 504:
        raise SystemExit("longest side must be 504")
    net = _load_net()
    example = torch.zeros(1, 3, height, width, dtype=torch.float32)
    wrapper = _export_module(net)
    with torch.inference_mode():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov_model.reshape([1, 3, height, width])
    IR_XML.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, IR_XML, compress_to_fp16=False)
    print(f"wrote {IR_XML} for {height}x{width}")
    print("outputs are depth_raw then sky, before sky-fill")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=504)
    parser.add_argument("--width", type=int, default=336)
    args = parser.parse_args()
    try:
        export_ir(args.height, args.width)
    except KeyboardInterrupt:
        sys.exit(130)
