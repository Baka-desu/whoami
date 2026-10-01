"""CUDA PyTorch backends. Tensor path is live RUGD/DA3 on NVIDIA. YOLOE stays a stub."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ugv_perception.adapter.output import AdapterError, Instance


class CudaPytorchBackend:
    id = "cuda_pytorch"

    def load(self, weights_path: str, **engine_args: object) -> None:
        raise AdapterError("cuda_pytorch YOLOE is not wired; live path is RUGD")

    def run(self, rgb: NDArray[np.uint8]) -> tuple[Instance, ...]:
        raise AdapterError("cuda_pytorch YOLOE is not wired; live path is RUGD")


class CudaPytorchTensorBackend:
    """SegFormer logits or DA3 depth_raw+sky on CUDA. Same blob contract as OpenVINO."""

    id = "cuda_pytorch"

    def __init__(self) -> None:
        self._model = None
        self._kind: str | None = None
        self._hw: tuple[int, int] | None = None
        self._fallback_logits: np.ndarray | None = None
        self.seg_post_disabled = False
        self.device = "cuda"

    def load(self, weights_path: str, kind: str = "rugd") -> None:
        path = Path(weights_path)
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"safetensors missing: {path / 'model.safetensors'}")
        if kind not in ("rugd", "da3"):
            raise ValueError("kind must be rugd or da3")
        try:
            import torch
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        if not torch.cuda.is_available():
            raise AdapterError("CUDA is not available")
        try:
            if kind == "rugd":
                self._model = _load_rugd(path)
            else:
                self._model = _load_da3(path)
        except AdapterError:
            raise
        except ImportError as exc:
            raise AdapterError("CUDA backend dependency missing") from exc
        except Exception as exc:
            raise AdapterError("CUDA load failed") from exc
        self._kind = kind
        self.device = "cuda"

    def ensure_hw(self, height: int, width: int) -> None:
        if self._model is None:
            raise AdapterError("CudaPytorchTensorBackend.load() was not called")
        self._hw = (int(height), int(width))

    def run(self, blob: NDArray[np.float32]) -> np.ndarray:
        if self._fallback_logits is not None:
            out = self._fallback_logits
            self._fallback_logits = None
            return out
        return self.run_all(blob)[0]

    def run_seg(
        self, blob: NDArray[np.float32], out_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate + softmax on CUDA. Labels/scores copied out once."""
        if self._model is None or self._kind != "rugd":
            raise AdapterError("CUDA seg decode requires a loaded RUGD net")
        if not isinstance(blob, np.ndarray) or blob.dtype != np.float32 or blob.ndim != 4:
            raise TypeError("blob must be float32 NCHW")
        oh, ow = int(out_hw[0]), int(out_hw[1])
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        logits = None
        try:
            tensor = torch.from_numpy(blob).to("cuda")
            with torch.inference_mode():
                logits = self._model(pixel_values=tensor).logits
                if not torch.isfinite(logits).all():
                    raise AdapterError("RUGD logits are not finite")
                up = F.interpolate(
                    logits, size=(oh, ow), mode="bilinear", align_corners=False
                )
                prob = torch.softmax(up, dim=1)
                labels = torch.argmax(prob, dim=1)
                scores = torch.gather(prob, 1, labels.unsqueeze(1)).squeeze(1)
                return (
                    labels[0].detach().cpu().numpy().astype(np.int32, copy=False),
                    scores[0].detach().cpu().numpy().astype(np.float32, copy=False),
                )
        except AdapterError as exc:
            if "not finite" in str(exc):
                raise
            self.seg_post_disabled = True
            raise
        except Exception as exc:
            self.seg_post_disabled = True
            if logits is not None:
                self._fallback_logits = logits.detach().cpu().numpy()
            raise AdapterError("CUDA seg decode failed") from exc

    def run_all(self, blob: NDArray[np.float32]) -> list[np.ndarray]:
        if self._model is None or self._kind is None:
            raise AdapterError("CudaPytorchTensorBackend.load() was not called")
        if not isinstance(blob, np.ndarray) or blob.dtype != np.float32 or blob.ndim != 4:
            raise TypeError("blob must be float32 NCHW")
        try:
            import torch
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        try:
            tensor = torch.from_numpy(blob).to("cuda")
            with torch.inference_mode():
                if self._kind == "rugd":
                    logits = self._model(pixel_values=tensor).logits
                    return [logits.detach().cpu().numpy()]
                depth, sky = self._model(tensor)
                return [
                    depth.detach().cpu().numpy(),
                    sky.detach().cpu().numpy(),
                ]
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("CUDA run failed") from exc


    def run_decoded(
        self, blob: NDArray[np.float32], out_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """RUGD on CUDA, decoded on the GPU: (labels int32, top-class probability float32) at out_hw.

        Same maths as adapter.rugd.decode_rugd_logits, in float64: bilinear resize of the logits with
        half-pixel centres and edge clamp (torch align_corners=False), argmax, and the top class's
        softmax probability 1 / sum(exp(l - l_max)). Saves ~0.7 s of CPU per 640x480 frame.
        """
        if self._model is None or self._kind != "rugd":
            raise AdapterError("run_decoded needs a loaded RUGD model")
        if not isinstance(blob, np.ndarray) or blob.dtype != np.float32 or blob.ndim != 4:
            raise TypeError("blob must be float32 NCHW")
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        try:
            tensor = torch.from_numpy(blob).to("cuda")
            with torch.inference_mode():
                logits = self._model(pixel_values=tensor).logits
                if not bool(torch.isfinite(logits).all()):
                    raise AdapterError("RUGD logits are not finite")
                a = logits.double()
                if tuple(a.shape[-2:]) != tuple(out_hw):
                    a = F.interpolate(a, size=tuple(out_hw), mode="bilinear", align_corners=False)
                a = a[0]
                top, labels = a.max(dim=0)
                total = torch.exp(torch.clamp(a - top, -80.0, 80.0)).sum(dim=0)
                scores = (1.0 / total).float()
                return (
                    labels.to(torch.int32).cpu().numpy(),
                    scores.cpu().numpy(),
                )
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("CUDA run failed") from exc


def _load_rugd(path: Path) -> object:
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        str(path), local_files_only=True
    )
    model.eval()
    return model.to("cuda")


def _load_da3(path: Path) -> object:
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from safetensors.torch import load_file

    net = create_object(load_config(MODEL_REGISTRY["da3metric-large"]))
    state = load_file(str(path / "model.safetensors"))
    if any(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise AdapterError(
            "DA3 checkpoint does not match the net "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )
    net.eval()
    return _Da3Head(net).to("cuda")


class _Da3Head:
    """depth_raw and sky from the depth head, before sky-fill. Matches the IR export."""

    def __init__(self, net: object) -> None:
        self.net = net

    def __call__(self, pixel_values: object) -> tuple[object, object]:
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

    def to(self, device: str) -> "_Da3Head":
        self.net = self.net.to(device)
        return self

    def eval(self) -> "_Da3Head":
        self.net.eval()
        return self
