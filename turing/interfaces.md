# Internal interfaces

ROS is the process boundary. Inside Dev 1, modules talk through these types so each task is replaceable without rewriting the others.

Implementation language: Python 3 for v1 (fast to swap adapters). ROS 2 msgs wrap the same structs at the process edge (T07).

## Types

```python
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

CANONICAL = {0, 1, 2}
UNKNOWN, TRAVERSABLE, HAZARD = 0, 1, 2

@dataclass(frozen=True)
class ImageFrame:
    """One real camera frame. Never synthesized."""
    rgb: NDArray[np.uint8]          # H,W,3  BGR or RGB, documented per source
    stamp_ns: int                   # image time, not receive time
    frame_id: str                   # optical frame
    camera_info: dict               # CameraInfo fields; required to publish the port
    encoding: str

@dataclass(frozen=True)
class RawSemOutput:
    """Adapter language. Must not leak past remap."""
    adapter_id: str
    # Either dense labels (H,W) of model-class ids, or instance maps reduced to dense by the adapter.
    label_ids: NDArray[np.int32]    # H,W
    raw_scores: NDArray[np.float32] # H,W  model-native, NOT [0,1] yet
    id_to_name: dict[int, str]      # Python int keys; T06 converts np.int64 before T03
    stamp_ns: int                   # MUST equal source ImageFrame.stamp_ns
    frame_id: str                   # MUST equal source frame_id
    hw: tuple[int, int]             # mask resolution; default = source

@dataclass(frozen=True, slots=True)
class CanonicalMask:
    """T01 kernel. Authority: subarchs/subarch1.md — not this dump."""
    header: object                   # FrameHeader {stamp_ns, frame_id}
    classes: NDArray[np.uint8]       # H,W  {0,1,2}
    confidence: NDArray[np.float32]  # H,W  [0,1]
    valid: bool                      # producer_ok ∧ structural; not “arrays looked legal”
    age_s: float                     # informational only; freshness = now vs header.stamp
    scale: float                     # v1 product: 1.0


@dataclass(frozen=True)
class DepthFrame:
    depth: NDArray[np.float32]      # H,W meters or relative; unit in `unit`
    unit: str                       # "meter" | "relative"
    stamp_ns: int
    frame_id: str
```

`CanonicalMask.classes` that contains any value outside `{0,1,2}` is a **bug**. Validators (T01) raise; the port node must not publish it.

## Protocols

```python
class Source:
    def open(self) -> None: ...
    def read(self) -> ImageFrame: ...          # blocks; real frame or raises
    def close(self) -> None: ...

class InferenceBackend:
    """T12 — engine only. Intel Arc: OpenVINO GPU. Else: CUDA PyTorch.
    T07 must not import this. See HARDWARE.md (runtime only, not the port)."""
    id: str   # "openvino_gpu" | "cuda_pytorch"
    def load(self, weights_path: str, **engine_args) -> None: ...
    def run(self, rgb: NDArray[np.uint8]) -> object: ...  # BackendTensors; adapter-private

class Adapter:
    id: str
    def infer(self, frame: ImageFrame) -> RawSemOutput: ...

class Remapper:
    def apply(self, raw: RawSemOutput) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """Returns (class_hw in {0,1,2} before gates, raw_scores copied).
        Raises if adapter_id has no YAML."""

class ConfidenceGate:
    def apply(self, classes: NDArray[np.uint8], raw_scores: NDArray[np.float32], adapter_id: str, runner_up: NDArray[np.float32] | None = None) -> tuple[NDArray[np.uint8], NDArray[np.float32], float, bool]:
        """Normalize → [0,1] (no per-frame minmax). Gates. Returns
        (classes, conf, known_fraction, collapse_candidate).
        known = classes != 0 after all gates including kappa.
        T04 does not set degraded; T05/T07 do."""

class Freshness:
    def evaluate(self, stamp_ns: int, now_ns: int) -> tuple[float, bool, bool]:
        """Returns (age_s, is_fresh, degraded)."""

class DepthAdapter:
    def infer(self, frame: ImageFrame) -> DepthFrame: ...
```

## Config files (owned by Dev 1)

```
config/cameras/<name>.yaml          # T02  — real intrinsics
config/ontologies/<adapter>.yaml    # T03  — name → {0,1,2}
config/perception/<adapter>.yaml    # T04  — τ_trav, τ_haz, τ_min, κ, normalizer
config/perception/port.yaml         # T05  — perception_max_age, publish topics
```

No remap YAML for an adapter → that adapter is illegal to instantiate.

## Pipeline (pure, testable without ROS)

```python
def to_canonical(raw, remapper, gate, freshness, now_ns) -> CanonicalMask:
    classes, scores = remapper.apply(raw)
    classes, conf = gate.apply(classes, scores, raw.adapter_id)
    age_s, is_fresh, degraded = freshness.evaluate(raw.stamp_ns, now_ns)
    valid = is_fresh and not degraded and conf_ok(conf)
    if not valid:
        # Do not rewrite history: classes may still exist, but valid=False
        # and the node will set perception_degraded.
        pass
    return CanonicalMask(...)
```

Stamp reuse is illegal: `raw.stamp_ns == frame.stamp_ns` is checked in T07.
`frame_id` reuse is illegal the same way: mask optical frame equals source optical frame.

T07 composition has no `if adapter_id == ...` and no backend imports. Backend choice stays inside the adapter (T12).

## Efficiency rules

- One `Source.read()` per tick. YOLOE and Depth Anything share the same `ImageFrame`.
- Remap + gates are in-place numpy. No GPU round-trip.
- Adapter owns GPU. Port node does not copy RGB unless an adapter requires a contiguous layout.
- Mask resolution defaults to source resolution. Downscale only via explicit config, and then `port_meta` records the scale (architecture §8.5).
- If VRAM cannot hold YOLOE-seg + Depth Anything together, run them **sequential on the same frame**, not on alternate frames (alternate frames = silent stamp mismatch).
