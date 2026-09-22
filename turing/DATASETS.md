# Datasets, weights, and what you must provide

v1 **does not train**. Architecture deferred “learned adapter conf calibration tooling”; gates are manual YAML.

There is also **no dummy data**. That means we cannot invent a camera. Something real has to exist before T02/T06 can be marked done.

## Required for the product path (not training)

### 1. Camera + calibration — **you provide**

- Calibrated vision sensor: stereo or RGB-D recommended, mono acceptable as minimum.
- `CameraInfo` / calibration YAML for that camera (intrinsics, distortion, optical `frame_id`).
- For `live_cam`: working device (V4L2 / ROS camera driver).
- For eval: outdoor recordings from **that** camera (rosbag or video + sidecar calibration). Do not substitute random internet clips.

**Now:** no camera **device**. T02 **decode + ROS subscribe** shipped (fixture msgs). Outdoor live still Dev 5. Do not invent a camera.

T11 and outdoor product proof need a real stream. T06 GPU `infer` needs IR (below).

### 2. YOLOE pretrained segmentation weights — **download, no training**

T06 uses Ultralytics YOLOE as a **promptable segmenter**, outdoor default adapter.

- Pin: **`yoloe-26s-seg`** → OpenVINO IR `weights/yoloe-26s-seg.xml` (see `config/adapters/yoloe.yaml`). 26m only if 26s is weak.
- We do **not** train YOLOE on this project unless outdoor quality is later judged insufficient.
- You must allow a one-time download (or drop the `.pt` file into `turing/weights/`).

### 3. Ontology we author — **not a dataset**

YOLOE is open-vocabulary. The “dataset” for classes is a **prompt list + remap YAML** we write:

- Prompts: dirt/gravel path, grass, vegetation, sky, person, vehicle, rock, water, fence, tree, etc.
- Remap: each prompt → `{0,1,2}`.

That YAML is product config (`config/ontologies/yoloe.yaml`). It is not training data.

## Required only if T08 is in scope (optional geometry)

### 4. Depth Anything V2 pretrained — **download, no training**

- Official Depth Anything V2 checkpoint (indoor/outdoor relative or metric variant, pinned at implement time).
- Same camera frames as T02. No synthetic depth.

T08 can be skipped for a mask-only v1. Architecture marks it optional. Semantic-only navigation is weaker outdoors (low vegetation, ruts); say if you want T08 in the first ship.

## Not required now

| Item | Why |
|---|---|
| RUGD | Eval profile only. Useful later to score outdoor prompts. Real photos, not synthetic — allowed **if you want it**. |
| ORFD / Yamaha-CMU / Off-Road | Fine-tune sets. Only if YOLOE+prompts fail outdoors. |
| Tutorial ONNX weights | T09 eval scaffold. Not product. Pull only when we build T09. |
| Any simulator dataset | Forbidden as Dev 1 product evidence. |
| Labeling / training GPU job | Not in v1. |

## When I would ask you to train

Only after T06+T07 run on **your outdoor camera** and the port is systematically wrong (path as hazard, rocks as traversable) in a way YAML prompts/gates cannot fix.

Then we would need:

- A labeled outdoor set in the same domain (your frames labeled `{0,1,2}`, or RUGD remapped to `{0,1,2}`).
- A fine-tune of YOLOE-seg, still publishing through the same remap/conf/port.

Until that failure is observed, **do not collect a training set**.

## Action list for you

1. Camera / outdoor recording — Dev 5 stream; T02 consume path shipped.
2. Download **YOLOE-26s-seg** and export OpenVINO 2026.4.0 GPU IR to `turing/weights/` (not in git).
3. Say whether T08 (Depth Anything) is in the first ship.
4. Optional: if you already have RUGD locally, tell me the path — eval only, not now.

I will not generate images, use Gazebo as a stand-in for outdoor perception, or ship a stub adapter to “unblock” navigation.
