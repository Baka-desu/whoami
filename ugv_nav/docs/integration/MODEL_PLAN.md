# Model plan: RUGD SegFormer-B5 + Depth Anything 3 Metric Large

**Status:** plan only. No weights are on the integration PC (`turing/weights/` holds only `.gitkeep`), and
nothing was downloaded or run. Sources: `user_manual.md`, `turing/scripts/*`, `turing/HARDWARE.md`,
`turing/src/ugv_perception/backend/*`. Owner: Dev 1 (`dev.md` §4). Written 2026-10-01.

## 1. Why both models block the whole stack

After the 2026-10-01 DA3 amendment (`architecture.md` §6, §10; `dev.md` §3 `/perception/depth_cloud`):

```
camera ─► RUGD SegFormer-B5 ─► /segmentation/mask ─► Dev 3 costmaps ─► Dev 4 Nav2
       └► DA3 Metric Large  ─► /perception/depth_cloud ─► Dev 2 RTAB-Map RGB-D (REQUIRED) ─► TF + /ugv/pose_valid
```

- Without **RUGD**: `adapter_node` refuses to start ("RUGD weights missing"), so there's no mask and
  `/ugv/perception_degraded` has no publisher. The §12 perception watch trips and the robot holds.
- Without **DA3**: Dev 2 gets no depth, so RTAB-Map RGB-D has no input. There's no `map->odom`, so
  `/ugv/pose_valid=false` and Nav2 can't use a pose. The robot holds.

## 2. What to fetch (same files for Intel and NVIDIA)

| Model | HF repo | Lands in | Needed by |
|---|---|---|---|
| SegFormer-B5 (25 RUGD classes → `{0,1,2}`) | `JasonTStanley/RUGD-Segformer` | `turing/weights/rugd-segformer/` (`model.safetensors`, `config.json`) | CUDA path directly; Intel path via export |
| Depth Anything 3 Metric Large | `depth-anything/DA3METRIC-LARGE` (Apache-2.0) | `turing/weights/da3metric-large/` (`model.safetensors`) | CUDA path directly; Intel path via export |

```bash
cd turing
bash scripts/fetch_rugd_segformer.sh      # pip --user huggingface_hub>=0.24, snapshot_download
bash scripts/fetch_da3metric_large.sh
```

Git never stores weights (`turing/.gitignore`). Licence check for the RUGD checkpoint: "see the card"
(`user_manual.md`). Read it before shipping.

## 3. Pick the engine for the target machine

`backend/device.py:52-60` picks in this order: **Intel OpenVINO GPU + `.xml` IR → CUDA + `model.safetensors` →
OpenVINO CPU + IR → none**.

### This integration laptop (probed 2026-10-01, read only)
- Intel Core i7-13620H, 15.6 GB RAM, **Intel UHD Graphics (iGPU)** and **NVIDIA GeForce RTX 4060 Laptop GPU
  (8 GB)**. The RTX shows PnP status `Unknown` and `nvidia-smi` needs admin, so check the driver first.
- **Don't export OpenVINO IR on this machine.** OpenVINO lists the UHD iGPU as `GPU`, so with an `.xml`
  present the picker would run SegFormer-B5 and DA3-Large on the iGPU instead of the RTX. Keep only the
  safetensors folders, so the picker falls through to CUDA.
- ROS 2 Lyrical is Linux-only here, so run the perception node in **WSL2 `Ubuntu-26.04`** with the
  Windows NVIDIA driver's WSL CUDA support (or Docker with `--gpus all`), not native Windows.

### Intel Arc box (Dev 1's B580, `HARDWARE.md`)
```bash
pip install openvino==2026.4.0 torch transformers safetensors   # + depth_anything_3 (see §4)
.venv/bin/python scripts/export_rugd_segformer_openvino.py      # -> weights/rugd-segformer.{xml,bin}, 1x3x640x640
.venv/bin/python scripts/export_da3metric_openvino.py --height <H> --width <W>   # see §5 for H/W
```

### NVIDIA box
Needs the NVIDIA driver, PyTorch with CUDA, `transformers`, `safetensors` and `depth_anything_3`. No export
and no OpenVINO.

## 4. Python dependencies nobody installs yet

`turing/pyproject.toml` declares only `numpy` and `pyyaml`. The runtime and export scripts also import:

| Package | Used by | Path |
|---|---|---|
| `openvino==2026.4.0` | Intel runtime + exports | `backend/openvino_gpu.py`, `scripts/export_*` |
| `torch` (CUDA build on NVIDIA) | CUDA runtime, exports | `backend/cuda_pytorch.py` |
| `transformers` | SegFormer on CUDA, RUGD export | `cuda_pytorch.py:95` |
| `safetensors` | CUDA loaders, DA3 export | `cuda_pytorch.py:105-107` |
| `depth_anything_3` | DA3 on CUDA + DA3 export | `cuda_pytorch.py`, `export_da3metric_openvino.py:45-46`. **No script installs it** |
| `huggingface_hub>=0.24` | fetch scripts | `scripts/fetch_*.sh` |
| `Pillow` | `export_rugd_segformer_openvino.py --run` | |

Action (Dev 1): add these as optional extras in `pyproject.toml` (`[intel]`, `[nvidia]`, `[fetch]`), and pin
the `depth_anything_3` install source (PyPI name or git URL).

## 5. Known risks to check on the first run with weights

1. **DA3 input size vs camera aspect.** `depth/geometry.py:two_step_hw` resizes the long side to 504 and
   both sides to multiples of 14. A 640×480 camera needs **378×504**, but `user_manual.md` exports
   **336×504** (3:2). `openvino_gpu.py:365` reshapes the IR on first compile, so a static IR may still work.
   If it doesn't, the error is **swallowed** by `adapter_node.py:221-222`: no depth or cloud is published
   and nothing is logged. **Check:** `/perception/depth/image` publishes at the mask rate. Export with
   `--height`/`--width` = `model_hw(camera H, W)`.
2. **`config/perception/depth.yaml` is never read.** `enabled: false` is ignored: depth runs whenever its
   weights exist, and the constants are hardcoded in `depth/geometry.py:7-12`.
3. **CameraInfo QoS.** `adapter_node.py:36-43` subscribes TRANSIENT_LOCAL. A volatile driver CameraInfo won't
   match, so the node sees no `K` and fails closed. Dev 5's driver must publish CameraInfo transient-local,
   or Dev 1 must relax the subscription. This is a Dev 1 / Dev 5 decision.
4. **VRAM.** Fit both models on 8 GB; `HARDWARE.md` designs for the smaller NVIDIA card. Record the
   measured peak in `depth.yaml: gpu_peak_bytes` (currently `null`).
5. **Test asset missing.** `thetestimage1.jpg`, used by `test_v1_*`, isn't in the repo.
6. `test_backend_seam.py:310,329` import `openvino` before the weight-skip check, so they error where
   openvino isn't installed instead of skipping.

## 6. Acceptance once weights exist (in order)

1. `cd turing && python -m pytest`: the weight-gated tests in `test_backend_seam.py`, `test_device_pick.py`
   and `test_v1_*.py` run instead of skipping, and pass.
2. Node on a recorded bag (`Image` + `CameraInfo`):
   `python -m ugv_perception.node.adapter_node --ros-args -p adapter:=rugd`
   - `/segmentation/mask` is `mono8`, every pixel in `{0,1,2}`, and the stamp equals the image stamp.
   - `/ugv/perception_degraded` is `false` while fresh, and `true` within `perception_max_age` after the
     bag pauses.
   - `/perception/depth/image` (32FC1, metres) and `/perception/depth_cloud` publish with the image stamp.
3. Dev 2 on the same bag: `ros2 launch ugv_localization bag_eval.launch.py`, then `/ugv/pose_valid` turns
   `true` and `map->odom->base_link` is at ≥ 15 Hz (`tf_rate_check`).
4. Operator gateway: `GET /api/v1/safety/status` shows the camera, perception, localization and TF watches
   `ok` (see `VERIFICATION_2026-10-01.md`).
