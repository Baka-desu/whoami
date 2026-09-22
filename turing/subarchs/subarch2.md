# Sub-architecture 2 — Consume camera data (T02)

**Task:** [T02](../tasks/T02-ingestion.md)  
**Depends on (code/import):** T01 `stamp_ns` Python `int > 0`; shipped T06 `ImageFrame` (`adapter/frame.py`). **Do not fork a second frame type.**  
**Does not own:** the camera device, V4L2, launch, URDF extrinsics (Dev 5), `config/cameras/` authorship (Dev 2).  
**Does not import:** remap, gates, freshness, `pack`, OpenVINO, `compose_tick`.  
**Status (2026-09-18):** `decode_frame` and `ros_bridge` (sensor_msgs → views) **shipped**. `PerceptionAdapterNode` subscribes. No V4L2. No camera device on this desktop — tests use **message fixtures** and a ROS spin with fixture `Image`+`CameraInfo` (not a dummy camera driver). Outdoor `live_cam` still needs Dev 5 publishing a real stream.  
**ROS on this box:** Lyrical (RHEL 10). `pytest.importorskip` if `sensor_msgs` is missing.  
**Authority:** [`architecture.md`](../../architecture.md) §5 dual fan-out, §8.4, §8.5. Architecture wins over `dev.md` “Dev 1 capture node.”  
**Not authority:** `dev.md` task 1 V4L2/capture; old T02 `LiveCameraSource` opening the device; `HARDWARE.md` “T02 skipped until a camera exists” as a ban on **coding the converter**.

This file is the architecture of T02. The task file is the build checklist. If they disagree, this file wins, then `architecture.md`.

---

## Files this subarch refers to

### Authority (read; do not fork)

| File | Why T02 cares |
|---|---|
| [`architecture.md`](../../architecture.md) §5, §8.4, §8.5 | Shared vision sensor; stamp = image time; `frame_id` + matching `CameraInfo`; no silent stamp reuse |
| [`subarch1.md`](subarch1.md) I4, I5, I9 | `stamp_ns` Python `int > 0`; non-empty `frame_id`; CameraInfo pairing is **frame_id identity** in T01; **T02 owns K/D truth** |
| [`subarch6.md`](subarch6.md) | `ImageFrame` already shipped — T02 **fills** it |
| [`subarch7.md`](subarch7.md) | `compose_tick(frame: ImageFrame \| None)` — T02 does not live inside the tick |
| Dev 5 / Dev 2 split | Dev 5 launches the driver; Dev 2 owns `config/cameras/`; Dev 1 **consumes topics** |

### Existing code T02 binds to (do not modify in T02)

| File | Binding |
|---|---|
| `turing/src/ugv_perception/adapter/frame.py` | `ImageFrame(rgb, stamp_ns, frame_id)` — fill, don’t replace |
| `turing/src/ugv_perception/adapter/pack.py` | rgb uint8 HWC, 3 channels |
| `turing/src/ugv_perception/compose/tick.py` | consumes `ImageFrame`; T02 does not call it |
| `turing/src/ugv_perception/port/validate.py` | `assert_camera_info_pair` is frame_id only; T02 still checks K exists |

### Shipped files

| File | Role |
|---|---|
| `turing/src/ugv_perception/ingest/msgs.py` | ROS-agnostic `ImageView` + `CameraInfoView` |
| `turing/src/ugv_perception/ingest/decode.py` | `decode_frame(image, camera_info) -> ImageFrame` |
| `turing/src/ugv_perception/ingest/ros_bridge.py` | `sensor_msgs` Image/CameraInfo → views (copies `data`) |
| `turing/src/ugv_perception/tests/test_ingest.py` | S1–S15; no rclpy required |
| `turing/src/ugv_perception/tests/test_ros_bridge.py` | ROS msg conversion; `importorskip` without ROS |

Subscribe lives on **T07** `PerceptionAdapterNode` (not a separate `ros_source.py`).

Do **not** add `LiveCameraSource` / V4L2.  
Do **not** add `DummySource`.  
Do **not** edit `adapter/frame.py` unless a field is missing for T06 (it isn’t).  
Do **not** put `CameraInfo` K on `ImageFrame` (T06/T07 don’t need K; T07 ROS can **passthrough** the same `CameraInfo` topic).

---

## When coding (after this subarch is complete)

1. Freeze this file. Implement **`decode_frame` first** (no ROS, no device).  
2. Do **not** edit `pack.py`, `compose/`, `port/`, T06, T12.  
3. Map encodings **only** `rgb8` → RGB copy, `bgr8` → RGB swap. Anything else → raise. Copy out of padded rows (`step >= width*3`); do not alias `Image.data`.  
4. `stamp_ns` from the **image header**, Python `int > 0`. Never `time.now()`, never receive time.  
5. `frame_id` from image header; must equal `CameraInfo.header.frame_id`.  
6. Missing `CameraInfo` → raise. Identity/`zeros` `K` → raise. Do not invent `K`.  
7. Do not undistort, rectify, or run stereo (Dev 2).  
8. Tests: fixture buffers with **known pixel patterns** (encoding/BGR tests). That is not a dummy camera and not outdoor RGB.  
9. Outdoor bag / Dev 5 live camera: still needed for **product** proof, not for kernel tests.  
10. T07 node: subscribe → `ros_bridge` → `decode_frame` → `compose_tick`. T02 **decode** stays ROS-free; `ros_bridge` may import `sensor_msgs`.

---

## 1. Role

The camera is a **shared sensor**. Dev 5 orchestrates it. Dev 1 **consumes its data**.

```
Dev 5 camera driver ──► /camera/image + /camera/camera_info
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     T02 decode_frame                    Dev 2 RTAB-Map
     ImageFrame (rgb, stamp, frame_id)
              ▼
     T06 infer → T03 → T04 → T05 → T07 compose_tick
```

If T02 lies about stamp or `frame_id`, T05 age is wrong and Dev 3 projects into the wrong place. T02’s job is **honest conversion**, not capture.

---

## 2. Scope

### T02 owns

| Piece | Why |
|---|---|
| `decode_frame` | Image + CameraInfo → shipped `ImageFrame` |
| Encoding → uint8 HWC RGB | T06 pack |
| Stamp/frame identity + CameraInfo pair | §8.4, §8.5 |
| Reject missing/fake `K` | T01 left K to T02 |
| ROS subscribe (on T07 node) | consume Dev 5 topics via `ros_bridge` |

### T02 does not own

| Piece | Owner |
|---|---|
| Camera driver, launch, exposure | Dev 5 |
| `config/cameras/` YAML authorship | Dev 2 |
| TF `camera` → `base_link` | Dev 2 |
| `ImageFrame` type | T06 (already shipped) |
| YOLOE / OpenVINO | T06 / T12 |
| Port publish | T07 |
| Undistort / stereo | Dev 2 |

---

## 3. Types

Use **existing** `ImageFrame`. Add ingest views (not ROS msgs in the kernel):

```python
@dataclass(frozen=True, slots=True)
class ImageView:
    stamp_ns: int              # Python int > 0; image header
    frame_id: str              # non-empty
    height: int
    width: int
    encoding: str              # "rgb8" | "bgr8" only
    step: int
    data: bytes                # packed row-major

@dataclass(frozen=True, slots=True)
class CameraInfoView:
    stamp_ns: int              # may differ slightly; pairing is frame_id, not stamp
    frame_id: str
    height: int
    width: int
    k: tuple[float, ...]       # 9 values, row-major 3x3
    # d may be present; T02 does not undistort; must not be used to invent RGB
```

```python
def decode_frame(image: ImageView, camera_info: CameraInfoView) -> ImageFrame: ...
```

No `camera_info=None` overload.

---

## 4. Invariants (fail closed)

| ID | Rule |
|---|---|
| S1 | `type(stamp_ns) is int` and `stamp_ns > 0` on the **image**. Reject `np.int64`, `True`, `0` |
| S2 | Image `frame_id` non-empty `str`; equals `CameraInfoView.frame_id` |
| S3 | `encoding` in `{"rgb8", "bgr8"}` else raise. `rgb8` → RGB; `bgr8` → channel swap to RGB |
| S4 | `rgb.dtype == uint8`, shape `(height, width, 3)`, `height>0`, `width>0` |
| S5 | `type(step) is int`, `step > 0`, `step >= width * 3`, `len(data) >= step * height`. Decode **row padding**: each row uses `data[r*step : r*step + width*3]` only. `ImageFrame.rgb` is packed H×W×3 with **no** stride padding. Do not assume `step == width * 3` |
| S6 | `k` has exactly **9 finite** values; `fx = k[0] > 0`, `fy = k[4] > 0`. Reject all-zero `K`. Reject `K == I_3` (`(1,0,0, 0,1,0, 0,0,1)`) as a reserved placeholder. No other identity clause |
| S7 | `CameraInfoView.height/width` must equal image `height/width` |
| S8 | Do not set `ImageFrame.stamp_ns` from `CameraInfo` stamp or `time.now()` |
| S9 | `decode_frame` does not import `rclpy`, `cv2`, OpenVINO, `pack`, `compose` |
| S10 | No `DummySource`, no V4L2, no `np.zeros` camera class |
| S11 | Decode failure → **raise** (T07 ROS may then pass `frame=None` / degraded). Do not emit a black frame |
| S12 | `bgr8` fixture `[B,G,R]` becomes RGB `[R,G,B]` (tested) |
| S13 | Kernel tests do not require a device or bag |
| S14 | Subscribe path is `ros_bridge`, not V4L2. Fixture `sensor_msgs` tests; no dummy driver |
| S15 | `ImageFrame.rgb` is a **new** uint8 array. It must **not** alias `ImageView.data` (or the ROS `Image.data` buffer). `decode_frame` copies pixel bytes out of each row. Later mutation of the message must not change the frame |

Pairing with T01 I9: after decode, `ImageFrame.frame_id == CameraInfo.frame_id`. T07 `make_mask` may pass CameraInfo later; T02 has already refused a fake K.

---

## 5. ROS wrapper (after Dev 5)

Not this kernel. Subscriber on agreed image + `camera_info` topics → `ImageView`/`CameraInfoView` → `decode_frame` → `compose_tick`.  
If either topic missing this cycle: do not decode a black image; let T07 see `frame=None`.

Bag eval: same `decode_frame` on bag messages. Still not a camera driver.

---

## 6. Aptness vs `architecture.md`

| Clause | Fit |
|---|---|
| §5 dual fan-out, shared sensor | **High** — consume, don’t drive |
| §8.4 stamp = image time | **High** — S1, S8 |
| §8.5 CameraInfo + frame_id | **High** — S2, S6, S7 |
| `dev.md` Dev 1 capture node | **Rejected** — architecture + Dev 5 bringup win |
| Old T02 V4L2 `LiveCameraSource` | **Rejected** |

---

## 7. Tests (`test_ingest.py`)

Message fixtures only. **Every S1–S15.**

| ID | Test |
|---|---|
| S1 | `stamp_ns=0` / `np.int64` raise; **`True` → raise** (`type(True) is not int`) |
| S2 | mismatched `frame_id` raise |
| S3 | `mono8` raise; `rgb8` OK |
| S5 | `step == 0` raise; `step < width*3` raise; padded `step = width*3+1` → rgb has no pad byte |
| S6 | missing info raise; `K=zeros` raise; `K=I_3` raise; `fx=0` raise; `fy=0` raise; non-finite `K` raise |
| S7 | CameraInfo 640×480 vs image 4×4 raise |
| S8 | CameraInfo stamp ignored |
| S12 | bgr8 swap |
| S15 | mutate source `data` after decode → `ImageFrame.rgb` unchanged |
| S9 | `ingest/` no `rclpy` / `cv2` / `openvino` |
| S14 | `image_msg_to_view` on a fixture `sensor_msgs/Image` |

A fixture may use a small **non-zero** RGB pattern (e.g. one red pixel) to test encoding. That is a contract buffer, not `DummySource`.

---

## 8. Done when

- `decode_frame` + S1–S15 pass with **no camera**.  
- `ros_bridge` converts real `sensor_msgs`; T07 node subscribes.  
- Outdoor live stream still Dev 5. `ImageFrame` type unchanged.

## 9. Non-goals

V4L2, camera launch, undistort, stereo, YOLOE, `/segmentation/*`, `/cmd_vel`, inventing `K`.
