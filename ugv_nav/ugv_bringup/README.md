# ugv_bringup: camera driver (Dev 5)

One V4L2 capture, published for the system and for the web UI. No second pipeline, no relay node.

```
ros2 launch ugv_bringup camera.launch.py calibration_file:=/path/to/real_calibration.yaml [device:=/dev/video0]
```

| Topic | Type | QoS | For |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` rgb8 | reliable, volatile, depth 1 | Dev 1 perception, Dev 2 RTAB-Map |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | reliable, transient local, depth 1 | Dev 1, Dev 2, safety arbiter. One per image, same stamp |
| `/image_raw/compressed` | `sensor_msgs/CompressedImage` jpeg | reliable, volatile, depth 1 | web UI (rosbridge), `compressed_rate_hz` (default 5) |
| `/camera_info` | `sensor_msgs/CameraInfo` | reliable, transient local, depth 1 | web UI, published with each compressed frame |

The internal names (`/camera/*`) are what Dev 1 and Dev 2 already use; the UI defaults to
`/image_raw/compressed` and `/camera_info` (`ui/src/App.tsx`). The driver publishes both sets from the
same capture, so neither side was renamed.

`calibration_file` has no default and the driver fails closed: no valid calibration (Dev 2's
`ugv_localization.camera` refuses zero or fake K), or a camera whose resolution differs from the
calibration, means it publishes nothing. Capture a calibration with
`ros2 run ugv_localization camera_info_to_yaml`.

Not covered here: the full `profile:=live_cam|bag` bringup, URDF, motor driver.
