# ugv_bringup: camera driver (Dev 5)

One V4L2 capture, published for the system and for the web UI. No second pipeline, no relay node.

```
ros2 launch ugv_bringup camera.launch.py calibration_file:=/path/to/real_calibration.yaml [device:=/dev/video0]
```

The stamp on every message is when the frame arrived from the capture driver, not the sensor's exposure time.

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
`ugv_localization.camera` refuses zero or fake K), a missing, empty or malformed file, or a camera whose
resolution differs from the calibration, means it exits with one clear message and publishes nothing.

## Calibrating a real camera

The driver will not run without a calibration, and calibrating needs the camera's images, so there is a
calibration mode that breaks the loop. It publishes only raw `/camera/image_raw` (no CameraInfo, no UI
stream), which nothing downstream accepts, and the safety arbiter sees the camera as silent.

```
ros2 launch ugv_bringup camera.launch.py calibration_mode:=true width:=640 height:=480
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025     --ros-args -r image:=/camera/image_raw -r camera:=/camera
# COMMIT writes the YAML; save it as ugv_nav/config/cameras/<camera_name>.yaml (see config/cameras/README.md)
# then stop calibration mode and launch normally with calibration_file:=...
```

Calibrate at the resolution you will run at: K is only valid there.

## Full stack: `bringup.launch.py profile:=live_cam`

```
ros2 launch ugv_bringup bringup.launch.py profile:=live_cam     calibration_file:=<camera yaml> device:=<V4L2 path or stream URL>     camera_x:=<m> camera_y:=<m> camera_z:=<m> camera_pitch_deg:=<deg>     perception_src:=<repo>/turing/src [mode:=mapping|localize] [robot:=primary]
```

Starts camera driver, robot description (`ugv_robot_description`: base_link -> camera_optical_frame from the
measured mount, no defaults), Dev 1 perception, Dev 2 localization, Dev 3 semantic costmap (`ugv_costmap`),
Dev 4 Nav2, the safety arbiter and the operator API. `sim` and `bag` are refused until they are wired.

## live_cam on a Windows laptop (ROS in Docker)

Docker on Windows cannot open a USB webcam, so the host serves it and the driver reads the stream.

1. Host (Windows Python + opencv-python): `python ugv_nav/ugv_bringup/scripts/webcam_stream.py` serves
   `http://<host>:8090/cam.mjpg` (newest frame only, no backlog). Stop it with Ctrl+C: it holds the camera.
2. Image: `docker build -t ugv-live -f ugv_nav/ugv_bringup/docker/live.Dockerfile ugv_nav/ugv_bringup/docker`
   (needs `ugv-lyrical-nav2` first, see that Dockerfile).
3. Container with the GPU, the repo and WSLg (for the calibrator window):
   ```
   docker run -it --gpus all -p 8080:8080 -v <repo>:/repo        -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix -e DISPLAY=:0 ugv-live
   ```
   then build the workspace from `/repo` and use `device:=http://host.docker.internal:8090/cam.mjpg`.

Not covered here: motor driver, wheel odometry, Gazebo `sim` profile.
