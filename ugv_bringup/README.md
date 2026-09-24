# ugv_bringup

**Owner:** Dev 5. Master launch files and runtime profiles (dev.md Dev 5 task 5, architecture.md §4).

## Profiles

```bash
ros2 launch ugv_bringup bringup.launch.py profile:=live_cam   # default, real outdoor deploy
ros2 launch ugv_bringup bringup.launch.py profile:=sim        # Gazebo, integration/CI
ros2 launch ugv_bringup bringup.launch.py profile:=bag bag_path:=/path/to/bag.mcap
```

Every profile launches `safety.launch.py` (the arbiter + motor driver -- see [`../ugv_safety/README.md`](../ugv_safety/README.md)). Only the *source* of `Image` + `CameraInfo` changes:

| Profile | Image/CameraInfo source | Notes |
|---|---|---|
| `live_cam` | `camera.launch.py` -> `v4l2_camera_node` | Dev 1 / Dev 2 consume `/camera/image_raw` + `/camera/camera_info`; neither opens the device (architecture.md §5). |
| `sim` | `sim.launch.py` -> Gazebo camera + diff_drive plugins in `ugv_robot_description/urdf/ugv.urdf.xacro` | Wheels also come from the sim plugin here, not `motor_driver_node`. |
| `bag` | `ros2 bag play <bag_path>` on the same topics | Offline outdoor check (architecture.md §4); the bag itself is not owned by Dev 5. |

## Files

```
launch/bringup.launch.py   # profile dispatch (OpaqueFunction), always includes safety.launch.py
launch/safety.launch.py    # arbiter_node + motor_driver_node
launch/camera.launch.py    # live_cam: v4l2_camera_node -> /camera/image_raw + /camera/camera_info
launch/sim.launch.py       # sim: gz sim + robot_state_publisher + spawn
worlds/outdoor.world       # minimal outdoor world: ground, dirt-track markers, lethal obstacles
```

`camera_info_url` in `camera.launch.py` defaults to empty (uncalibrated). Dev 2 owns `config/cameras/` and real intrinsics -- pass `camera_info_url:=file:///...` once that calibration exists. Do not treat the default as a calibrated camera (kill list: mono USB marketed as outdoor-meter-ready).
