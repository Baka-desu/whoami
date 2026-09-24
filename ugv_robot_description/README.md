# ugv_robot_description

**Owner:** Dev 5 (dev.md task 4, architecture.md §7/§13 item 9).

`urdf/ugv.urdf.xacro` -- differential-drive chassis, two driven wheels + passive rear caster, and a camera mount with a `camera_optical_frame` child link (REP103 convention: z-forward/x-right/y-down). This is the `frame_id` Dev 5's camera driver stamps into `CameraInfo`/`Image` headers, and what Dev 1 and Dev 2 both bind to (architecture.md §8.5).

Dimensions are shared with two other files and must stay consistent if changed:

| Value | Here | Also in |
|---|---|---|
| `wheel_radius` | `0.08` | `ugv_safety/node/motor_driver_node.py` default param |
| `track_width` | `0.40` | `ugv_safety/node/motor_driver_node.py` default param |
| chassis `0.50 x 0.35` | -- | `config/robots/footprint_{primary,secondary}.yaml` |

`<gazebo>` blocks at the bottom are sim-only: `libgazebo_ros_diff_drive.so` subscribes to `/cmd_vel` directly (the "Base Wheels / Sim" consumer in dev.md §3's interface table -- separate from `ugv_safety`'s real-hardware motor driver seam), and `libgazebo_ros_camera.so` publishes the same `/camera/image_raw` + `/camera/camera_info` contract the live camera driver does.

Dual footprints (primary = full envelope + margin, secondary = tighter) live in `config/robots/`, not here -- footprint is a Nav2 costmap concept, not URDF geometry.
