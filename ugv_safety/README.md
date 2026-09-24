# ugv_safety -- Dev 5 workspace

**Owner:** Dev 5 (Safety Authority & Platform)
**Authority:** [`architecture.md`](../architecture.md) wins over [`dev.md`](../dev.md) on any conflict.
**Product path:** every candidate twist and every health signal in the system funnels through here before a wheel turns.

## What Dev 5 is

Dev 5 is the **final gate**, not the brain. The brain is RTAB-Map + Nav2 (Dev 2 / Dev 4). Dev 5 owns:

1. **The 4-tier priority arbiter** -- sole authoritative publisher to base `/cmd_vel` (architecture.md §3.1).
2. **The system health watchdog table** -- independent arrival-based liveness check on camera, perception, pose, and Nav2 (architecture.md §12).
3. **The deceleration ramp** -- turns "zero twist" into a controlled stop rather than a discontinuous jump.
4. **The motor driver seam** -- diff-drive kinematics from final `/cmd_vel` to wheel commands (real-hardware seam; the `sim` profile uses the Gazebo plugin instead).
5. **Platform**: URDF/xacro, dual footprints, outdoor Gazebo world, and the `profile:=live_cam|sim|bag` launch dispatch (`ugv_bringup/`), including the shared camera driver launch that Dev 1 and Dev 2 both consume.

Downstream consumers (nothing -- Dev 5 is the end of the chain on `/cmd_vel`) never see arbitration internals; they see one topic with one publisher.

## Module map

```
ugv_safety/src/ugv_safety/
  arbiter/    types.py, decide.py       pure kernel: ArbiterInputs -> ArbiterDecision
  watchdog/   table.py, monitor.py      arrival-timestamp timeout table (config/safety/safety_timeouts.yaml)
  decel/      ramp.py                   rate-limited slew from current twist to target twist
  motor/      kinematics.py             twist -> (left_rad_s, right_rad_s), real-hardware seam
  node/       arbiter_node.py           rclpy: wires the above to ROS topics, sole /cmd_vel publisher
              motor_driver_node.py      rclpy: /cmd_vel -> /ugv/wheel_cmd (not used by sim profile)
  tests/                                pytest, no rclpy required
```

Same split Dev 1 uses in `turing/`: arbitration/watchdog/ramp/kinematics logic is pure Python, testable without ROS; the `node/` layer is a thin rclpy wrapper with no decision logic of its own.

## The 4-tier precedence (architecture.md §3.1)

| Level | Trigger | Output |
|---|---|---|
| 1 | `/ugv/e_stop == true` | zero twist |
| 2 | Watchdog trip: camera / perception mask / pose-TF / Nav2 heartbeat silent or stale (`config/safety/safety_timeouts.yaml`, 0.5s each) | zero twist |
| 3 | `/ugv/perception_degraded == true` OR `/ugv/pose_valid == false` | zero twist (hold) |
| 4 | none of the above | forward `/cmd_vel_nav2` |

`decide()` in `arbiter/decide.py` always targets an exact zero or exact passthrough; `decel/ramp.py` is what makes the *published* `/cmd_vel` approach that target smoothly instead of jumping.

The watchdog table is deliberately arrival-based, not value-based: it trips both when a node reports bad state honestly (e.g. `/ugv/pose_valid=false`, which is caught at Level 3 by `pose_valid`) and when a node goes completely silent (Level 2 -- the honesty flag itself stops updating). Both failure modes end at zero twist either way.

## Interfaces consumed / produced

| Topic | Type | Direction | Watchdog name |
|---|---|---|---|
| `/ugv/e_stop` | `std_msgs/Bool` (latched) | in (operator / `ugv_eval`) | -- (Level 1, not in the table) |
| `/ugv/perception_degraded` | `std_msgs/Bool` | in (Dev 1) | `perception_mask` |
| `/ugv/pose_valid` | `std_msgs/Bool` | in (Dev 2) | `pose_tf` |
| `/cmd_vel_nav2` | `geometry_msgs/Twist` | in (Dev 4) | `nav2_heartbeat` |
| `/camera/image_raw` | `sensor_msgs/Image` | in (Dev 5's own driver, heartbeat only) | `camera` |
| `/cmd_vel` | `geometry_msgs/Twist` | **out, sole publisher** | -- |
| `/ugv/safety_status` | `std_msgs/String` | out (diagnostics: `level=.. hold=.. reason=..`) | -- |
| `/ugv/wheel_cmd` | `std_msgs/Float64MultiArray` `[left_rad_s, right_rad_s]` | out (real-hardware seam only) | -- |

## What we do not do

- We do not compute paths or control candidates -- that is Nav2 (Dev 4).
- We do not write costmap cells -- that is Dev 3.
- We do not decide *why* perception or pose are invalid -- we only trust the flags Dev 1 / Dev 2 publish, plus our own independent liveness check.
- `sim` profile physics/wheels/camera come from Gazebo plugins in `ugv_robot_description/urdf/ugv.urdf.xacro`, not from `motor_driver_node` or `ugv_bringup/launch/camera.launch.py`.

## CLI & verification

```bash
# Run the safety authority + motor driver seam
ros2 launch ugv_bringup safety.launch.py

# Pure-kernel tests (no ROS required)
python -m pytest ugv_safety

# Assert e-stop and watch the published twist ramp to zero
ros2 run ugv_eval ugv-estop --set true
ros2 topic echo /cmd_vel
ros2 topic echo /ugv/safety_status
```
