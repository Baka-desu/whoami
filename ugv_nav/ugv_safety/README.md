# ugv_safety: safety arbiter (Dev 5)

The only publisher of the base `/cmd_vel` (architecture.md §3.1). Nav2 computes `/cmd_vel_nav2`; this
node decides what reaches the base.

```
ros2 launch ugv_safety safety.launch.py [config_path:=...]
```

| Level | Condition | `/cmd_vel` |
|---|---|---|
| 1 | `/ugv/e_stop` asserted (latched until an explicit `false`) | zero, immediately |
| 2 | health fail: `/camera/camera_info`, `/ugv/perception_degraded`, `/ugv/pose_valid` or `/ugv/nav2_heartbeat` silent or never received, or Nav2 heartbeat `false` | zero |
| 3 | `/ugv/perception_degraded` true, or `/ugv/pose_valid` false | zero (hold) |
| 4 | otherwise | `/cmd_vel_nav2`, clamped; older than 0.5 s, non-finite or absent means zero |

A fresh arbiter holds at level 2 until every watched source has spoken. The zero is immediate by default, as
architecture.md §3.1 says; `ramp_on_hold: true` in the config adds a smooth stop on levels 2 and 3 (dev.md Dev 5
task 3) and needs the owner's agreement. `/cmd_vel` is published at
`publish_rate_hz` whatever the inputs do. `/ugv/safety_status` (type undefined in dev.md, so a plain
string like the other status topics) is `L<level> <NAME>[: reasons]`, published on change.

Timeouts and limits: `ugv_nav/config/safety/safety_timeouts.yaml`. The camera row has no number in
the spec (1.0 s is a choice); the speed limits are Nav2's placeholders and need Dev 5's real values.

The TF row of the §12 table is covered by `/ugv/pose_valid`, which Dev 2 sets false when TF is missing.
Dev 4's open "hold-aware BT" option (`/ugv/safety_hold`) is not implemented here.
