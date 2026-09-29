# ugv_eval

**Owner:** Dev 5. E-stop CLI utility and safety verification tools (dev.md task 6).

```bash
ros2 run ugv_eval ugv-estop --set true    # hard kill: arbiter holds at Level 1, ramps to zero
ros2 run ugv_eval ugv-estop --set false   # release
```

Publishes `/ugv/e_stop` latched (TRANSIENT_LOCAL), matching the subscription QoS in `ugv_safety.node.arbiter_node` so a late-joining arbiter still sees the current state instead of defaulting to not-stopped.

`bool_arg.py` holds the `--set true/false` parsing with no `rclpy` import, so it's testable on a machine without a ROS install (`python -m pytest ugv_eval`). The Definition-of-Done precedence check itself (estop / stale perception / invalid pose all -> zero `/cmd_vel`) lives with the kernel it exercises, in `ugv_safety/src/ugv_safety/tests/test_safety_precedence.py`.
