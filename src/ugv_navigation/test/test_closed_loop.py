"""
Closed-loop A->B tests of the Dev 4 stack (dev.md Dev 4 tasks 2, 3, 6).

For each TEST-ONLY scenario (test/closed_loop/scenarios.py) this launches
closed_loop.launch.py: the real navigation.launch.py (Smac2D + RPP + BT + recoveries)
driving a kinematic fake base on /cmd_vel_nav2, with the scenario map fed to both
costmaps. It sends a NavigateToPose goal and checks that the robot arrives without
its footprint touching lethal or unknown cells, respects the RPP
velocity bounds, reacts to a hazard that appears mid-run, and that nothing publishes
/cmd_vel. Per-scenario timings go to a benchmark CSV (UGV_CLOSED_LOOP_CSV).

It does not test perception, localization, Dev 3's real costmaps or Dev 5's safety
authority; those are stand-ins here.
"""

import csv
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from geometry_msgs.msg import Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import yaml

PKG = Path(__file__).resolve().parent.parent
HARNESS = PKG / 'test' / 'closed_loop'
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(PKG))

from scenarios import (  # noqa: E402, I100 (needs the sys.path entry above)
    cells_with, LETHAL, min_clearance, RESOLUTION, SCENARIOS, UNKNOWN)
from ugv_navigation.testbench_core import (  # noqa: E402
    format_summary, GoalRun, TwistStats, yaw_to_quaternion)

RPP = yaml.safe_load((PKG / 'config' / 'controller_server.yaml').read_text())[
    'controller_server']['ros__parameters']['FollowPath']
ROBOT_RADIUS = 0.2              # test fixture value
GOAL_TOLERANCE = 0.3            # goal checker xy 0.25 + integration slack
# Footprint (radius) must stay off lethal AND unknown cells; half-cell discretisation.
MIN_CLEARANCE = ROBOT_RADIUS - RESOLUTION / 2
VEL_EPS = 1e-3
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
RESULTS = []


@pytest.fixture(scope='module', autouse=True)
def ros_and_benchmark():
    rclpy.init()
    yield
    rclpy.shutdown()
    if RESULTS:
        print('\n' + format_summary([r['run'] for r in RESULTS]))
        out = os.environ.get('UGV_CLOSED_LOOP_CSV')
        if out:
            with open(out, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['scenario', 'status', 'elapsed_s', 'recoveries',
                                 'path_length_m', 'min_lethal_clearance_m',
                                 'min_unknown_clearance_m', 'cmd_msgs',
                                 'max_abs_v', 'max_abs_w'])
                for r in RESULTS:
                    run = r['run']
                    writer.writerow([r['scenario'], run.status, f'{run.elapsed_s:.2f}',
                                     run.recoveries, f'{r["length"]:.2f}',
                                     f'{r["clearance"]:.3f}', f'{r["unknown"]:.3f}',
                                     run.twist.count,
                                     f'{run.twist.max_abs_linear:.3f}',
                                     f'{run.twist.max_abs_angular:.3f}'])
            print(f'benchmark CSV: {out}')


class Recorder:
    """Records odometry poses, candidate twists and scenario map versions."""

    def __init__(self, node):
        self.poses = []
        self.stats = TwistStats()
        self.twists = []
        self.maps = []
        self.global_costmap = False
        node.create_subscription(Odometry, '/odom', self.on_odom, 50)
        node.create_subscription(Twist, '/cmd_vel_nav2', self.on_twist, 50)
        node.create_subscription(OccupancyGrid, '/test/scenario_map', self.maps.append, LATCHED)
        node.create_subscription(OccupancyGrid, '/global_costmap/costmap',
                                 self.on_costmap, LATCHED)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.poses.append((p.x, p.y))

    def on_twist(self, msg):
        self.stats.add(msg.linear.x, msg.angular.z)
        self.twists.append((msg.linear.x, msg.angular.z))

    def on_costmap(self, _msg):
        self.global_costmap = True


def spin_until(node, predicate, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        rclpy.spin_once(node, timeout_sec=0.05)
    return predicate()


def bt_navigator_active(node):
    client = node.create_client(GetState, '/bt_navigator/get_state')
    try:
        if not client.wait_for_service(timeout_sec=0.5):
            return False
        future = client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
        return future.done() and future.result().current_state.label == 'active'
    finally:
        node.destroy_client(client)


def run_scenario(scenario):
    proc = subprocess.Popen(
        ['ros2', 'launch', str(HARNESS / 'closed_loop.launch.py'),
         f'scenario:={scenario.name}'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    node = rclpy.create_node(f'closed_loop_test_{scenario.name}')
    rec = Recorder(node)
    try:
        assert spin_until(node, lambda: bt_navigator_active(node), 60.0), \
            'Nav2 did not become active'
        assert spin_until(node, lambda: rec.global_costmap and rec.maps, 20.0), \
            'scenario map never reached the global costmap'

        client = ActionClient(node, NavigateToPose, '/navigate_to_pose')
        assert client.wait_for_server(timeout_sec=10.0)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        x, y, yaw = scenario.goal
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        (goal.pose.pose.orientation.x, goal.pose.pose.orientation.y,
         goal.pose.pose.orientation.z, goal.pose.pose.orientation.w) = yaw_to_quaternion(yaw)

        recoveries = [0]

        def on_feedback(fb):
            recoveries[0] = fb.feedback.number_of_recoveries

        rec.poses.clear()
        start = time.monotonic()
        send = client.send_goal_async(goal, feedback_callback=on_feedback)
        rclpy.spin_until_future_complete(node, send, timeout_sec=10.0)
        handle = send.result()
        assert handle is not None and handle.accepted, 'goal rejected'
        result = handle.get_result_async()
        spin_until(node, result.done, scenario.timeout_s)
        elapsed = time.monotonic() - start
        if not result.done():
            handle.cancel_goal_async()
            status, code, msg = 'TIMEOUT', -1, ''
        else:
            wrapped = result.result()
            status = 'SUCCEEDED' if wrapped.status == 4 else f'STATUS_{wrapped.status}'
            code, msg = wrapped.result.error_code, wrapped.result.error_msg
        spin_until(node, lambda: False, 0.5)  # let the last odometry arrive
        final_cmd_vel = node.get_publishers_info_by_topic('/cmd_vel')
        run = GoalRun(scenario.goal, status, elapsed, recoveries[0], code, msg, rec.stats)
        return run, rec, final_cmd_vel
    finally:
        node.destroy_node()
        os.killpg(proc.pid, signal.SIGINT)
        try:
            output, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
        if os.environ.get('UGV_CLOSED_LOOP_VERBOSE'):
            print(output)


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_closed_loop_scenario(name):
    scenario = SCENARIOS[name]
    run, rec, cmd_vel_publishers = run_scenario(scenario)
    final_map = rec.maps[-1].data
    lethal, unknown_cells = cells_with(final_map, LETHAL), cells_with(final_map, UNKNOWN)
    clearance = min((min_clearance(x, y, lethal) for x, y in rec.poses),
                    default=float('inf'))
    unknown = min((min_clearance(x, y, unknown_cells) for x, y in rec.poses),
                  default=float('inf'))
    length = sum(math.dist(a, b) for a, b in zip(rec.poses, rec.poses[1:]))
    RESULTS.append({'scenario': name, 'run': run, 'length': length, 'clearance': clearance,
                    'unknown': unknown})

    assert run.status == 'SUCCEEDED', f'{name}: {run.status} {run.error_code} {run.error_msg}'
    gx, gy, _ = scenario.goal
    fx, fy = rec.poses[-1]
    assert math.hypot(fx - gx, fy - gy) <= GOAL_TOLERANCE, f'{name}: stopped at {fx, fy}'

    # Footprint never overlaps a lethal cell.
    assert clearance >= MIN_CLEARANCE, f'{name}: came within {clearance:.3f} m of lethal'
    # Unknown != free (arch §8.1): nor an unknown cell.
    assert unknown >= MIN_CLEARANCE, f'{name}: came within {unknown:.3f} m of unknown'

    # Candidate twists stay inside the RPP velocity window (BackUp may reverse).
    assert rec.stats.max_abs_linear <= RPP['max_linear_vel'] + VEL_EPS
    assert rec.stats.max_abs_angular <= RPP['max_angular_vel'] + VEL_EPS
    if run.recoveries == 0:
        assert rec.stats.min_linear >= -VEL_EPS, f'{name}: reversed without a recovery'
    # Boundary: only Dev 5 may publish /cmd_vel; nothing here does.
    assert cmd_vel_publishers == []

    if name == 'wall_gap':
        crossing = [y for x, y in rec.poses if abs(x - 2.0) < 0.1]
        assert crossing and all(1.0 < y < 1.8 for y in crossing), 'did not use the opening'
    if name == 'corridor':
        inside = [y for x, y in rec.poses if 1.0 < x < 3.0]
        assert inside and all(abs(y) < 0.6 for y in inside), 'left the corridor'
    if name == 'dynamic_obstacle':
        assert len(rec.maps) >= 2, 'the late hazard was never published'
