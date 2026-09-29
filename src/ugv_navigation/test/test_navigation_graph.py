"""
Live ROS graph check of navigation.launch.py (Dev 4 boundary).

Launches the real launch file with autostart:=false and a TEST-ONLY costmap fixture,
then CONFIGURES the Nav2 servers. Configuring loads the Smac2D, RPP, behavior and
navigator plugins with the Dev 4 params and creates every publisher and action server,
without needing TF, odometry or costmap data from Dev 2/3. The last test then ACTIVATES
all four servers in lifecycle_manager order, using a TEST-ONLY identity static TF
map -> odom -> base_link in place of Dev 2. bt_navigator activation makes
BehaviorTree.CPP build navigate_to_pose_ugv.xml; on Lyrical that needs the planner's
is_path_valid service (ValidatePath), which only exists once the planner is active.

It does not drive the robot and does not claim end-to-end navigation works.
"""

import os
from pathlib import Path
import signal
import subprocess
import time

from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from nav2_msgs.action import NavigateToPose
import pytest
import rclpy
from rclpy.action import ActionClient

PKG = Path(__file__).resolve().parent.parent
LAUNCH_FILE = PKG / 'launch' / 'navigation.launch.py'
FIXTURE = PKG / 'test' / 'fixtures' / 'test_only_costmaps.yaml'
SERVERS = ['planner_server', 'controller_server', 'behavior_server', 'bt_navigator']
TWIST = 'geometry_msgs/msg/Twist'
# TEST-ONLY stand-in for Dev 2's TF chain (identity transforms).
TEST_TF = [('map', 'odom'), ('odom', 'base_link')]


def change_state(node, name, transition_id):
    client = node.create_client(ChangeState, f'/{name}/change_state')
    assert client.wait_for_service(timeout_sec=30.0), f'{name} did not start'
    request = ChangeState.Request()
    request.transition.id = transition_id
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
    node.destroy_client(client)
    return future.done() and future.result().success


@pytest.fixture(scope='module')
def configured_stack():
    proc = subprocess.Popen(
        ['ros2', 'launch', str(LAUNCH_FILE), f'costmap_params_file:={FIXTURE}',
         'autostart:=false'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    tf_procs = [subprocess.Popen(
        ['ros2', 'run', 'tf2_ros', 'static_transform_publisher',
         '--frame-id', parent, '--child-frame-id', child],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for parent, child in TEST_TF]
    rclpy.init()
    node = rclpy.create_node('dev4_graph_test')
    try:
        for name in SERVERS:
            assert change_state(node, name, Transition.TRANSITION_CONFIGURE), \
                f'{name} failed to configure'
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()
        for tf_proc in tf_procs:
            tf_proc.terminate()
            tf_proc.wait(timeout=10)
        os.killpg(proc.pid, signal.SIGINT)
        try:
            output, _ = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
        print(output)


def wait_for_publishers(node, topic, expected, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        infos = node.get_publishers_info_by_topic(topic)
        if {i.node_name for i in infos} >= expected:
            return infos
        rclpy.spin_once(node, timeout_sec=0.1)
    return node.get_publishers_info_by_topic(topic)


def test_candidate_topic_published_by_controller_and_behaviors(configured_stack):
    infos = wait_for_publishers(configured_stack, '/cmd_vel_nav2',
                                {'controller_server', 'behavior_server'})
    assert {i.node_name for i in infos} == {'controller_server', 'behavior_server'}
    assert {i.topic_type for i in infos} == {TWIST}


def test_nothing_publishes_final_cmd_vel(configured_stack):
    rclpy.spin_once(configured_stack, timeout_sec=0.5)
    assert configured_stack.get_publishers_info_by_topic('/cmd_vel') == []


def test_navigate_to_pose_action_server_exists(configured_stack):
    client = ActionClient(configured_stack, NavigateToPose, '/navigate_to_pose')
    assert client.wait_for_server(timeout_sec=10.0)
    client.destroy()


def test_stack_activates_with_ugv_behavior_tree(configured_stack):
    # Runs last. Same order as lifecycle_manager; bt_navigator activation parses the BT
    # XML and instantiates every BT node against the live servers.
    for name in SERVERS:
        assert change_state(configured_stack, name, Transition.TRANSITION_ACTIVATE), \
            f'{name} failed to activate'
    # Active publishers still respect the boundary.
    infos = wait_for_publishers(configured_stack, '/cmd_vel_nav2',
                                {'controller_server', 'behavior_server'})
    assert {i.node_name for i in infos} == {'controller_server', 'behavior_server'}
    assert configured_stack.get_publishers_info_by_topic('/cmd_vel') == []
