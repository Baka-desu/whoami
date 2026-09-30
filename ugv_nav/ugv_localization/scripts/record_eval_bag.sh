#!/usr/bin/env bash
# Record an evaluation bag for Dev 2 (sim now, real camera later).
# Records sensor INPUTS only (+ ground truth if present); excludes /tf so the stack
# regenerates odom->base_link and map->odom on replay, and excludes /clock because
# `ros2 bag play --clock` (bag_eval.launch.py) publishes its own.
#
#   bash record_eval_bag.sh eval_bags/sim_loop1
# Override topics via env: IMAGE_TOPIC, CAMERA_INFO_TOPIC, DEPTH_TOPIC, DEPTH_CLOUD_TOPIC, GT_DEPTH_TOPIC,
# WHEEL_ODOM_TOPIC, GT_TOPIC. Set GT_DEPTH_TOPIC="" off-sim (no ground-truth depth camera).
set -euo pipefail

OUT="${1:?usage: record_eval_bag.sh <output_dir>}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/image_raw}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/camera_info}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/perception/depth/image}"              # Dev 1 DA3 depth image (primary)
DEPTH_CLOUD_TOPIC="${DEPTH_CLOUD_TOPIC:-/perception/depth_cloud}"  # Dev 1 DA3 PointCloud2 (fallback)
GT_DEPTH_TOPIC="${GT_DEPTH_TOPIC-/camera/depth/image_raw}"      # Dev 5 sim depth camera; TBD
WHEEL_ODOM_TOPIC="${WHEEL_ODOM_TOPIC:-/wheel/odom}"
GT_TOPIC="${GT_TOPIC:-/ground_truth/odom}"                      # Dev 5 sim; TBD

TOPICS=("$IMAGE_TOPIC" "$CAMERA_INFO_TOPIC" "$DEPTH_TOPIC" "$DEPTH_CLOUD_TOPIC" "$WHEEL_ODOM_TOPIC" "$GT_TOPIC" /tf_static)
if [[ -n "$GT_DEPTH_TOPIC" ]]; then
  TOPICS+=("$GT_DEPTH_TOPIC")
fi
exec ros2 bag record -o "$OUT" "${TOPICS[@]}"
