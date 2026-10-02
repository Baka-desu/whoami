# Full live_cam runtime: ROS 2 Lyrical + Nav2 + RTAB-Map + camera calibration + CUDA PyTorch for Dev 1's
# RUGD SegFormer-B5 / Depth Anything 3 Metric Large (NVIDIA path, user_manual.md §3; no OpenVINO).
#
#   docker build -t ugv-lyrical-nav2 -f src/ugv_navigation/docker/lyrical-nav2.Dockerfile src/ugv_navigation/docker
#   docker build -t ugv-live -f ugv_nav/ugv_bringup/docker/live.Dockerfile ugv_nav/ugv_bringup/docker
#
# Run instructions: ugv_nav/ugv_bringup/README.md ("live_cam on a Windows laptop").
FROM ugv-lyrical-nav2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-lyrical-rtabmap-slam ros-lyrical-rtabmap-sync ros-lyrical-rtabmap-odom ros-lyrical-rtabmap-util \
    ros-lyrical-rtabmap-msgs ros-lyrical-nav2-costmap-2d ros-lyrical-robot-state-publisher \
    ros-lyrical-camera-calibration ros-lyrical-std-srvs ros-lyrical-tf2-msgs ros-lyrical-tf2-ros-py \
    ros-lyrical-ros2bag ros-lyrical-launch-testing-ament-cmake \
    python3-opencv python3-fastapi python3-uvicorn python3-pydantic python3-httpx \
    python3-pip git curl \
  && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.8 (cp314 wheels). --ignore-installed: never uninstall Debian-owned Python packages.
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# SegFormer (transformers) + DA3 runtime deps. Debian-owned packages pip wants to upgrade (idna, click) cannot be
# uninstalled, so they are installed alongside first.
RUN pip3 install --break-system-packages --ignore-installed --no-deps idna click  && pip3 install --break-system-packages --no-cache-dir     transformers safetensors einops addict omegaconf tqdm imageio huggingface_hub

# Depth Anything 3 model code. Its pyproject pins python<=3.13 and pulls numpy<2, open3d, xformers, pycolmap
# (no cp314 wheels); the model path only needs the deps above (xformers / gsplat / e3nn imports are optional),
# so it is installed without them. Verified on Lyrical (Python 3.14): da3metric-large loads and runs on CUDA.
RUN git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git /opt/Depth-Anything-3  && pip3 install --break-system-packages --no-deps --ignore-requires-python /opt/Depth-Anything-3

# ros-lyrical-camera-calibration 7.1.7 calls get_logger().warn(), which Lyrical's rclpy removed, so
# cameracalibrator crashes on start. Same fix as ours (.warning) until upstream ships one.
RUN sed -i 's/get_logger()\.warn(/get_logger().warning(/g' \
    /opt/ros/lyrical/lib/python3.14/site-packages/camera_calibration/camera_calibrator.py
# Its SAVE writes an empty tarball: ndarray.tostring() was removed in numpy 2 (.tobytes() is the same bytes).
RUN grep -rl '\.tostring()' /opt/ros/lyrical/lib/python3.14/site-packages/camera_calibration \
    | xargs -r sed -i 's/\.tostring()/.tobytes()/g'
