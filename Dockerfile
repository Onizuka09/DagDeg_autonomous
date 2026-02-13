# Use the official ROS 2 Humble base image
FROM ros:humble

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    vim 	\
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-sensor-msgs \
    ros-humble-ament-index-python \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install \
    "numpy<2.0" \
    opencv-python \
    tflite-runtime

# 3. Setup Workspace
# We work in /ros2_ws, NOT /ros2_ws/src
WORKDIR /ros2_ws

# Copy your source code into the container
# Adjust this path if your src folder is named differently on your Pi
COPY ./ROS2/autonomous_robot_ws/src /ros2_ws/src

# 4. Install ROS dependencies using rosdep
RUN . /opt/ros/humble/setup.sh && \
    apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# 5. Build the workspace (from /ros2_ws)
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# 6. Source the setup script automatically
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash"]
