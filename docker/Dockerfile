# Use the official ROS 2 Humble base image
FROM ros:humble

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-sensor-msgs \
    ros-humble-ament-index-python \
    libextstack7 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
# We install tflite-runtime specifically. 
# Note: For x86_64 (Laptop), this works via pip. 
# For Raspberry Pi, you might need the specific wheel.
RUN pip3 install --upgrade pip
RUN pip3 install \
    numpy \
    opencv-python \
    tflite-runtime

# Copy ros2 
WORKDIR /ros2_ws
COPY ./ROS2/autonomous_robot_ws/src /ros2_ws

# 4. Install ROS dependencies using rosdep
RUN . /opt/ros/humble/setup.sh && \
    apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*


# 5. Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# 6. Source the setup script automatically on shell entry
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Set the default command to bash
CMD ["bash"]