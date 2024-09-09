# Use official ROS Noetic base image with Python 3.8
FROM osrf/ros:noetic-desktop-full

# Set environment variables for ROS and Python
ENV ROS_VERSION noetic
ENV ROS_DISTRO noetic
ENV PYTHON_VERSION 3.8.10
ENV DEBIAN_FRONTEND noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    python3-pip\
    libeigen3-dev \
    ros-noetic-nav-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-eigen \
    ros-noetic-ackermann-msgs \ 
    ros-noetic-geometry2 \
    ros-noetic-hector-gazebo \ 
    ros-noetic-hector-models \
    ros-noetic-jsk-rviz-plugins \
    ros-noetic-ros-control \ 
    ros-noetic-ros-controllers \ 
    ros-noetic-velodyne-simulator \
    ros-noetic-roscpp \
    ros-noetic-std-msgs \ 
    libgmock-dev \
    libgtest-dev \
    ros-noetic-tf2 \
    && rm -rf /var/lib/apt/lists/*

# Add X11 dependencies
RUN apt-get update && apt-get install -y \
    libx11-dev \
    x11-xserver-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libqt5widgets5 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies (Numpy, Pandas)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy pandas 

# Install CasADi version 3.6.6
RUN python3 -m pip install casadi==3.6.6

# Install rosdep
RUN apt-get update && apt-get install -y python3-rosdep

# Create a catkin workspace
RUN mkdir -p /root/catkin_ws/src

# Copy the entire project into the container's workspace
COPY . /root/catkin_ws/src

# Install package dependencies using rosdep
RUN rosdep install --from-paths /root/catkin_ws/src --ignore-src -r -y

# Build the workspace using catkin
WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Set the entrypoint to bash and source the catkin workspace
ENTRYPOINT ["/bin/bash", "-c", "source /root/catkin_ws/devel/setup.bash && bash"]

