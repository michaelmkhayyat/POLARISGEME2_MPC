cmake_minimum_required(VERSION 2.8.3)

project(gem_gazebo)

find_package(catkin REQUIRED COMPONENTS
  ackermann_msgs
  controller_manager
  controller_manager_msgs
  effort_controllers
  gazebo_ros
  gem_description
  geometry_msgs
  joint_state_controller
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  vision_msgs
)

catkin_package()

catkin_install_python(PROGRAMS scripts/gem_control.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS scripts/gem_sensor_info.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
                                         
catkin_install_python(PROGRAMS scripts/path_and_obstacles_publisher.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
                      
catkin_install_python(PROGRAMS scripts/obstacle_spawner.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
                      
install(DIRECTORY config launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
