#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import csv
import os
import numpy as np
import rospy

class Logger:
    def __init__(self):
        # Expand the user path to get the absolute path
        current_dir = os.path.dirname(__file__)

        # Initialize variables
        self.logs = []  # In-memory log storage
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.v = 0.0
        self.beta = 0.0
        self.yaw = 0.0
        self.yaw_old = 0.0
        self.yaw_rate = 0.0
        self.lr = 0.875  # Distance from rear axle to CoG (vehicle parameter)

        # Initialize ROS node
        rospy.init_node('logger', anonymous=True)

        # Set up the subscriber for odometry data
        rospy.Subscriber("/gem/base_footprint/odom", Odometry, self.odom_callback)
        spawn_obstacles = rospy.get_param('/spawn_obstacles', False)

        self.flag = os.path.join(current_dir, "./mpc_data/flag.txt")

        with open(self.flag, 'w') as flag:
            obstacle_flag = 1 if spawn_obstacles else 0
            flag.write(f"spawn_obstacles: {obstacle_flag}\n")
        
        log_filename = "mpc_data.csv"

        # Define the log file path
        self.log_file = os.path.join(current_dir, "./mpc_data/", log_filename)
	
        rospy.on_shutdown(self.save_data)  # Ensure data is saved when ROS shuts down

        # Log that the logger is initialized
        rospy.loginfo("Waypoints Logger initialized. Ready to log data.")
        
    def odom_callback(self, msg):
        """
        Callback function to handle odometry data, calculate position, velocity, and yaw.
        """
        # Extract position and orientation from the odometry message
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)

        # Unwrap the yaw angle to avoid discontinuities
        yaw = self.unwrap(self.yaw_old, yaw)
        self.yaw_old = yaw

        # Adjust the x and y based on the rear axle to CoG distance (lr)
        self.x += self.lr * np.cos(yaw)
        self.y += self.lr * np.sin(yaw)
        self.yaw = yaw
        # Compute the vehicle's velocity at the CoG
        v_rear_x = msg.twist.twist.linear.x
        v_rear_y = msg.twist.twist.linear.y
        self.yaw_rate = msg.twist.twist.angular.z
        v_cog_x = v_rear_x - self.lr * self.yaw_rate * np.sin(yaw)
        v_cog_y = v_rear_y + self.lr * self.yaw_rate * np.cos(yaw)

        # Use calculate_local_velocities to get local velocities vx, vy
        self.vx, self.vy = self.calculate_local_velocities(v_cog_x, v_cog_y, yaw)

        # Speed (magnitude of velocity)
        self.v = np.sqrt(self.vx**2 + self.vy**2)

        # Side-slip angle (beta)
        self.beta = np.arctan2(self.vy, self.vx)

        # Get current timestamp
        timestamp = rospy.get_time()

        # Append the data to the in-memory log
        self.logs.append([timestamp, self.x, self.y, self.vx, self.vy, self.v, self.beta, self.yaw, self.yaw_rate])

        # Log the current odometry data
        #rospy.loginfo(f"Logged data at {timestamp}: x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}, v={self.v}, beta={self.beta}, yaw={self.yaw}")

    def calculate_local_velocities(self, v_global_x, v_global_y, yaw):
        """
        Transforms the global velocity components to the local velocity components 
        in the vehicle's frame of reference.
        """
        # Create the rotation matrix
        rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)],
                                    [-np.sin(yaw), np.cos(yaw)]])

        # Global velocity vector
        global_velocity = np.array([v_global_x, v_global_y])

        # Transform to the vehicle's local frame
        local_velocity = np.dot(rotation_matrix, global_velocity)

        v_local_x = local_velocity[0]
        v_local_y = local_velocity[1]

        return v_local_x, v_local_y

    def unwrap(self, previous_angle, new_angle):
        """
        Unwrap the angle to prevent discontinuities.
        """
        d = new_angle - previous_angle
        if d > np.pi:
            d -= 2 * np.pi
        elif d < -np.pi:
            d += 2 * np.pi
        return previous_angle + d

    def save_data(self):
        """
        This function is called when ROS is shut down.
        It saves all the logged data to a CSV file.
        """
        rospy.loginfo(f"Saving odometry data to {self.log_file}")

        # Write the collected data to the CSV file
        try:
            with open(self.log_file, mode="w") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "x", "y", "vx", "vy", "v", "beta", "yaw"])  # Header
                writer.writerows(self.logs)
            rospy.loginfo("Data saved successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to save data: {str(e)}")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Log data continues via odom_callback and this loop waits until shutdown
            #rospy.loginfo("Logger running... Waiting for odometry data.")
            rate.sleep()

if __name__ == '__main__':
    logger = Logger()
    logger.run()
