#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import csv
import os
import numpy as np

class WaypointsLogger:
    def __init__(self, log_file="mpc_data.csv"):
        # Initialize the log file and clear it if it already exists
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # Set up ROS node and subscriber
        rospy.init_node('waypoints_logger', anonymous=True)
        rospy.Subscriber("/gem/base_footprint/odom", Odometry, self.odom_callback)

        # Initialize variables
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.yaw = 0.0
        self.yaw_old = 0.0
        self.yaw_rate = 0.0
        self.lr = 0.875  # Distance from rear axle to CoG (vehicle parameter)

        rospy.on_shutdown(self.save_data)

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

        # Calculate velocity at the center of gravity (CoG)
        v_rear_x = msg.twist.twist.linear.x
        v_rear_y = msg.twist.twist.linear.y
        self.yaw_rate = msg.twist.twist.angular.z
        v_cog_x = v_rear_x - self.lr * self.yaw_rate * np.sin(yaw)
        v_cog_y = v_rear_y + self.lr * self.yaw_rate * np.cos(yaw)
        self.v = np.sqrt(v_cog_x**2 + v_cog_y**2)

        # Log the data
        self.log_data()

    def log_data(self):
        """
        Log the current position, velocity, and yaw data into a CSV file.
        """
        with open(self.log_file, mode="a") as file:
            writer = csv.writer(file)
            writer.writerow([self.x, self.y, self.v, self.yaw])

    def save_data(self):
        """
        This function is called when ROS is shut down.
        It ensures that all data is properly logged.
        """
        rospy.loginfo(f"Saving odometry data to {self.log_file}")

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

    def run(self):
        """
        Keep the node running to continuously log data.
        """
        rospy.spin()

if __name__ == '__main__':
    logger = WaypointsLogger(log_file="mpc_data.csv")
    logger.run()
