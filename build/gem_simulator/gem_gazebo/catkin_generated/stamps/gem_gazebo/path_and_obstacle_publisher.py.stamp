#!/usr/bin/env python3

import os
import math
import numpy as np
import rospkg
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from vision_msgs.msg import Detection3D, Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion

# Constants
WORLD_FRAME_ID = "world"
WAYPOINTS_FILE = "waypoints.csv"
LEFT_BOUNDARY_FILE = "left_boundary_points.csv"
RIGHT_BOUNDARY_FILE = "right_boundary_points.csv"
OBSTACLES_FILE = "obstacles.csv"


class PathAndObstaclesPublisher:
    """
    A ROS node for publishing waypoints, path boundaries, and obstacles to appropriate topics.
    """

    def __init__(self):
        rospy.loginfo("Initializing PathAndObstaclesPublisher node...")
        self.rate = rospy.Rate(50)
        self.mpc_pos_log = []

        # Set up ROS package paths
        rospack = rospkg.RosPack()
        mpc_gazebo_dir = rospack.get_path("gem_gazebo")

        # Load data files
        self.waypoints = self.load_csv(os.path.join(mpc_gazebo_dir, "data", WAYPOINTS_FILE))
        self.left_boundary_points = self.load_csv(os.path.join(mpc_gazebo_dir, "data", LEFT_BOUNDARY_FILE))
        self.right_boundary_points = self.load_csv(os.path.join(mpc_gazebo_dir, "data", RIGHT_BOUNDARY_FILE))
        self.obstacles = self.load_csv(os.path.join(mpc_gazebo_dir, "data", OBSTACLES_FILE), skip_header=1)

        # Set up ROS publishers and subscribers
        self.setup_ros_pub_sub()

        # TF broadcaster for simulating localization
        self.broadcaster = tf.TransformBroadcaster()

    @staticmethod
    def load_csv(file_path, skip_header=0):
        """
        Loads a CSV file into a numpy array.
        :param file_path: Path to the CSV file.
        :param skip_header: Number of lines to skip at the beginning of the file.
        :return: Numpy array of CSV data.
        """
        rospy.loginfo(f"Loading CSV data from {file_path}")
        return np.genfromtxt(file_path, delimiter=",", skip_header=skip_header)

    def setup_ros_pub_sub(self):
        """
        Sets up ROS publishers and subscribers.
        """
        rospy.loginfo("Setting up ROS publishers and subscribers...")
        self.odometry_sub = rospy.Subscriber("/gem/base_footprint/odom", Odometry, self.odom_cb, queue_size=1)
        self.path_pub = rospy.Publisher("/mpc/path_info", Path, queue_size=1)
        self.path_rviz_pub = rospy.Publisher("/mpc/path/rviz", Path, queue_size=1)
        self.path_boundary_l_rviz_pub = rospy.Publisher("/mpc/path_boundary_l/rviz", Path, queue_size=1)
        self.path_boundary_r_rviz_pub = rospy.Publisher("/mpc/path_boundary_r/rviz", Path, queue_size=1)
        self.obstacles_pub = rospy.Publisher("/mpc/obstacles_info", Detection3DArray, queue_size=1)
        self.obstacles_rviz_pub = rospy.Publisher("/mpc/obstacles/markers", MarkerArray, queue_size=1)

    def odom_cb(self, msg):
        """
        Callback function for odometry data. Updates position and orientation of the vehicle.
        :param msg: ROS Odometry message.
        """
        _, _, yaw = euler_from_quaternion(
            [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        )
        speed = np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.mpc_pos_log.append((msg.pose.pose.position.x, msg.pose.pose.position.y, yaw, speed))

        # Broadcast TF for vehicle localization
        self.broadcaster.sendTransform(
            (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
            rospy.Time.now(),
            msg.child_frame_id,
            msg.header.frame_id,
        )

    def publish_path(self):
        """
        Publishes the waypoints and path boundaries to their respective ROS topics.
        """
        self.publish_waypoints(self.waypoints, self.path_rviz_pub, "Path waypoints")
        self.publish_waypoints(self.left_boundary_points, self.path_boundary_l_rviz_pub, "Left boundary")
        self.publish_waypoints(self.right_boundary_points, self.path_boundary_r_rviz_pub, "Right boundary")
        self.publish_waypoints(self.waypoints, self.path_pub, "Path GPS waypoints")

    def publish_waypoints(self, points, publisher, description):
        """
        Helper method to publish waypoints to a ROS Path topic.
        :param points: Array of waypoint coordinates.
        :param publisher: ROS publisher to publish the Path message.
        :param description: Description for logging.
        """
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = WORLD_FRAME_ID
        for pt in points:
            pose = PoseStamped()
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        rospy.loginfo(f"Publishing {description}")
        publisher.publish(path_msg)

    def publish_obstacles(self):
        """
        Publishes obstacle data to Detection3DArray and MarkerArray topics.
        """
        detection_msg = Detection3DArray()
        detection_msg.header.stamp = rospy.Time.now()
        detection_msg.header.frame_id = WORLD_FRAME_ID

        marker_msg = MarkerArray()

        for idx, obstacle in enumerate(self.obstacles):
            self.create_obstacle_detection(idx, obstacle, detection_msg)
            self.create_obstacle_marker(idx, obstacle, marker_msg)

        rospy.loginfo("Publishing obstacles data")
        self.obstacles_pub.publish(detection_msg)
        self.obstacles_rviz_pub.publish(marker_msg)

    @staticmethod
    def create_obstacle_detection(idx, obstacle, detection_msg):
        """
        Creates and appends an obstacle detection to the Detection3DArray.
        """
        detection = Detection3D()
        detection.bbox.center.position.x = obstacle[0]
        detection.bbox.center.position.y = obstacle[1]
        detection.bbox.size.x = obstacle[2]
        detection.bbox.size.y = obstacle[2]
        detection_msg.detections.append(detection)

    @staticmethod
    def create_obstacle_marker(idx, obstacle, marker_msg):
        """
        Creates and appends a marker to visualize the obstacle in RViz.
        """
        # Obstacle marker (cylinder)
        marker = Marker()
        marker.header.frame_id = WORLD_FRAME_ID
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.CYLINDER
        marker.id = idx
        marker.scale.x = obstacle[2] * 2
        marker.scale.y = obstacle[2] * 2
        marker.scale.z = 1.0
        marker.color.g = 1.0
        marker.color.a = 0.6
        marker.pose.position.x = obstacle[0]
        marker.pose.position.y = obstacle[1]
        marker.pose.position.z = marker.scale.z * 0.5
        marker_msg.markers.append(marker)

    def run(self):
        """
        Main loop to periodically publish path and obstacle data.
        """
        rospy.loginfo("Starting to publish path and obstacle data...")
        while not rospy.is_shutdown():
            self.publish_path()
            self.publish_obstacles()
            self.rate.sleep()


def main():
    rospy.init_node("path_and_obstacles_publisher", anonymous=True)
    publisher = PathAndObstaclesPublisher()
    try:
        publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("PathAndObstaclesPublisher node interrupted.")


if __name__ == "__main__":
    main()

