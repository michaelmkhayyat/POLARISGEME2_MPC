#!/usr/bin/env python3

# ROS Headers
import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan, Imu, NavSatFix
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from vision_msgs.msg import Detection3D, Detection3DArray
from nav_msgs.msg import Odometry
import sys

# Gazebo Headers
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from control_msgs.msg import JointControllerState

# Python Packages
import numpy as np
from numpy import linalg as LA
import casadi as ca
import math
from dataclasses import dataclass

from mpc_classes import Waypoints, VehicleParameters, VehicleDynamics

class MPCC:
    """
    Class to handle MPC controller for tracking waypoints. The problem is formulated using CasADi
    and the solver backend is IPOPT (https://github.com/coin-or/Ipopt).
    """
    def __init__(self, spawn_obstacles):

        # With Obstacles

        self.spawn_obstacles = spawn_obstacles

        # Optimizer hyperparameters

        self.T = 0.1
        self.N = 20

        # Waypoints 

        self.waypoints = Waypoints()
        self.vehicle_params = VehicleParameters()
        self.dynamics  = VehicleDynamics(vehicle_params=self.vehicle_params, T = self.T)
        

        ## States in GLOBAL COORDINATES

        self.x = 0                              # x-position in a global coordinate system
        self.y = 0                              # y-position in a global coordinate system
        self.delta = 0                          # steering angle of front wheels
        self.v = 0                              # velocity in local x-direction
        self.yaw = 0                            # yaw angle
        self.yaw_old = 0                        # yaw angle at (n-1), for unwrap
        self.yaw_rate = 0                       # yaw rate
        self.beta = 0                           # slip angle at vehicle center

        ## States in LOCAL COORDINATES

        self.vx_local = 0
        self.vy_local = 0
        
        ## Control

        self.delta_dot = 0  # steering angle velocity of front wheels
        self.acc = 0        # longitudinal acceleration


        # Initalization of state and control vector

        self.current_arclength = 0
        self.state = [0,0,0,0,0,0,0]
        self.control = [0,0]

        # Contouring control parameters
        
        self.nu_min = 0 
        self.nu_max = 20

        # ROS Publishers and Subscribers

        self.ackermann_pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.path_sub = rospy.Subscriber("/mpc/path_info/", Path, self.path_callback)
        self.obs_sub = rospy.Subscriber("/mpc/obstacles_info", Detection3DArray, self.obstacle_callback)
        self.planned_path_pub = rospy.Publisher('/mpc/planned_path/rviz', MarkerArray, queue_size=10)
        self.gps_sub     = rospy.Subscriber("/gem/gps/fix", NavSatFix, self.gps_callback)
        self.imu_sub     = rospy.Subscriber("/gem/imu", Imu, self.imu_callback)
        self.odom_sub    = rospy.Subscriber("/gem/base_footprint/odom", Odometry, self.odom_callback)
        self.right_steering_angle_sub = rospy.Subscriber("/gem/right_steering_ctrlr/state", JointControllerState, self.right_steering_angle_callback)
        self.left_steering_angle_sub = rospy.Subscriber("/gem/left_steering_ctrlr/state", JointControllerState, self.left_steering_angle_callback)
        self.rate        = rospy.Rate(20)
        self.ackermann_msg = AckermannDrive()

        # Variables for the right and left steering angle
        self.right_wheel_steering_angle = 0.0
        self.left_wheel_steering_angle = 0.0

        self.lat         = 0.0
        self.lon         = 0.0
        self.alt         = 0.0
        self.imu_yaw     = 0.0
        self.imu_yaw_rate = 0.0

        self.x_dot = 0.0
        self.y_dot = 0.0
        self.gazebo_yaw = 0.0
        self.vx_local = 0.0
        self.vy_local = 0.0

        self.obstacles_positions = []
        self.obstacles_radii = []
        self.path_positions_x = []
        self.path_positions_y = []

    
    def set_cost_params(self, qc, ql, Ru, Rv, Rx, gamma):
        """
        Sets up the cost function parameters.
        """

        self.qc = qc
        self.ql = ql
        self.Ru = Ru
        self.Rx = Rx
        self.Rv = Rv
        self.gamma = gamma

    def right_steering_angle_callback(self, msg):
        """
        Callback function to process the right wheel steering angle.
        """
        self.right_wheel_steering_angle = msg.process_value

    def left_steering_angle_callback(self, msg):
        """
        Callback function to process the right wheel steering angle.
        """
        self.left_wheel_steering_angle = msg.process_value 

    def gps_callback(self, msg):
        """
        Callback function to process the GPS coordinates
        """
        self.lat = msg.latitude
        self.lon = msg.longitude
        self.alt = msg.altitude

    def imu_callback(self, msg):
        """
        Callback function to process the IMU data
        """
        orientation_q      = msg.orientation
        angular_velocity   = msg.angular_velocity
        linear_accel       = msg.linear_acceleration

        orientation_list   = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        self.imu_yaw       = yaw
        self.imu_yaw_rate = angular_velocity.z

    def odom_callback(self, msg: Odometry):
        """
        Callback function to handle odometry data.
        Note that a transformation from the rear-axle to the CoG is necessary.
        The odometry obtains data with respect to 'base_link' and not the CoG.

        The function transforms the position, orientation, and velocities from the 
        rear axle frame ('base_link') to the CoG frame. This includes compensating for 
        the offset between the rear axle and the CoG, accounting for the vehicle's yaw angle 
        and yaw rate.
        
        The steps are provided in the comments seen below.
        """
        #  Convert the orientation from quaternion to Euler angles to obtain the yaw (heading).
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        # Unwrap the yaw angle to ensure a smooth transition without discontinuities at +/- pi
        self.gazebo_yaw = yaw
        yaw = self.unwrap(self.yaw_old, yaw)                            
        self.yaw_old = yaw
        
        #  Adjust the vehicle's position by projecting the rear axle coordinates to the CoG, 
        # using the yaw angle and the known distance between the rear axle and the CoG.
        self.x = self.x + self.vehicle_params.geometry.lr * np.cos(yaw)  
        self.y = self.y + self.vehicle_params.geometry.lr * np.sin(yaw)
        self.yaw = yaw

        # Compute the vehicle's velocity at the CoG by adjusting the rear axle's linear 
        # velocities (in both x and y directions) based on the yaw rate and vehicle geometry.
        v_rear_x = msg.twist.twist.linear.x
        v_rear_y = msg.twist.twist.linear.y
        self.yaw_rate = msg.twist.twist.angular.z
        v_cog_x = v_rear_x - self.vehicle_params.geometry.lr * self.yaw_rate * np.sin(yaw)
        v_cog_y = v_rear_y + self.vehicle_params.geometry.lr * self.yaw_rate * np.cos(yaw)
        self.v = np.sqrt(v_cog_x**2 + v_cog_y**2)
        self.x_dot = v_cog_x
        self.y_dot = v_cog_y
        
        # Convert global velocities (in x and y) to local velocities in the vehicle's 
        # frame of reference.
        v_local_x, v_local_y = self.calculate_local_velocities(v_cog_x, v_cog_y, yaw)

        # Store the local velocities and compute the vehicle's side slip angle
        # using the local velocity components.
        self.vx_local = v_local_x
        self.vy_local = v_local_y
        self.beta = ca.atan2(v_local_y, v_local_x)

    def obstacle_callback(self, msg):
        """
        Callback function to handle incoming obstacle data.
        """
        self.obstacles_positions = []
        self.obstacles_radii = []

        for detection in msg.detections:
            position = detection.bbox.center.position
            radius = detection.bbox.size.x  # Assuming size.x represents the radius

            self.obstacles_positions.append((position.x, position.y))
            self.obstacles_radii.append(radius)

    def path_callback(self, msg):
        """
        Callback function to handle incoming path data.
        """
        path_changed = self.waypoints.load_path(msg)        # Load the new path points into the Path class
        if path_changed:
            self.waypoints.cubic_spline(step=10)            # Generate cubic spline after loading the path

    def publish_marker_visualization(self,predicted_waypoints_x, predicted_waypoints_y):
        """
        Helper function to publish predicted future positions of the vehicle
        """       
        marker_array = MarkerArray()
        predicted_waypoints_x = predicted_waypoints_x[2:]
        predicted_waypoints_y= predicted_waypoints_y[2:]

        for i in range(len(predicted_waypoints_x)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "mpc_predicted"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = predicted_waypoints_x[i]
            marker.pose.position.y = predicted_waypoints_y[i]
            marker.pose.position.z = 0  
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Set transparency

            marker_array.markers.append(marker)

            # Publish the markers
        self.planned_path_pub.publish(marker_array)
    
    def unwrap(self,previous_angle, new_angle):
        """
        Helper function that unwraps the angle difference to keep it within the range [-pi, pi].
        It ensures ensures that the transition between the previous angle and the new incoming
        angle remains smooth by correcting for angle wrap-arounds that occur when the difference
        between angles exceeds pi or is less than -pi.
        """
        d = new_angle - previous_angle
        if d > math.pi:
            d -= 2 * math.pi
        elif d < -math.pi:
            d += 2 * math.pi
        return previous_angle + d
    
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

    def formulate_ocp(self):


        # Defining the optimization problem
        self.MPC = ca.casadi.Opti()

        # Defining the symbolic variables for the states and control
        self.X = self.MPC.variable(self.dynamics.NX, self.N+1)
        self.U = self.MPC.variable(self.dynamics.NY, self.N)

        # Defining the symbolic variables for the contouring control
        self.TH = self.MPC.variable(self.N+1)  # Progress
        self.NU = self.MPC.variable(self.N)    # Projected velocity on the reference path

        # Defining the parameters of the initial state and initial control.
        self.x0_param = self.MPC.parameter(self.dynamics.NX)
        self.u0_param = self.MPC.parameter(self.dynamics.NY)
        
        # Defining the cost function for the optimization
        # as defined in: 
        # Liniger, Alexander, Alexander Domahidi, and Manfred Morari. "Optimization‐based autonomous 
        # racing of 1: 43 scale RC cars." Optimal Control Applications and Methods 36, no. 5 (2015): 628-647.

        J = 0
        for k in range(self.N+1):
            J += self.qc*self.waypoints.e_c(self.X[0,k], self.X[1,k], self.TH[k])**2 + self.ql*self.waypoints.e_l(self.X[0,k], self.X[1,k], self.TH[k])**2
            J += self.Rx[4]*self.X[4,k]**2 + self.Rx[6]*self.X[6,k]**2 - self.gamma*self.TH[-1]*self.T

        for k in range(self.N-1):
            J += self.Ru[0]*(self.U[0,k+1]-self.U[0,k])**2 + self.Ru[1]*(self.U[1,k+1]-self.U[1,k])**2 + self.Rv*(self.NU[k+1]-self.NU[k])**2

        # Defining the state dynamics constraints
        for k in range(self.N):
            # Compute the derivative (state dynamics) at the current time step
            state_derivative = self.dynamics.ode_dyn(self.X[:, k], self.U[:, k])
            # Update the state using RK4
            new_state = self.dynamics.rk4_step(self.X[:, k], self.U[:, k])
            self.MPC.subject_to(self.X[:, k+1] == new_state)
            self.MPC.subject_to(self.TH[k+1] == self.TH[k] + self.T*self.NU[k])


        if self.spawn_obstacles:
            # Defining the obstacle avoidance constraints
            for k in range(self.N + 1):
                print(len(self.obstacles_positions))
                for idx, (obs_x, obs_y) in enumerate(self.obstacles_positions):
                    obs_radius = self.obstacles_radii[idx]
                    dist_sq = (self.X[0, k] - obs_x)**2 + (self.X[1, k] - obs_y)**2 - (0.5*obs_radius+self.vehicle_params.geometry.radius)**2
                    margin = 0.1
                    J += 0.25*ca.if_else(dist_sq>=margin, -ca.log(dist_sq),  0.5 * (((dist_sq - 2 * margin) / margin)**2 - 1) - ca.log(margin)) 

        # Defining the road boundary constraints
        self.MPC.subject_to(-4.25 +self.vehicle_params.geometry.radius <= self.waypoints.e_c(self.X[0,k], self.X[1,k], self.TH[k]))
        self.MPC.subject_to(self.waypoints.e_c(self.X[0,k], self.X[1,k], self.TH[k]) <= 4.25 - self.vehicle_params.geometry.radius )

        # Defining the state and control constraints
        self.MPC.subject_to(0 <= self.TH)
        self.MPC.subject_to(self.TH <= self.waypoints.L)
        self.MPC.subject_to(self.nu_min <= self.NU)
        self.MPC.subject_to(self.NU <= self.nu_max)

        self.MPC.subject_to(self.vehicle_params.actuation_constraints.min_steering_angle_rate <= self.U[0,:])
        self.MPC.subject_to(self.U[0,:] <= self.vehicle_params.actuation_constraints.max_steering_angle_rate)

        self.MPC.subject_to(self.vehicle_params.actuation_constraints.min_acceleration<= self.U[1,:])
        self.MPC.subject_to(self.U[1,:] <= self.vehicle_params.actuation_constraints.max_acceleration)

        self.MPC.subject_to(self.vehicle_params.state_constraints.min_steering_angle <= self.X[2,:])
        self.MPC.subject_to(self.X[2,:] <= self.vehicle_params.state_constraints.max_steering_angle)

        self.MPC.subject_to(self.vehicle_params.state_constraints.min_velocity <= self.X[3,:])
        self.MPC.subject_to(self.X[3,:] <= self.vehicle_params.state_constraints.max_velocity)

        self.MPC.subject_to(self.vehicle_params.state_constraints.min_lateral_velocity <= self.X[4,:])
        self.MPC.subject_to(self.X[4,:] <= self.vehicle_params.state_constraints.max_lateral_velocity)

        self.MPC.subject_to(self.vehicle_params.state_constraints.min_yaw_rate <= self.X[6,:])
        self.MPC.subject_to(self.X[6,:] <= self.vehicle_params.state_constraints.max_yaw_rate)

        
        self.MPC.subject_to(self.X[:,0] == self.x0_param[0:7])

        self.MPC.minimize(J)


        # We have intentionally set the max_iter to be low in the IPOPT solver.
        # As when we are running in RH fashion, we do not care about having a very accurate
        # solution immediately, as with time (while running at high frequency), we will get more optimal
        # solutions.
        p_opts = {'verbose_init': False,'jit': True}
        s_opts = {'tol': 1e-1, 'print_level': 0, 'max_iter': 20}
        rospy.logwarn("BUILDING SOLVER...")
        self.MPC.solver('ipopt', p_opts, s_opts)

        #Warm up
        self.MPC.set_value(self.x0_param, self.state)
        self.MPC.set_value(self.u0_param, self.control)
        
        # sol = self.MPC.solve()
        # self.MPC.set_initial(self.U, sol.value(self.U))
        # self.MPC.set_initial(self.X, sol.value(self.X))
        # self.MPC.set_initial(self.NU, sol.value(self.NU))
        # self.MPC.set_initial(self.TH, sol.value(self.TH))

    def solve_ocp(self):
        
        # Constructing the current states vector based on the
        # available states.
        x1_p = self.x
        x2_p  = self.y
        x3_p = 0.5*self.left_wheel_steering_angle + 0.5*self.right_wheel_steering_angle
        x4_p = self.vx_local 
        x5_p = self.vy_local
        x6_p = self.yaw
        x7_p = self.yaw_rate

        self.state = np.array([x1_p, x2_p, x3_p, x4_p, x5_p, x6_p,x7_p])
        self.control = np.array([self.delta_dot , self.acc ])
        self.MPC.set_value(self.x0_param, self.state)
        self.MPC.set_value(self.u0_param, self.control)

        try:
            sol = self.MPC.solve()
          
        except RuntimeError:

            control = self.MPC.debug.value(self.U)
            state = self.MPC.debug.value(self.X)
            TH = self.MPC.debug.value(self.TH)
            NU = self.MPC.debug.value(self.NU)
            self.current_arclength = self.MPC.debug.value(self.TH)[0]


            # Extract the predicted waypoints (x, y) from the solution
            predicted_waypoints_x = state[0, :]  # All x values
            predicted_waypoints_y = state[1, :]  # All y values
            self.publish_marker_visualization(predicted_waypoints_x, predicted_waypoints_y)
        
        else:

            # Extract the predicted waypoints (x, y) from the solution
            predicted_waypoints_x = sol.value(self.X)[0, :]  # All x values
            predicted_waypoints_y = sol.value(self.X)[1, :]  # All y values

            self.publish_marker_visualization(predicted_waypoints_x, predicted_waypoints_y)

            control = sol.value(self.U)
            state = sol.value(self.X)
            self.MPC.set_initial(self.X, np.hstack((sol.value(self.X)[:,1:], sol.value(self.X)[:,-1:])))
            self.MPC.set_initial(self.U, np.hstack((sol.value(self.U)[:,1:], sol.value(self.U)[:,-1:])))    
            self.MPC.set_initial(self.TH, np.hstack((sol.value(self.TH)[1:], sol.value(self.TH)[-1:])))
            self.MPC.set_initial(self.NU, np.hstack((sol.value(self.NU)[1:], sol.value(self.NU)[-1:]))) 
            self.current_arclength = sol.value(self.TH)[0]

        self.desired_delta = state[2,1]
        self.desired_v = state[3,1]
        self.delta_dot = control[0,0]
        self.acc = control[1,0]

    def run(self):
        if self.current_arclength < self.waypoints.L - 15:
            self.solve_ocp()
            self.ackermann_msg.acceleration = self.acc
            self.ackermann_msg.steering_angle_velocity = self.delta_dot
            self.ackermann_msg.steering_angle = self.desired_delta
            self.ackermann_msg.speed = self.desired_v
        else:
            self.ackermann_msg.speed = 0
            self.ackermann_msg.steering_angle = 0.0 

        self.ackermann_pub.publish(self.ackermann_msg)

def main():
    rospy.init_node('mpc_node')
    spawn_obstacles = rospy.get_param('/spawn_obstacles', True)
    mpc_controller = MPCC(spawn_obstacles)
    rospy.sleep(1)
    rate = rospy.Rate(20)
    mpc_controller.set_cost_params(qc=0.5, 
                                   ql=0.1, 
                                   Ru = np.array([10, 10]),
                                   Rv = 0.1,
                                   Rx = np.array([0,0,0,0,1,0,1]),
                                   gamma=1)
    mpc_controller.formulate_ocp()
    while not rospy.is_shutdown():
        mpc_controller.run()
        rate.sleep()

if __name__ == '__main__':
    main()
