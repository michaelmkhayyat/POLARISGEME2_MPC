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

@dataclass
class TireParameters:
    Csf: float
    Csr: float

@dataclass
class VehicleGeometry:
    length: float
    lf:float
    lr:float
    wheelbase: float
    radius: float

@dataclass
class VehicleIntertia:
    mass: float
    Izz: float

@dataclass
class VehicleStateConstraints:
    min_velocity: float
    max_velocity: float
    min_lateral_velocity: float
    max_lateral_velocity: float
    min_yaw_rate: float
    max_yaw_rate: float
    min_steering_angle: float
    max_steering_angle: float

@dataclass
class VehicleActuationConstraints:
    min_acceleration: float
    max_acceleration: float
    min_steering_angle_rate: float
    max_steering_angle_rate: float

class Waypoints:
    """
    Class to handle track-related calculations such as cubic spline interpolation,
    path point storage, and calculation of tangents or derivatives.
    """
    def __init__(self):
        # Path-related attributes
        self.path_points_x = []
        self.path_points_y = []
        self.previous_path_x = None
        self.previous_path_y = None
        self.cs_x = None
        self.cs_y = None
        self.Phi = None
        self.L = 0  # Length of the path in terms of waypoints

    def load_path(self, path_msg):
        """
        Load path points from a ROS Path message, and check if they have changed.
        """
        new_path_x = np.array([pose.pose.position.x for pose in path_msg.poses])
        new_path_y = np.array([pose.pose.position.y for pose in path_msg.poses])

        # Check if the path has changed
        if self.previous_path_x is not None and self.previous_path_y is not None:
            if np.array_equal(new_path_x, self.previous_path_x) and np.array_equal(new_path_y, self.previous_path_y):
                #rospy.loginfo("Path has not changed, skipping cubic spline generation.")
                return False  # No change, skip further processing

        # If the path has changed, update the stored path and process
        self.previous_path_x = new_path_x
        self.previous_path_y = new_path_y
        self.path_points_x = new_path_x
        self.path_points_y = new_path_y
        return True  # Path changed, proceed with processing

    def cubic_spline(self, step):
        """
        Generate a b-spline interpolant for the path points.
        """
        x_list = self.path_points_x
        y_list = self.path_points_y

        self.wp_len = len(x_list)
        l_list = np.arange(0, self.wp_len, 1)
        self.L = int(self.wp_len/step)*step
        self.cs_x = ca.interpolant('cs_x','bspline',[l_list[::step]],x_list[::step])
        self.cs_y = ca.interpolant('cs_y','bspline',[l_list[::step]],y_list[::step])
        th = ca.MX.sym('th')

        # Tangent angle

        self.Phi = ca.Function('Phi', [th], [ca.atan2(ca.jacobian(self.cs_y(th),th),(ca.jacobian(self.cs_x(th),th)))])

        X = ca.MX.sym('X')
        Y = ca.MX.sym('Y')
        th = ca.MX.sym('th')

        self.e_c = ca.Function('e_c', [X, Y, th], [ca.sin(self.Phi(th))*(X - self.cs_x(th)) - ca.cos(self.Phi(th))*(Y - self.cs_y(th))])
        self.e_l = ca.Function('e_l', [X, Y, th], [-ca.cos(self.Phi(th))*(X - self.cs_x(th)) - ca.sin(self.Phi(th))*(Y - self.cs_y(th))]) 

class VehicleParameters:
    """
    Class to define the parameters of the POLARIS GEM E2 Vehicle.
    """
    def __init__(self):
        self.geometry = VehicleGeometry(
            length = 3.0,                   # Length of the vehicle in meters
            wheelbase = 1.75,               # Wheelbase (distance between front and rear axles) in meters
            lf = 0.875,                     # Distance between front axle and estimated CoG in meters
            lr = 0.875,                     # Distance between rear axle and estimated CoG in meters
            radius = 1.5,                   # Esimated radius of the bounding circle for collision avoidance in meters
        )
        
        self.state_constraints = VehicleStateConstraints(
            min_velocity = 0.0,             # Minimum absolute velocity in m/s
            max_velocity = 20.0/3.6,        # Maximum absolute velocity in m/s
            min_lateral_velocity = -0.5,      # Minimum lateral velocity in the vehicle reference frame in m/s
            max_lateral_velocity = 0.5,       # Maximum lateral velocity in the vehicle reference frame in m/s
            min_yaw_rate = -0.5,            # Minimum yaw rate in rad/s
            max_yaw_rate = 0.5,             # Maximum yaw rate in rad/s
            min_steering_angle = -0.61,      # Minimum front wheels steering angle in rad
            max_steering_angle = 0.61      # Maximum front wheels steering angle in rad
        )
        
        self.actuation_constraints = VehicleActuationConstraints(
            min_acceleration = -5.0,        # Minimum commanded acceleration (braking) in m/s^2
            max_acceleration = 5.0,         # Maximum commanded acceleration in m/s^2
            min_steering_angle_rate = -0.4, # Minimum commanded steering angle rate in radians
            max_steering_angle_rate = 0.4   # Maximum commanded steering angle rate in radians
        )

        self.tire = TireParameters(
            Csf = 2000,                     # Front tire cornering stiffness
            Csr = 2000                      # Rear tire cornering stiffness                     
        )

        self.inertia = VehicleIntertia(
            mass = 734,                     # Mass of the vehicle in kgs
            Izz =  465                      # Yaw moment of inertia in kgm^2
        )

class VehicleDynamics:
    """
    Class to handle the vehicle dynamics, including dynamics modelin, 
    state propagation (RK4) and CasADi functions for dynamics equations.
    """
    def __init__(self, vehicle_params: VehicleParameters, T: float):

        self.vehicle = vehicle_params       # Vehicle parameter
        self.T = T                          # Discretization time-step for dynamics
        self.NX = 7                         # Number of states
        self.NY = 2                         # Number of controls

        self.setup_dynamics_model()

    def setup_dynamics_model(self):
        """
        Sets up the dynamic model of the vehicle using CasADi.
        The vehicle model is the dynamic bicycle model. It is adapted from [1, 2]

        [1] Choi, Young-Min, and Jahng-Hyon Park. "Game-based lateral and longitudinal 
        coupling control for autonomous vehicle trajectory tracking." IEEE Access 10 (2021): 31723-31731.

        [2] Liniger, Alexander, Alexander Domahidi, and Manfred Morari. "Optimization‐based autonomous 
        racing of 1: 43 scale RC cars." Optimal Control Applications and Methods 36, no. 5 (2015): 628-647.

        STATES: [x]
            X:          -> x-position in the global reference frame in meters
            Y:          -> y-position in the global reference frame in meters             
            delta:      -> steering angle at front wheels in rad
            vx:         -> longitudinal velocity in the vehicle reference frame in m/s
            vy:         -> lateral velocity in the vehicle reference frame in m/s
            psi:        -> yaw angle of the vehicle in rad
            psi_dot     -> yaw rate of the vehicle in rad/s

        CONTROLS: [u]
            delta_dot:  -> steering angle rate command at the front wheels in rad/s
            acc:        -> acceleration command in m/s^2

        PARAMETERS:
            lf:         -> distance between front axle and CoG in meters
            lr          -> distance between rear axle and CoG in meters
            m           -> mass of the vehicle in kg
            Izz         -> yaw moment of inertia in kgm^2
            C_Sf        -> front wheels cornering stiffness in N/rad
            C_Sr        -> rear wheels cornering stiffness in N/rad

        The ODE that governs the dynamics of the vehicle is described as:
            \dot{X} = vx*cos(psi) - vy*sin(psi),
            \dot{Y} = vx*sin(psi) + vy*cos(psi),
            \dot{delta} = delta_dot,
            \dot{vx} = psi_dot*vy + acc,
            \dot{vy} = -psi_dot*vx + 2/m * (Fcf*cos(delta) + Fcr),
            \dot{psi} = vx/(lf + lr) * tan(delta),
            \dot{psi_dot} = 2/Izz * (lf*Fcf - lr*Fcr),

            where:
                Fcf = C_Sf*alpha_f,
                Fcr = C_Sr*alpha_r,
                beta = -atan(vy/vx),
                alpha_f = delta - beta - lf*psi_dot/vx,
                alpha_r = beta + lr*psi_dot/vx.
            
            Fcf and Fcr are the front and rear lateral forces, respectively
            beta is the vehicle side slip angle at the CoG in rad
            alpha_f is the slip angle at the front and rear axles, respectively

                **  Note that a more complex tire dynamics model could be incorporated,
                    as described in [2] (with minimal changes to our code). However, since 
                    this is not for racing purposes, we resort to the linearized tire mode 
                    for computational efficiency - it is often better to run the controller
                    of a less accurate vehicle model at a high frequency than that of a very
                    accurate vehicle model at a much lower frequency.
        """
        
        # Defining symbolic variables for the states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        delta = ca.SX.sym('delta')
        vx = ca.SX.sym('vx')
        vy = ca.SX.sym('vy')
        yaw = ca.SX.sym('yaw') 
        yaw_dot = ca.SX.sym('yaw_dot')

        # Defining symbolic variables for the controls
        delta_dot = ca.SX.sym('delta_dot')
        v_dot = ca.SX.sym('v_dot')

        x = np.array([x, y, delta, vx, vy, yaw, yaw_dot])
        u = np.array([delta_dot, v_dot])

        lr = self.vehicle.geometry.lr
        lf = self.vehicle.geometry.lf
        I = self.vehicle.inertia.Izz
        m = self.vehicle.inertia.mass
        C_Sf = self.vehicle.tire.Csf
        C_Sr = self.vehicle.tire.Csr
        eps = 1e-3
        lwb = lr+lf

        beta = -ca.atan2(vy,(vx+ca.exp(-0.5*vx)))
        alpha_r = ca.atan2(yaw_dot*lr - vy,vx+ca.exp(-0.5*vx))
        alpha_f = u[0] - ca.atan2(yaw_dot*lr + vy,vx+ca.exp(-0.5*vx))
        Fcf = C_Sf*alpha_f
        Fcr = C_Sr*alpha_r
        #Fcf = C_Sf*ca.sin(1.2*ca.arctan(alpha_f))
        #Fcr = C_Sf*ca.sin(1.2*ca.arctan(alpha_r))
        rhs= np.array([x[3]*ca.cos(x[5]) - x[4]*ca.sin(x[5]),
                      x[3]*ca.sin(x[5]) + x[4]*ca.cos(x[5]),
                      u[0],
                      x[6]*x[4] + u[1],
                      -x[6]*x[3] + 2/m*(Fcf*ca.cos(x[2]) + Fcr),
                      x[3]/(lwb)*ca.tan(x[2]),
                      2/I*(lf*Fcf - lr*Fcr)])
        self.ode_dyn = ca.Function('ode_dyn', [x,u], [rhs])

    def rk4_step(self, state, control):
        """
        Runge-Kutta 4 (RK4) integration for forward state propagation.
        """
        k1 = self.ode_dyn(state, control)
        k2 = self.ode_dyn(state + (self.T / 2.0) * k1, control)
        k3 = self.ode_dyn(state + (self.T / 2.0) * k2, control)
        k4 = self.ode_dyn(state + self.T * k3, control)

        new_state = state + (self.T / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state
    
    def euler_step(self, state, control):
        """
        Euler forward integration for forward state propagation.
        """
        der = self.ode_dyn(state, control)

        new_state = state + self.T * der
        return new_state

class MPCC:
    """
    Class to handle MPC controller for tracking waypoints. The problem is formulated using CasADi
    and the solver backend is IPOPT (https://github.com/coin-or/Ipopt).
    """
    def __init__(self, spawn_obstacles):

        # With Obstacles

        self.spawn_obstacles = spawn_obstacles

        # Optimizer hyperparameters

        self.T = 0.2
        self.H = 15

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

        self.arc = 0
        self.state = [0,0,0,0,0,0,0]
        self.control = [0,0]

        # Contouring control parameters
        
        self.nu_min = 0 
        self.nu_max = 20

        # ROS Publishers and Subscribers

        self.ackermann_pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.path_sub = rospy.Subscriber("/mpc/path_info/", Path, self.path_callback)
        self.obs_sub = rospy.Subscriber("/mpc/obstacles_info", Detection3DArray, self.obstacle_callback)
        self.marker_pub = rospy.Publisher('/mpc/planned_path/rviz', MarkerArray, queue_size=10)
        self.gps_sub     = rospy.Subscriber("/gem/gps/fix", NavSatFix, self.gps_callback)
        self.imu_sub     = rospy.Subscriber("/gem/imu", Imu, self.imu_callback)
        self.odom_sub    = rospy.Subscriber("/gem/base_footprint/odom", Odometry, self.odom_callback)
        self.right_steering_angle_sub = rospy.Subscriber("/gem/right_steering_ctrlr/state", JointControllerState, self.right_steering_angle_callback)
        self.left_steering_angle_sub = rospy.Subscriber("/gem/left_steering_ctrlr/state", JointControllerState, self.left_steering_angle_callback)
        
        self.rate        = rospy.Rate(10)
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

        self.obstacle_positions = []
        self.obstacle_radii = []
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
        self.obstacle_positions = []
        self.obstacle_radii = []

        for detection in msg.detections:
            position = detection.bbox.center.position
            radius = detection.bbox.size.x  # Assuming size.x represents the radius

            self.obstacle_positions.append((position.x, position.y))
            self.obstacle_radii.append(radius)

    def path_callback(self, msg):
        """
        Callback function to handle incoming path data.
        """
        path_changed = self.waypoints.load_path(msg)        # Load the new path points into the Path class
        if path_changed:
            self.waypoints.cubic_spline(step=40)            # Generate cubic spline after loading the path

    def publish_marker_visualization(self,predicted_waypoints_x, predicted_waypoints_y):
        """
        Helper function to publish predicted future positions of the vehicle
        """       
        marker_array = MarkerArray()

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
        self.marker_pub.publish(marker_array)
    
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
        self.X = self.MPC.variable(self.dynamics.NX, self.H+1)
        self.U = self.MPC.variable(self.dynamics.NY, self.H)

        # Defining the symbolic variables for the contouring control
        self.TH = self.MPC.variable(self.H+1)  # Progress
        self.NU = self.MPC.variable(self.H)    # Projected velocity on the reference path

        # Defining the parameters of the initial state and initial control.
        self.x0_param = self.MPC.parameter(self.dynamics.NX)
        self.u0_param = self.MPC.parameter(self.dynamics.NY)
        
        # Defining the cost function for the optimization
        # as defined in: 
        # Liniger, Alexander, Alexander Domahidi, and Manfred Morari. "Optimization‐based autonomous 
        # racing of 1: 43 scale RC cars." Optimal Control Applications and Methods 36, no. 5 (2015): 628-647.

        J = 0
        for k in range(self.H+1):
            J += self.qc*self.waypoints.e_c(self.X[0,k], self.X[1,k], self.TH[k])**2 + self.ql*self.waypoints.e_l(self.X[0,k], self.X[1,k], self.TH[k])**2
            J += self.Rx[4]*self.X[4,k]**2 + self.Rx[6]*self.X[6,k]**2 - self.gamma*self.TH[-1]

        for k in range(self.H-1):
            J += self.Ru[0]*(self.U[0,k+1]-self.U[0,k])**2 + self.Ru[1]*(self.U[1,k+1]-self.U[1,k])**2 + self.Rv*(self.NU[k+1]-self.NU[k])**2

        # Defining the state dynamics constraints
        for k in range(self.H):
            # Compute the derivative (state dynamics) at the current time step
            state_derivative = self.dynamics.ode_dyn(self.X[:, k], self.U[:, k])
            # Update the state using RK4
            new_state = self.dynamics.rk4_step(self.X[:, k], self.U[:, k])
            self.MPC.subject_to(self.X[:, k+1] == new_state)
            self.MPC.subject_to(self.TH[k+1] == self.TH[k] + self.T*self.NU[k])


        if self.spawn_obstacles:
            # Defining the obstacle avoidance constraints
            for k in range(self.H + 1):
                print(len(self.obstacle_positions))
                for idx, (obs_x, obs_y) in enumerate(self.obstacle_positions):
                    obs_radius = self.obstacle_radii[idx]
                    dist_sq = (self.X[0, k] - obs_x)**2 + (self.X[1, k] - obs_y)**2 - (0.5*obs_radius+self.vehicle_params.geometry.radius)**2
                    margin = 0.1
                    J += 0.25*ca.if_else(dist_sq>=margin, -ca.log(dist_sq),  0.5 * (((dist_sq - 2 * margin) / margin)**2 - 1) - ca.log(margin)) 

        # Defining the road boundary constraints
        self.MPC.subject_to(-4 +self.vehicle_params.geometry.radius <= self.waypoints.e_c(self.X[0,k], self.X[1,k], self.TH[k]))
        self.MPC.subject_to(self.waypoints.e_c(self.X[0,k], self.X[1,k], self.TH[k]) <= 4 - self.vehicle_params.geometry.radius )

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

        p_opts = {'verbose_init': False,'jit': True}
        s_opts = {'tol': 1e-1, 'print_level': 0, 'max_iter': 20}
        rospy.logwarn("INTIALIZING SOLVER...")
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
            self.arc = self.MPC.debug.value(self.TH)[0]


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
            self.arc = sol.value(self.TH)[0]

        self.desired_delta = state[2,1]
        self.desired_v = state[3,1]
        self.delta_dot = control[0,0]
        self.acc = control[1,0]

    def run(self):
        if self.arc < self.waypoints.L - 10:
            self.solve_ocp()
            self.ackermann_msg.acceleration = self.acc
            self.ackermann_msg.steering_angle_velocity = self.delta_dot
            self.ackermann_msg.steering_angle = self.desired_delta
            self.ackermann_msg.speed = self.desired_v
        else:
            self.ackermann_msg.speed = 0.0
            self.ackermann_msg.steering_angle = 0.0 

        self.ackermann_pub.publish(self.ackermann_msg)

def main():
    rospy.init_node('mpc_node')
    spawn_obstacles = rospy.get_param('/spawn_obstacles', True)
    mpc_controller = MPCC(spawn_obstacles)
    rospy.sleep(1)
    rate = rospy.Rate(20)
    mpc_controller.set_cost_params(qc=0.1, 
                                   ql=0.1, 
                                   Ru = np.array([10, 10]),
                                   Rv = 0.1, 
                                   Rx = np.array([0,0,0,0,1,0,1]),
                                   gamma=0.1)
    mpc_controller.formulate_ocp()
    while not rospy.is_shutdown():
        mpc_controller.run()
        rate.sleep()

if __name__ == '__main__':
    main()
