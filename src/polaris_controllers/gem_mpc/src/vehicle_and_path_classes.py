# Python Packages
import numpy as np
from numpy import linalg as LA
import casadi as ca
import math
from dataclasses import dataclass

"""
This file contains the classes which define the tire parameters, vehicle parameters (dynamic and geometric).
"""
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
        self.x_interpolant = None
        self.y_interpolant = None
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
    
    def calculate_cumulative_path_length(self,x_points, y_points):

        distances = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
        cumulative_length = np.zeros(len(x_points))
        cumulative_length[1:] = np.cumsum(distances)
        total_length = np.sum(distances)
        
        return total_length, cumulative_length

    def cubic_spline(self, step):
        """
        Generate a b-spline interpolant for the path points.
        """
        x_list = self.path_points_x
        y_list = self.path_points_y

        self.L, self.l_list = self.calculate_cumulative_path_length(x_list,y_list)

        self.x_interpolant = ca.interpolant('x_interpolant','bspline',[self.l_list],x_list)
        self.y_interpolant = ca.interpolant('y_interpolant','bspline',[self.l_list],y_list)
        th = ca.MX.sym('th')

        self.Phi = ca.Function('Phi', [th], [ca.atan2(ca.jacobian(self.y_interpolant(th),th),(ca.jacobian(self.x_interpolant(th),th)))])

        X = ca.MX.sym('X')
        Y = ca.MX.sym('Y')
        th = ca.MX.sym('th')

        self.e_c = ca.Function('e_c', [X, Y, th], [ca.sin(self.Phi(th))*(X - self.x_interpolant(th)) - ca.cos(self.Phi(th))*(Y - self.y_interpolant(th))])
        self.e_l = ca.Function('e_l', [X, Y, th], [-ca.cos(self.Phi(th))*(X - self.x_interpolant(th)) - ca.sin(self.Phi(th))*(Y - self.y_interpolant(th))]) 

        self.x_samples = np.array([self.x_interpolant(l).full().item() for l in self.l_list])
        self.y_samples = np.array([self.y_interpolant(l).full().item() for l in self.l_list])

    def find_closest_point_discrete(self, target_x, target_y, previous_index=None, search_window=200):
        # Limit the search to a window of 200 points around the previous closest index
        if previous_index is not None:
            start_index = max(0, previous_index - search_window // 2)
            end_index = min(len(self.l_list), previous_index + search_window // 2)
            search_l_list = self.l_list[start_index:end_index]
            base_index = start_index  # To adjust the index to the global index
        else:
            search_l_list = self.l_list
            base_index = 0
        
        # Sample the x and y points along the limited arc length
        x_samples = np.array([self.x_interpolant(l).full().item() for l in search_l_list])
        y_samples = np.array([self.y_interpolant(l).full().item() for l in search_l_list])
        
        # Calculate the Euclidean distance between each sampled point and the target point (target_x, target_y)
        distances = np.sqrt((x_samples - target_x)**2 + (y_samples - target_y)**2)
        
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
        
        # Get the arc length corresponding to the closest point
        l_closest = search_l_list[min_index]
        
        # Get the (x, y) coordinates of the closest point
        x_closest = x_samples[min_index]
        y_closest = y_samples[min_index]
        
        # Return the global index of the closest point
        closest_index = base_index + min_index
        
        return closest_index, l_closest

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