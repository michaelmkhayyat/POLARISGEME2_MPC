#!/usr/bin/env python3

import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import casadi as ca

class SimulationPlotter:
    def __init__(self, base_dir=os.path.dirname(__file__)):
        current_dir = os.path.dirname(__file__)
        self.log_file = os.path.join(current_dir, "./mpc_data/mpc_data.csv")
        self.flag_file = os.path.join(current_dir, "./mpc_data/flag.txt")
        self.waypoints_file = os.path.join(current_dir, "../../../gem_simulator/gem_gazebo/data/waypoints.csv")
        self.obstacles_file = os.path.join(current_dir, "../../../gem_simulator/gem_gazebo/data/obstacles.csv")
        self.left_boundary_file = os.path.join(current_dir, "../../../gem_simulator/gem_gazebo/data/left_boundary_points.csv")
        self.right_boundary_file = os.path.join(current_dir, "../../../gem_simulator/gem_gazebo/data/right_boundary_points.csv")
        self.output_dir = os.path.join(current_dir, "./plots")

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        
        with open(self.flag_file, 'r') as flag_file:
                # Read the first line and extract the flag value
                line = flag_file.readline().strip()
                flag_value = line.split(":")[1].strip()

                # Convert the flag to an integer
                self.spawn_obstacles = int(flag_value)

        # Load data
        self.logs = self.load_csv(self.log_file)
        self.waypoints = self.load_csv(self.waypoints_file)
        if self.spawn_obstacles:
            self.obstacles = self.load_csv(self.obstacles_file)
        else:
            self.obstacles = []
        self.left_boundary = self.load_csv(self.left_boundary_file)
        self.right_boundary = self.load_csv(self.right_boundary_file)
        self.wp_len = 0
        self.create_spline_interpolant() 

    def load_csv(self, file_path):
        """
        Load a CSV file and return the data as a numpy array.
        """
        file_path = os.path.expanduser(file_path)
        return np.genfromtxt(file_path, delimiter=',', skip_header=1)

    def plot_states_over_time(self):
        """
        Plot vehicle states (vx, vy, v, beta, yaw, yaw_rate) over time in a grid layout and save as .png.
        """
        # Convert rospy time into relative time
        time = self.logs[:, 0] - self.logs[0, 0]

        vx = self.logs[:, 3]
        vy = self.logs[:, 4]
        v = self.logs[:, 5]
        beta = self.logs[:, 6]
        yaw = self.logs[:, 7]
        yaw_rate = self.logs[:, 8]

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))

        # Plot vehicle states in a grid layout
        axs[0, 0].plot(time[::10], vx[::10], color="blue", linewidth=1.5)
        axs[0, 0].set_title("Longitudinal Velocity")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("vx (m/s)")
        axs[0, 0].grid(True)

        axs[0, 1].plot(time[::10], vy[::10],  color="green", linewidth=1.5)
        axs[0, 1].set_title("Lateral Velocity")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("vy (m/s)")
        axs[0, 1].grid(True)

        axs[1, 0].plot(time[::10], v[::10],  color="purple", linewidth=1.5)
        axs[1, 0].set_title("Absolute Speed")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("v (m/s)")
        axs[1, 0].grid(True)

        axs[1, 1].plot(time[::10], beta[::10],  color="orange", linewidth=1.5)
        axs[1, 1].set_title("Slip Angle")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("beta (rad)")
        axs[1, 1].grid(True)

        axs[2, 0].plot(time[::10], yaw[::10],  color="red", linewidth=1.5)
        axs[2, 0].set_title("Yaw")
        axs[2, 0].set_xlabel("Time (s)")
        axs[2, 0].set_ylabel("yaw (rad)")
        axs[2, 0].grid(True)

        axs[2, 1].plot(time[::10], yaw_rate[::10], color="magenta", linewidth=1.5)
        axs[2, 1].set_title("Yaw Rate")
        axs[2, 1].set_xlabel("Time (s)")
        axs[2, 1].set_ylabel("yaw rate (rad/s)")
        axs[2, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "vehicle_states_over_time.png"))
        plt.show()
    
    def create_spline_interpolant(self):
        """
        Create cubic spline interpolant for the path points.
        The steps here should match the steps in the MPC script to make sure the 
        evalution is meaningful.
        """
        path_x = self.waypoints[:, 0]
        path_y = self.waypoints[:, 1]

        self.wp_len = len(path_x)
        step = 40 
        l_list = np.arange(0, self.wp_len, 1)
        self.cs_x = ca.interpolant('cs_x', 'bspline', [l_list[::step]], path_x[::step])
        self.cs_y = ca.interpolant('cs_y', 'bspline', [l_list[::step]], path_y[::step])

        th = ca.MX.sym('th')
        self.Phi = ca.Function('Phi', [th], [ca.atan2(ca.jacobian(self.cs_y(th), th), ca.jacobian(self.cs_x(th), th))])
        X = ca.MX.sym('X')
        Y = ca.MX.sym('Y')
        th = ca.MX.sym('th')

        self.e_c = ca.Function('e_c', [X, Y, th], [ca.sin(self.Phi(th))*(X - self.cs_x(th)) - ca.cos(self.Phi(th))*(Y - self.cs_y(th))])
        self.e_l = ca.Function('e_l', [X, Y, th], [-ca.cos(self.Phi(th))*(X - self.cs_x(th)) - ca.sin(self.Phi(th))*(Y - self.cs_y(th))]) 

    def compute_cross_track_error_from_spline(self):
        """
        Compute cross-track error (cte) using the spline interpolant.
        """
        ctes = []
        x = self.logs[:, 1]  # Vehicle's actual x positions
        y = self.logs[:, 2]  # Vehicle's actual y positions
        num_samples = self.wp_len
        spline_param = np.linspace(0, num_samples - 1, num_samples)
        spline_x = np.array([self.cs_x(p) for p in spline_param]) # Reshape to 1D array
        spline_y = np.array([self.cs_y(p) for p in spline_param])  # Reshape to 1D array
        print(spline_x)
        for i in range(len(x)):
            distances = np.hypot(x[i] - spline_x, y[i] - spline_y)
            print(np.min(distances))
            ctes.append(np.min(distances))  # Closest waypoint
        return ctes

    def compute_cross_track_error(self):
        """
        Compute cross-track error (cte) using waypoints.
        """
        x = self.logs[:, 1]
        y = self.logs[:, 2]
        waypoints_x = self.waypoints[:, 0]
        waypoints_y = self.waypoints[:, 1]

        cte = []
        for i in range(len(x)):
            distances = np.hypot(x[i] - waypoints_x, y[i] - waypoints_y)
            cte.append(np.min(distances))

        return cte
    
    def plot_cross_track_error_from_spline(self):
        """
        Plot cross-track error calculated from spline over time.
        """
        time = self.logs[:, 0] - self.logs[0, 0]  # Convert to relative time
        self.ctes = self.compute_cross_track_error_from_spline()

        plt.figure()
        plt.plot(time[::10], self.ctes[::10], color="dodgerblue", linewidth=1.5)
        plt.axhline(y=1, color='r', linestyle='--', label='Error threshold')
        plt.title("Cross-Track Error (Using Spline) Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Cross-Track Error (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cross_track_error_from_spline.png"))
        plt.show()

    def plot_cross_track_error(self):
        """
        Plot the cross-track error over time and save as .png.
        """
        time = self.logs[:, 0] - self.logs[0, 0]  # Convert to relative time
        self.cte = self.compute_cross_track_error()

        plt.figure()
        plt.plot(time[::10], self.cte[::10], color="dodgerblue", linewidth=1.5)
        plt.title("Cross-Track Error Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Cross-Track Error (m)")
        plt.axhline(y=1, color='r', linestyle='--', label='Error threshold')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cross_track_error.png"))
        plt.show()

    def plot_waypoints_obstacles_and_trajectory(self):
        """
        Plot waypoints, obstacles, boundaries, and the vehicle trajectory, and save as .png.
        """
        # Waypoints
        waypoints_x = self.waypoints[:, 0]
        waypoints_y = self.waypoints[:, 1]


        # Left and right boundaries
        left_boundary_x = self.left_boundary[:, 0]
        left_boundary_y = self.left_boundary[:, 1]
        right_boundary_x = self.right_boundary[:, 0]
        right_boundary_y = self.right_boundary[:, 1]

        # Vehicle trajectory
        vehicle_x = self.logs[:, 1]
        vehicle_y = self.logs[:, 2]

        plt.figure()
        plt.plot(waypoints_x, waypoints_y, label="Waypoints", marker=".", markersize=0.1, color="crimson")
        plt.plot(left_boundary_x, left_boundary_y, label="Left Boundary", color="green", linestyle="--")
        plt.plot(right_boundary_x, right_boundary_y, label="Right Boundary", color="blue", linestyle="--")
        plt.plot(vehicle_x[::10], vehicle_y[::10], label="Vehicle Trajectory", color="orange")

        # Plot obstacles as circles.
        # Obstacles are plotted as circles not cubes because it is the collision avoidance
        # constraint
        if self.spawn_obstacles:
            obstacles_x = self.obstacles[:, 0]
            obstacles_y = self.obstacles[:, 1]
            obstacles_radius = self.obstacles[:, 2]
            for i in range(len(obstacles_x)):
                obstacle_circle = plt.Circle((obstacles_x[i], obstacles_y[i]), 0.5*obstacles_radius[i], color="gray", fill=True, alpha=0.5)
                plt.gca().add_patch(obstacle_circle)

        plt.axis("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "waypoints_obstacles_trajectory.png"))
        plt.show()
    def plot_cross_track_error_comparison(self):
        """
        Plot both cross-track errors (from spline and from waypoints) on the same plot over time.
        This will help us understand how well our MPC tracks the original waypoints and their approximation.
        This should also tell how good hte spline approximates the waypoints.
        """
        time = self.logs[:, 0] - self.logs[0, 0]  # Convert to relative time

        # Compute cross-track errors
        ctes_spline = self.ctes
        ctes_waypoints = self.cte

        # Create a single plot for both
        plt.figure()
        plt.plot(time[::10], ctes_spline[::10], label="Cross-Track Error (Spline)", color="dodgerblue", linewidth=1.5)
        plt.plot(time[::10], ctes_waypoints[::10], label="Cross-Track Error (Waypoints)", color="crimson", linestyle="--", linewidth=1.5)

        # Add title and labels
        plt.title("Cross-Track Error Comparison Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Cross-Track Error (m)")
        plt.axhline(y=1, color='r', linestyle='--', label='Error threshold')
        
        # Add grid and legend
        plt.grid(True)
        plt.legend()

        # Save the plot and display it
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cross_track_error_comparison.png"))
        plt.show()    
    def run(self):
        """
        Run all the plotting functions.
        """
        self.plot_states_over_time()
        self.plot_cross_track_error()
        self.plot_cross_track_error_from_spline()
        self.plot_waypoints_obstacles_and_trajectory()
        self.plot_cross_track_error_comparison()

if __name__ == '__main__':
    plotter = SimulationPlotter()
    plotter.run()

