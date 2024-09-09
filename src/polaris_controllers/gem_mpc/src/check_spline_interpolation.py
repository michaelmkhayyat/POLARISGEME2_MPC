#!/usr/bin/env python

import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt

class SplineApproximation:
    def __init__(self, csv_file):
        # Read waypoints from the CSV file
        self.waypoints = pd.read_csv(csv_file, header=None)
        self.x_points = self.waypoints.iloc[:, 0].values
        self.y_points = self.waypoints.iloc[:, 1].values

    def generate_spline(self, step=100):
        """Generate a spline using CasADi's bspline interpolation."""
        self.num_points = len(self.x_points)
        index_list = np.arange(0, self.num_points, 1)
        
        # Create CasADi bspline interpolants
        self.cs_x = ca.interpolant('cs_x', 'bspline', [index_list[::step]], self.x_points[::step])
        self.cs_y = ca.interpolant('cs_y', 'bspline', [index_list[::step]], self.y_points[::step])

    def plot_spline_vs_path(self, num_samples=500):
        """Plot the original path vs the spline approximation."""
        fig, ax = plt.subplots()

        # Original waypoints
        ax.plot(self.x_points, self.y_points, 'o-', label='Original Path', markersize=5)

        # Sample points along the spline
        spline_param = np.linspace(0, self.num_points - 1, num_samples)
        spline_x = np.array([self.cs_x(p) for p in spline_param]).reshape(-1)  # Reshape to 1D array
        spline_y = np.array([self.cs_y(p) for p in spline_param]).reshape(-1)  # Reshape to 1D array

        # Plot the spline approximation
        ax.plot(spline_x, spline_y, 'r-', label='Spline Approximation', linewidth=2)

        ax.set_title("Spline Approximation vs Original Path")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Initialize the class and provide the path to your CSV file
    csv_file = "~/Desktop/POLARISGEME2_MPC/src/gem_simulator/gem_gazebo/data/waypoints.csv"
    
    # Create a spline approximation instance
    spline_approx = SplineApproximation(csv_file)

    # Generate the spline approximation from the waypoints
    spline_approx.generate_spline(step=40)  # Adjust step size if needed

    # Plot and compare the original path and spline approximation
    spline_approx.plot_spline_vs_path(num_samples=500)  # Adjust num_samples if needed
