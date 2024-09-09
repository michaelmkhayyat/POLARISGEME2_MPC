import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_path_from_csv(file_path):
    """Reads the path from a CSV file."""
    path_data = pd.read_csv(file_path, header=None)
    x = path_data.iloc[:, 0].values
    y = path_data.iloc[:, 1].values
    return x, y

def compute_normals(x, y):
    """Computes the normals for each point on the path."""
    tangents = np.diff(np.column_stack((x, y)), axis=0)
    tangents = np.vstack([tangents, tangents[-1]])  # Repeat last tangent for shape consistency
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    tangents_normalized = tangents / tangent_lengths[:, None]
    normals = np.column_stack((-tangents_normalized[:, 1], tangents_normalized[:, 0]))
    return normals

def compute_boundaries(x, y, normals, width):
    """Computes the left and right boundaries of the path."""
    left_boundary_x = x + width * normals[:, 0]
    left_boundary_y = y + width * normals[:, 1]
    right_boundary_x = x - width * normals[:, 0]
    right_boundary_y = y - width * normals[:, 1]
    return left_boundary_x, left_boundary_y, right_boundary_x, right_boundary_y

def save_to_csv_file(filename, points):
    """Saves the points to a CSV file."""
    df = pd.DataFrame(points, columns=["x", "y"])
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

def plot_path_with_boundaries(x, y, left_x, left_y, right_x, right_y):
    """Plots the path with left and right boundaries."""
    plt.plot(x, y, label='Center Path', color='blue', marker='o')
    plt.plot(left_x, left_y, label='Left Boundary', linestyle='--', color='red', marker='x')
    plt.plot(right_x, right_y, label='Right Boundary', linestyle='--', color='green', marker='x')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path with Left and Right Boundaries')
    plt.grid(True)
    plt.show()

# Parameters
boundary_width = 4.5  # Distance from the path to the boundary
x, y = read_path_from_csv('path.csv')
normals = compute_normals(x, y)
left_x, left_y, right_x, right_y = compute_boundaries(x, y, normals, boundary_width)
center_points = np.column_stack((x, y))
left_boundary_points = np.column_stack((left_x, left_y))
right_boundary_points = np.column_stack((right_x, right_y))
save_to_csv_file("center_sample_points.csv", center_points)
save_to_csv_file("left_boundary_points.csv", left_boundary_points)
save_to_csv_file("right_boundary_points.csv", right_boundary_points)

plot_path_with_boundaries(x, y, left_x, left_y, right_x, right_y)

