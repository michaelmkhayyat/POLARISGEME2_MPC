import csv
import os
import time
import rospy
from gazebo_ros import gazebo_interface
from geometry_msgs.msg import Point, Pose, Quaternion
from tf.transformations import quaternion_from_euler

def load_model_template(filepath):
    try:
        with open(filepath, 'r') as file:
            model_template = file.read()
        return model_template
    except (FileNotFoundError, IOError) as e:
        rospy.logerr(f"Error loading model template from {filepath}: {str(e)}")
        return None

def spawn(model_name, size, positions, orientations, static=True):
    model_template_path = os.path.join(os.path.dirname(__file__), "../models/custom_cube/model.sdf")
    model_xml = load_model_template(model_template_path)
    
    if model_xml is None:
        rospy.logerr("Failed to load model template.")
        return None
    
    try:
        rospy.wait_for_service('/gazebo/spawn_sdf_model')

        # Replace placeholders in the model template
        model_xml = model_xml.replace("%MODEL_NAME%", model_name) \
                             .replace("%STATIC%", str(int(static))) \
                             .replace("%SIZE%", str(size))

        initial_pose = Pose(Point(*positions), Quaternion(*quaternion_from_euler(*orientations)))

        # Create unique model name with timestamp
        gazebo_model_name = f"{model_name}_{int(time.time() * 1000)}"

        rospy.loginfo(f"Model XML:\n{model_xml}")
        rospy.loginfo(f"Spawning model {gazebo_model_name} at position {positions} with orientation {orientations}")

        # Spawn the model in Gazebo
        gazebo_interface.spawn_sdf_model_client(
            gazebo_model_name, model_xml, "/", initial_pose, "", "/gazebo"
        )

        rospy.loginfo(f"Spawned {model_name} in Gazebo as {gazebo_model_name}")
        return gazebo_model_name
    except Exception as e:
        rospy.logerr(f"Failed to spawn model {model_name}: {str(e)}")
        return None

def load_obstacles(filename):
    """Load obstacles from CSV file."""
    try:
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            obstacles = [
                (float(row[0]), float(row[1]), float(row[2])) for row in reader
            ]
        return obstacles
    except (FileNotFoundError, ValueError) as e:
        rospy.logerr(f"Error loading obstacles from {filename}: {str(e)}")
        return []

def main():
    rospy.init_node("spawner", anonymous=True)
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../data/obstacles.csv")
    obstacles = load_obstacles(filename)

    if obstacles:
        for o in obstacles:
            spawn("obstacle", o[2], [o[0], o[1], 0.5], [0, 0, 0], static=True)
    else:
        rospy.logwarn("No obstacles to spawn.")

if __name__ == "__main__":
    main()
