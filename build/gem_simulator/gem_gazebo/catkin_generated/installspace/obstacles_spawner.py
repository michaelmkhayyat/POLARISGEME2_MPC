import csv
import os
import time
import rospy
from gazebo_ros import gazebo_interface
from geometry_msgs.msg import Point, Pose, Quaternion
from tf.transformations import quaternion_from_euler

# Template for spawning obstacles in Gazebo
MODEL_TEMPLATE = """<sdf version="1.4">
    <model name="%MODEL_NAME%">
        <static>%STATIC%</static>
        <link name="link">
            <inertial>
                <mass>1.0</mass>
                <inertia>
                    <ixx>0.083</ixx>
                    <ixy>0.0</ixy>
                    <ixz>0.0</ixz>
                    <iyy>0.083</iyy>
                    <iyz>0.0</iyz>
                    <izz>0.083</izz>
                </inertia>
            </inertial>
            <collision name="collision">
                <geometry>
                    <cylinder>
                        <radius>%RADIUS%</radius>
                        <length>%LENGTH%</length>
                    </cylinder>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <cylinder>
                        <radius>%RADIUS%</radius>
                        <length>%LENGTH%</length>
                    </cylinder>
                </geometry>
            </visual>
        </link>
    </model>
</sdf>"""


def spawn(model_name, radius, positions, orientations, static=True):
    try:
        rospy.wait_for_service('/gazebo/spawn_sdf_model')

        model_xml = MODEL_TEMPLATE.replace("%MODEL_NAME%", model_name) \
                                  .replace("%STATIC%", str(int(static))) \
                                  .replace("%RADIUS%", str(radius)) \
                                  .replace("%LENGTH%", "1")

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
            next(reader)
            obstacles = [
                (float(row[0]), float(row[1]), float(row[2])) for row in reader
            ]
        return obstacles
    except (FileNotFoundError, ValueError) as e:
        rospy.logerr(f"Error loading obstacles from {filename}: {str(e)}")
        return []


def main():
    rospy.init_node("spawner", anonymous=True)
    
    # Path to the CSV file
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../data/obstacles.csv")

    # Load obstacles from the file
    obstacles = load_obstacles(filename)

    if obstacles:
        for o in obstacles:
            spawn("obstacle", o[2], [o[0], o[1], 5], [0, 0, 0], static=True)
    else:
        rospy.logwarn("No obstacles to spawn.")
