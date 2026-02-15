from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get path to URDF file
    pkg_path = get_package_share_directory('dagdeg_nav')
    urdf_file = os.path.join(pkg_path, 'urdf', 'dagdeg_bot.urdf')
    
    # Read URDF contents
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    # Create nodes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False
        }]
    )
    
    joint_state_publisher = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )
    
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_path, 'config', 'robot_display.rviz')]
    )
    
    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        rviz2
    ])