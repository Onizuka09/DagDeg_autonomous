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
    
    # 1. Zetta Bridge Node (connects to real STM32)
    zetta_bridge = Node(
        package='dagdeg_nav',
        executable='zetta_bridge',
        name='zetta_bridge',
        output='screen',
        parameters=[{
            'port': '/dev/ttyS0',
            'baudrate': 115200
        }]
    )
    
    # 2. Robot State Publisher (visualizes robot model)
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
    
    # 3. Joint State Publisher (CRITICAL - publishes joint positions)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'rate': 50  # 50 Hz update rate
        }]
    )
    
    # 4. RViz for visualization
    rviz_config_path = os.path.join(pkg_path, 'config', 'real_robot.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )
    
    return LaunchDescription([
        zetta_bridge,
        joint_state_publisher,  # ADD THIS BEFORE robot_state_publisher
        robot_state_publisher,
        # rviz_node
    ])