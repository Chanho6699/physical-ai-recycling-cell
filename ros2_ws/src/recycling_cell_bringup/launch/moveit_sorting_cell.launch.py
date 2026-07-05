import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    panda_demo_launch = os.path.join(
        get_package_share_directory('moveit_resources_panda_moveit_config'),
        'launch',
        'demo.launch.py',
    )

    moveit_manipulation_launch = os.path.join(
        get_package_share_directory('recycling_cell_moveit_manipulation'),
        'launch',
        'moveit_manipulation_node.launch.py',
    )

    # demo.launch.py brings up move_group + ros2_control + RViz, which takes
    # several seconds to finish loading controllers. moveit_manipulation_node
    # is delayed so its MoveGroupInterface finds a fully initialized
    # move_group action server on first try instead of racing it.
    panda_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(panda_demo_launch),
    )

    moveit_manipulation_node = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(moveit_manipulation_launch),
            ),
        ],
    )

    # fake_perception_node/task_manager_node are delayed further so they only
    # start sending goals once moveit_manipulation_node's action servers are
    # actually registered.
    fake_perception_node = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='recycling_cell_perception',
                executable='fake_perception_node',
                name='fake_perception_node',
                output='screen',
            ),
        ],
    )

    task_manager_node = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='recycling_cell_task_manager',
                executable='task_manager_node',
                name='task_manager_node',
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        panda_demo,
        moveit_manipulation_node,
        fake_perception_node,
        task_manager_node,
    ])
