from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    force_pick_failure_arg = DeclareLaunchArgument(
        'force_pick_failure',
        default_value='false',
        description='Force fake_manipulation_node to fail every pick goal',
    )

    force_place_failure_arg = DeclareLaunchArgument(
        'force_place_failure',
        default_value='false',
        description='Force fake_manipulation_node to fail every place goal',
    )

    min_graspability_arg = DeclareLaunchArgument(
        'min_graspability',
        default_value='0.5',
        description='Minimum graspability_score required for a pick to succeed',
    )

    max_pick_retries_arg = DeclareLaunchArgument(
        'max_pick_retries',
        default_value='2',
        description='Maximum number of pick retries in task_manager_node',
    )

    max_place_retries_arg = DeclareLaunchArgument(
        'max_place_retries',
        default_value='1',
        description='Maximum number of place retries in task_manager_node',
    )

    force_pick_failure = LaunchConfiguration('force_pick_failure')
    force_place_failure = LaunchConfiguration('force_place_failure')
    min_graspability = LaunchConfiguration('min_graspability')
    max_pick_retries = LaunchConfiguration('max_pick_retries')
    max_place_retries = LaunchConfiguration('max_place_retries')

    fake_manipulation_node = Node(
        package='recycling_cell_manipulation',
        executable='fake_manipulation_node',
        name='fake_manipulation_node',
        output='screen',
        parameters=[{
            'force_pick_failure': force_pick_failure,
            'force_place_failure': force_place_failure,
            'min_graspability': min_graspability,
        }],
    )

    fake_perception_node = Node(
        package='recycling_cell_perception',
        executable='fake_perception_node',
        name='fake_perception_node',
        output='screen',
    )

    task_manager_node = Node(
        package='recycling_cell_task_manager',
        executable='task_manager_node',
        name='task_manager_node',
        output='screen',
        parameters=[{
            'max_pick_retries': max_pick_retries,
            'max_place_retries': max_place_retries,
        }],
    )

    return LaunchDescription([
        force_pick_failure_arg,
        force_place_failure_arg,
        min_graspability_arg,
        max_pick_retries_arg,
        max_place_retries_arg,
        fake_manipulation_node,
        fake_perception_node,
        task_manager_node,
    ])
