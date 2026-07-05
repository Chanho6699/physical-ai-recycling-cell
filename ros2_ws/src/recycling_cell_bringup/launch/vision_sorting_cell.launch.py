import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_source_arg = DeclareLaunchArgument(
        'image_source',
        default_value='synthetic',
        description="vision_perception_node image source: "
                     "'synthetic', 'image_file', or 'camera'",
    )

    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value='',
        description='Path to a test image, used when image_source=image_file',
    )

    publish_period_sec_arg = DeclareLaunchArgument(
        'publish_period_sec',
        default_value='5.0',
        description='Seconds between vision_perception_node publish ticks',
    )

    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='OpenCV camera device index, used when image_source=camera',
    )

    summary_period_sec_arg = DeclareLaunchArgument(
        'summary_period_sec',
        default_value='10.0',
        description='Seconds between monitor_node [CellMetrics] summaries',
    )

    image_source = LaunchConfiguration('image_source')
    image_path = LaunchConfiguration('image_path')
    publish_period_sec = LaunchConfiguration('publish_period_sec')
    camera_index = LaunchConfiguration('camera_index')
    summary_period_sec = LaunchConfiguration('summary_period_sec')

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

    # vision_perception_node/task_manager_node/monitor_node are delayed
    # further so they only start once moveit_manipulation_node's action
    # servers are actually registered.
    vision_perception_node = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='recycling_cell_vision',
                executable='vision_perception_node',
                name='vision_perception_node',
                output='screen',
                parameters=[{
                    'image_source': image_source,
                    'image_path': image_path,
                    'publish_period_sec': publish_period_sec,
                    'camera_index': camera_index,
                }],
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

    monitor_node = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='recycling_cell_monitor',
                executable='monitor_node',
                name='monitor_node',
                output='screen',
                parameters=[{
                    'summary_period_sec': summary_period_sec,
                    'enable_detailed_result_log': True,
                }],
            ),
        ],
    )

    return LaunchDescription([
        image_source_arg,
        image_path_arg,
        publish_period_sec_arg,
        camera_index_arg,
        summary_period_sec_arg,
        panda_demo,
        moveit_manipulation_node,
        vision_perception_node,
        task_manager_node,
        monitor_node,
    ])
