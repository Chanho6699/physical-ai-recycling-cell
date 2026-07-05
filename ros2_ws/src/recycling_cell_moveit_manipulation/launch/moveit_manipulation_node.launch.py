from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .to_moveit_configs()
    )

    moveit_manipulation_node = Node(
        package="recycling_cell_moveit_manipulation",
        executable="moveit_manipulation_node",
        name="moveit_manipulation_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
        ],
    )

    return LaunchDescription([
        moveit_manipulation_node,
    ])
