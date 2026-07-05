import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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

    enable_onnx_inference_arg = DeclareLaunchArgument(
        'enable_onnx_inference',
        default_value='false',
        description='Run ONNX Runtime YOLO inference instead of mock detections',
    )

    onnx_model_path_arg = DeclareLaunchArgument(
        'onnx_model_path',
        default_value='',
        description='Path to a .onnx YOLO model, used when enable_onnx_inference=true',
    )

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Minimum ONNX detection confidence to keep',
    )

    onnx_input_size_arg = DeclareLaunchArgument(
        'onnx_input_size',
        default_value='640',
        description='Square input resolution the ONNX model expects',
    )

    summary_period_sec_arg = DeclareLaunchArgument(
        'summary_period_sec',
        default_value='5.0',
        description='Seconds between monitor_node [CellMetrics] summaries',
    )

    image_folder_path_arg = DeclareLaunchArgument(
        'image_folder_path',
        default_value='',
        description='Directory to scan, used when image_source=image_folder',
    )

    image_extensions_arg = DeclareLaunchArgument(
        'image_extensions',
        default_value='.jpg,.jpeg,.png',
        description='Comma-separated file extensions to include when scanning image_folder_path',
    )

    loop_folder_arg = DeclareLaunchArgument(
        'loop_folder',
        default_value='false',
        description='Restart from the first image after the last one, instead of stopping',
    )

    publish_once_per_image_arg = DeclareLaunchArgument(
        'publish_once_per_image',
        default_value='true',
        description='Advance to the next image every tick instead of dwelling on the current one',
    )

    known_confidence_threshold_arg = DeclareLaunchArgument(
        'known_confidence_threshold',
        default_value='0.50',
        description='Minimum confidence required for a known-class object to be a pick candidate',
    )

    graspability_threshold_arg = DeclareLaunchArgument(
        'graspability_threshold',
        default_value='0.30',
        description='Below this, a known object still becomes a candidate but logs a low-graspability warning',
    )

    absolute_min_graspability_arg = DeclareLaunchArgument(
        'absolute_min_graspability',
        default_value='0.05',
        description='Hard floor below which a known object is skipped as physically too unstable to pick',
    )

    route_unknown_to_reject_bin_arg = DeclareLaunchArgument(
        'route_unknown_to_reject_bin',
        default_value='true',
        description='Route unknown/is_unknown detections to reject_bin instead of skipping them',
    )

    min_unknown_confidence_arg = DeclareLaunchArgument(
        'min_unknown_confidence',
        default_value='0.50',
        description='Minimum confidence required for an unknown detection to be routed to reject_bin',
    )

    min_unknown_graspability_arg = DeclareLaunchArgument(
        'min_unknown_graspability',
        default_value='0.10',
        description='Minimum graspability_score required for an unknown detection to be '
                     'routed to reject_bin (separate from the known-object threshold, since '
                     'perception nodes deliberately score unknown detections low)',
    )

    image_source = LaunchConfiguration('image_source')
    image_path = LaunchConfiguration('image_path')
    onnx_model_path = LaunchConfiguration('onnx_model_path')
    image_folder_path = LaunchConfiguration('image_folder_path')
    image_extensions = LaunchConfiguration('image_extensions')

    publish_period_sec = ParameterValue(
        LaunchConfiguration('publish_period_sec'), value_type=float)
    camera_index = ParameterValue(
        LaunchConfiguration('camera_index'), value_type=int)
    enable_onnx_inference = ParameterValue(
        LaunchConfiguration('enable_onnx_inference'), value_type=bool)
    confidence_threshold = ParameterValue(
        LaunchConfiguration('confidence_threshold'), value_type=float)
    onnx_input_size = ParameterValue(
        LaunchConfiguration('onnx_input_size'), value_type=int)
    summary_period_sec = ParameterValue(
        LaunchConfiguration('summary_period_sec'), value_type=float)
    loop_folder = ParameterValue(
        LaunchConfiguration('loop_folder'), value_type=bool)
    publish_once_per_image = ParameterValue(
        LaunchConfiguration('publish_once_per_image'), value_type=bool)
    known_confidence_threshold = ParameterValue(
        LaunchConfiguration('known_confidence_threshold'), value_type=float)
    graspability_threshold = ParameterValue(
        LaunchConfiguration('graspability_threshold'), value_type=float)
    absolute_min_graspability = ParameterValue(
        LaunchConfiguration('absolute_min_graspability'), value_type=float)
    route_unknown_to_reject_bin = ParameterValue(
        LaunchConfiguration('route_unknown_to_reject_bin'), value_type=bool)
    min_unknown_confidence = ParameterValue(
        LaunchConfiguration('min_unknown_confidence'), value_type=float)
    min_unknown_graspability = ParameterValue(
        LaunchConfiguration('min_unknown_graspability'), value_type=float)

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
                    'enable_onnx_inference': enable_onnx_inference,
                    'onnx_model_path': onnx_model_path,
                    'confidence_threshold': confidence_threshold,
                    'onnx_input_size': onnx_input_size,
                    'image_folder_path': image_folder_path,
                    'image_extensions': image_extensions,
                    'loop_folder': loop_folder,
                    'publish_once_per_image': publish_once_per_image,
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
                parameters=[{
                    'known_confidence_threshold': known_confidence_threshold,
                    'graspability_threshold': graspability_threshold,
                    'absolute_min_graspability': absolute_min_graspability,
                    'route_unknown_to_reject_bin': route_unknown_to_reject_bin,
                    'min_unknown_confidence': min_unknown_confidence,
                    'min_unknown_graspability': min_unknown_graspability,
                }],
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
        enable_onnx_inference_arg,
        onnx_model_path_arg,
        confidence_threshold_arg,
        onnx_input_size_arg,
        summary_period_sec_arg,
        image_folder_path_arg,
        image_extensions_arg,
        loop_folder_arg,
        publish_once_per_image_arg,
        known_confidence_threshold_arg,
        graspability_threshold_arg,
        absolute_min_graspability_arg,
        route_unknown_to_reject_bin_arg,
        min_unknown_confidence_arg,
        min_unknown_graspability_arg,
        panda_demo,
        moveit_manipulation_node,
        vision_perception_node,
        task_manager_node,
        monitor_node,
    ])
