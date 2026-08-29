from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    camera_parameters = {
        'camera_index': ParameterValue(LaunchConfiguration('camera_index'), value_type=int),
        'sensor_mode_index': ParameterValue(
            LaunchConfiguration('sensor_mode_index'), value_type=int),
        'frame_count': ParameterValue(LaunchConfiguration('frame_count'), value_type=int),
        'capture_buffer_count': ParameterValue(
            LaunchConfiguration('capture_buffer_count'), value_type=int),
        'frame_rate': ParameterValue(LaunchConfiguration('frame_rate'), value_type=float),
        'frame_id': ParameterValue(LaunchConfiguration('frame_id'), value_type=str),
    }
    container = ComposableNodeContainer(
        name='argus_pipeline_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='argus_inference',
                plugin='argus_inference::InferenceNode',
                name='yuv_inference_node',
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='argus_visualization',
                plugin='argus_visualization::VisualizationNode',
                name='yuv_visualization_node',
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='argus_camera',
                plugin='argus_camera::ArgusCameraNode',
                name='argus_camera_node',
                parameters=[camera_parameters],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )
    return LaunchDescription([
        DeclareLaunchArgument('camera_index', default_value='0'),
        DeclareLaunchArgument('sensor_mode_index', default_value='4'),
        DeclareLaunchArgument('frame_count', default_value='0'),
        DeclareLaunchArgument('capture_buffer_count', default_value='4'),
        DeclareLaunchArgument('frame_rate', default_value='0.0'),
        DeclareLaunchArgument('frame_id', default_value='camera'),
        container,
    ])
