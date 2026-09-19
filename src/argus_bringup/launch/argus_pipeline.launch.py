from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
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
                package='qwen_description',
                plugin='qwen_description::QwenDescriptionNode',
                name='qwen_description_node',
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
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )
    return LaunchDescription([
        SetEnvironmentVariable(
            name='EDGELLM_PLUGIN_PATH',
            value='/home/royfan/edge-llm-v0.10.1/build/libNvInfer_edgellm_plugin.so'),
        container,
    ])
