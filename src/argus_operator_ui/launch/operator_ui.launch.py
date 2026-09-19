from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="argus_operator_ui", executable="argus_operator_ui",
             name="argus_operator_ui", output="screen", parameters=[{
                 "image_topic": "/camera/image/compressed",
                 "result_topic": "/camera/inference/result",
                 "qwen_action_name": "/camera/inference/qwen",
             }])
    ])
