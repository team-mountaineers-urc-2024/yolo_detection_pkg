import argparse
import os
from pathlib import Path  # noqa: E402
import sys

# Hack to get relative import of .camera_config file working
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '../yolo_models'))

from launch import LaunchDescription  # noqa: E402
from launch.actions import DeclareLaunchArgument  # noqa: E402
from launch_ros.actions import Node  # noqa: E402
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
# from launch.substitutions import PathJoinSubstitution, TextSubstitution

from ament_index_python.packages import get_package_share_directory

USB_CAM_DIR = get_package_share_directory('usb_cam')


def generate_launch_description():
    ld = LaunchDescription()

    yolo_node = Node(
        package='yolo_detection_pkg', executable='yolo_node', output='screen',
        name='yolo_node',
        # namespace='',
        parameters=[
            {'image_topic' : LaunchConfiguration('image_topic')},
            {'calculation_freq' : LaunchConfiguration('calculation_freq')},
            {'confidence_threshold' : LaunchConfiguration('confidence_threshold')},
            {'model' : PathJoinSubstitution([FindPackageShare('yolo_detection_pkg'), 'yolo_models', LaunchConfiguration('model')])},
            {'ui_feedback' : LaunchConfiguration('ui_feedback')}
        ]
    )

    # What is the image topic
    ld.add_action(DeclareLaunchArgument(name='image_topic', default_value='image_raw',
                                        description='The Image topic YOLO will subscribe to'))
    
    # What is the calculaiton frequency
    ld.add_action(DeclareLaunchArgument(name='calculation_freq', default_value='60.0',
                                        description='The frequency at which images will be pulled'))
    
    # What is the confidence threshold
    ld.add_action(DeclareLaunchArgument(name='confidence_threshold', default_value='0.7',
                                        description='The minimum confidence threshold'))
    
    # The model to be used
    ld.add_action(DeclareLaunchArgument(name='model', default_value='SmallModel.pt',
                                        description='The model to be used [SmallModel.pt]'))

    # Should UI Feedback be given
    ld.add_action(DeclareLaunchArgument(name='ui_feedback', default_value='True',
                                        description='Boolean value'))

    ld.add_action(yolo_node)
    return ld

