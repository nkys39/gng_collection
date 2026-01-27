#!/usr/bin/env python3
# config:utf-8
from os.path import join
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_prefix = get_package_share_directory('gng_livox')
    gng_node = Node(
        package='gng_livox',
        executable='gng_livox',
        name='gng_livox',
        output='screen',
        parameters=[
            join(pkg_prefix, 'config/parameter.yaml')
        ]
    )
    ld = LaunchDescription()
    ld.add_action(gng_node)
    return ld