#!/usr/bin/env python3
# config:utf-8
from os.path import join
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_prefix = get_package_share_directory('gng_livox')
    gng_node119 = Node(
        package='gng_livox',
        namespace='livox119',
        executable='gng_livox',
        output='screen',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter119.yaml')
        ]
    )

    gng_node128 = Node(
        package='gng_livox',
        namespace='livox128',
        executable='gng_livox',
        output='screen',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter128.yaml')
        ]
    )

    gng_node130 = Node(
        package='gng_livox',
        namespace='livox130',
        executable='gng_livox',
        output='screen',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter130.yaml')
        ]
    )

    gng_node168 = Node(
        package='gng_livox',
        namespace='livox168',
        executable='gng_livox',
        output='screen',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter168.yaml')
        ]
    )

    ld = LaunchDescription()
    ld.add_action(gng_node119)
    ld.add_action(gng_node128)
    ld.add_action(gng_node130)
    ld.add_action(gng_node168)
    return ld