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
        output='screen',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter.yaml'),
            # {'use_sim_time': True}
        ]
    )
    # use_tf
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', join(pkg_prefix, 'rviz/test.rviz')],
        parameters=[
            # {'use_sim_time': True}
        ]
    )

    # gng_node128 = Node(
    #     package='gng_livox',
    #     namespace='livox128',
    #     executable='gng_livox',
    #     output='screen',
    #     parameters=[
    #         join(pkg_prefix, 'config/oic_parameter128.yaml')
    #     ]

    #     remappings=[
    #     ('/gng_node', '/livox128/gng_node'),
    #     ('/gng_edge', '/livox128/gng_edge'),
    #     ]
    # )

    ld = LaunchDescription()
    ld.add_action(gng_node)
    # ld.add_action(rviz2_node)# use_tf
    #ld.add_action(gng_node128)
    return ld