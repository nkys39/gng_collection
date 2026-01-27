#!/usr/bin/env python3
# config:utf-8
from os.path import join
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_prefix = get_package_share_directory('gng_livox')
    
    gng_node_front_up = Node(
        package='gng_livox',    
        executable='gng_livox',
        name='gng_livox_front_up',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter_multi.yaml')
        ],
        # output='screen',
        # use_tf
        remappings=[
            # ('/livox/lidar_192_168_1_146', '/livox/lidar_192_168_1_146'),
            ('/gng_node', '/gng_node_front_up'),
            ('/gng_edge', '/gng_edge_front_up'),
        ],
    )
    gng_node_front_down = Node(
        package='gng_livox',    
        executable='gng_livox',
        name='gng_livox_front_down',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter_multi.yaml')
        ],
        # output='screen',
        # use_tf
        remappings=[
            # ('/livox/lidar_192_168_1_146', '/livox/lidar_192_168_1_117'),
            ('/gng_node', '/gng_node_front_down'),
            ('/gng_edge', '/gng_edge_front_down'),
        ],
    )

    gng_node_right = Node(
        package='gng_livox',    
        executable='gng_livox',
        name='gng_livox_right',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter_multi.yaml')
        ],
        # output='screen',
        # use_tf
        remappings=[
            # ('/livox/lidar_192_168_1_146', '/livox/lidar_192_168_1_144'),
            ('/gng_node', '/gng_node_right'),
            ('/gng_edge', '/gng_edge_right'),
        ],
    )

    gng_node_left = Node(
        package='gng_livox',    
        executable='gng_livox',
        name='gng_livox_left',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter_multi.yaml')
        ],
        # output='screen',
        # use_tf
        remappings=[
            # ('/livox/lidar_192_168_1_146', '/livox/lidar_192_168_1_181'),
            ('/gng_node', '/gng_node_left'),
            ('/gng_edge', '/gng_edge_left'),
        ],
    )

    gng_node_back = Node(
        package='gng_livox',    
        executable='gng_livox',
        name='gng_livox_back',
        parameters=[
            join(pkg_prefix, 'config/oic_parameter_multi.yaml')
        ],
        # output='screen',
        # use_tf
        remappings=[
            # ('/livox/lidar_192_168_1_146', '/livox/lidar_192_168_1_103'),
            ('/gng_node', '/gng_node_back'),
            ('/gng_edge', '/gng_edge_back'),
        ],
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
    ld.add_action(gng_node_front_up)
    ld.add_action(gng_node_front_down)
    ld.add_action(gng_node_right)
    ld.add_action(gng_node_left)
    ld.add_action(gng_node_back)
    # ld.add_action(rviz2_node)# use_tf
    #ld.add_action(gng_node128)
    return ld