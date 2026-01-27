#!/usr/bin/env python3
# config:utf-8
"""Launch the trobot node with default configuration."""
# パスの検索時に使用するクラス
import os
# パッケージのinstall/shareディレクトリのパス検索クラス
from ament_index_python.packages import get_package_share_directory
# ノードの実行情報を記述するクラス($ ros2 runコマンドに相当)
from launch_ros.actions import Node
# 引数に渡したノード実行情報を実行するクラス
from launch import LaunchDescription
# ライフサイクルノードの実行情報を記述するクラス
from launch_ros.actions import LifecycleNode
# イベントハンドラーの宣言を記述するクラス
from launch.actions import RegisterEventHandler
# イベントハンドラーの発動条件を記述するクラス
from launch.event_handlers import OnProcessStart
from launch_ros.event_handlers import OnStateTransition
# イベントを記述するクラス
from launch.actions import EmitEvent
from launch.events import matches_action
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
# ログを出力をするクラス
from launch.actions import LogInfo
# 他のlaunchファイルを再利用する際に使用するクラス
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource,PythonLaunchDescriptionSource

def generate_launch_description():
    livox_bringup_dir = os.path.join(get_package_share_directory('livox_ros_driver2'))
    livox_launch_file = os.path.join(livox_bringup_dir,'launch', 'rviz_MID360_launch.py')
    gng_bringup_dir = os.path.join(get_package_share_directory('gng_livox'))
    gng_launch_file = os.path.join(gng_bringup_dir,'launch', 'gng_launch.py')

     # map -> odom tf2 staticノード設定
    map_odom_tf2 = Node(
        package = 'tf2_ros',
        executable = 'static_transform_publisher',
        name = 'map_odom_tf2',
        # arguments=[ 
        #     '--x', '0.0', '--y', '0', '--z', '0.0', 
        #     '--yaw', '0', '--pitch', '0', '--roll', '0', 
        #     '--frame-id', 'map', '--child-frame-id', 'odom', '--period_in_ms'
        # ],
        arguments=["0", "0", "0", "0", "0", "0", "odom", "base_footprint"],
    )

    # base_footprint -> base_link tf2 staticノード設定
    base_footprint_base_link = Node(
        package = 'tf2_ros',
        executable = 'static_transform_publisher',
        name = 'base_footprint_base_link',
        # arguments=[  
        #     '--x', '0.0', '--y', '0', '--z', '0.0', 
        #     '--yaw', '0', '--pitch', '0', '--roll', '0', 
        #     '--frame-id', 'base_footprint', '--child-frame-id', 'base_link'
        # ],
        arguments=["0", "0", "0.19", "0", "0", "0", "base_footprint", "base_link"],
    )

    # base_link -> laser tf2 staticノード設定
    # base_link_laser = Node(
    #     package = 'tf2_ros',
    #     executable = 'static_transform_publisher',
    #     name = 'base_link_laser',
    #     # arguments=[
    #     #     '--x', '0.10', '--y', '0', '--z', '0.18', 
    #     #     '--yaw', '0', '--pitch', '0', '--roll', '0', 
    #     #     '--frame-id', 'base_link', '--child-frame-id', 'laser'
    #     # ],
    #     arguments=["0.10", "0", "0.18", "0", "0", "0", "base_link", "laser"],
    # )

    # base_link -> livox tf2 staticノード設定
    base_link_livox = Node(
        package = 'tf2_ros',
        executable = 'static_transform_publisher',
        name = 'base_link_livox',
        # arguments=[
        #     '--x', '0.10', '--y', '0', '--z', '0.18', 
        #     '--yaw', '0', '--pitch', '0', '--roll', '0', 
        #     '--frame-id', 'base_link', '--child-frame-id', 'velodyne'
        # ],
        #arguments=["-0.00", "0", "0.2", "0", "1.570796", "0", "base_link", "livox_frame"],
        arguments=["-0.00", "0", "0.2", "0", "0.950796", "0", "base_link", "livox_frame"],
    )

    livox_launch_include = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(livox_launch_file),
    )
    gng_launch_include = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(gng_launch_file),
    )

    # 起動設定リスト
    # launch起動時に使用される関数の戻り値(起動するものをここに追加する)
    ld = LaunchDescription()

    ld.add_action(map_odom_tf2)
    # base_footprint -> base_link tf2起動設定
    ld.add_action(base_footprint_base_link)
    # base_link -> laser tf2 staticノード設定
    # ld.add_action(base_link_laser)
    # base_link -> oakd tf2 staticノード設定
    ld.add_action(base_link_livox)

    # livox起動設定
    ld.add_action(livox_launch_include)
    # gng起動設定
    ld.add_action(gng_launch_include)

    return ld