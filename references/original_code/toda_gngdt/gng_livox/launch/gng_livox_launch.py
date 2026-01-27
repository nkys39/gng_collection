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
    livox_bringup_dir = os.path.join(get_package_share_directory('livox_ros2_driver'))
    livox_launch_file = os.path.join(livox_bringup_dir,'launch', 'livox_lidar_rviz_launch.py')
    gng_bringup_dir = os.path.join(get_package_share_directory('gng_livox'))
    gng_launch_file = os.path.join(gng_bringup_dir,'launch', 'gng_launch.py')

    livox_launch_include = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(livox_launch_file),
    )
    gng_launch_include = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(gng_launch_file),
    )

    # 起動設定リスト
    # launch起動時に使用される関数の戻り値(起動するものをここに追加する)
    ld = LaunchDescription()
    # livox起動設定
    ld.add_action(livox_launch_include)
    # gng起動設定
    ld.add_action(gng_launch_include)

    return ld