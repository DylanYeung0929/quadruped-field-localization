import os

import launch
import launch.actions
import launch.events

import launch_ros
import launch_ros.actions
import launch_ros.events

from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node

import lifecycle_msgs.msg

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    ld = launch.LaunchDescription()

    # ---------------------------------------------------------
    # 修改 1: 删除了原本在这里的 lidar_tf 和 imu_tf 静态变换节点
    # 防止与 FAST-LIO 的 TF 树冲突
    # ---------------------------------------------------------

    localization_param_dir = launch.substitutions.LaunchConfiguration(
        'localization_param_dir',
        default=os.path.join(
            get_package_share_directory('lidar_localization_ros2'),
            'param',
            'localization.yaml'))

    lidar_localization = launch_ros.actions.LifecycleNode(
        name='lidar_localization',
        namespace='',
        package='lidar_localization_ros2',
        executable='lidar_localization_node',
        parameters=[localization_param_dir],
        # ---------------------------------------------------------
        # 修改重点在这里！
        # ---------------------------------------------------------
        remappings=[
            # 1. 雷达数据匹配
            # 左边 '/cloud' 是节点内部的名字（不要改）
            # 右边 '/livox/lidar' 是你的真实雷达话题
            # (如果你想用 FAST_LIO 处理好的点云，也可以保留 '/cloud_registered')
            ('/cloud', '/cloud_registered'),

            # 2. 里程计辅助 (强烈建议加上)
            # 左边 '/odom' 是节点内部想要听的里程计话题
            # 右边 '/Odometry' 是 FAST_LIO 发出来的里程计话题
            ('/odom', '/Odometry')
        ],
        output='screen')

    to_inactive = launch.actions.EmitEvent(
        event=launch_ros.events.lifecycle.ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(lidar_localization),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    from_unconfigured_to_inactive = launch.actions.RegisterEventHandler(
        launch_ros.event_handlers.OnStateTransition(
            target_lifecycle_node=lidar_localization,
            goal_state='unconfigured',
            entities=[
                launch.actions.LogInfo(msg="-- Unconfigured --"),
                launch.actions.EmitEvent(event=launch_ros.events.lifecycle.ChangeState(
                    lifecycle_node_matcher=launch.events.matches_action(lidar_localization),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                )),
            ],
        )
    )

    from_inactive_to_active = launch.actions.RegisterEventHandler(
        launch_ros.event_handlers.OnStateTransition(
            target_lifecycle_node=lidar_localization,
            start_state = 'configuring',
            goal_state='inactive',
            entities=[
                launch.actions.LogInfo(msg="-- Inactive --"),
                launch.actions.EmitEvent(event=launch_ros.events.lifecycle.ChangeState(
                    lifecycle_node_matcher=launch.events.matches_action(lidar_localization),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )

    ld.add_action(from_unconfigured_to_inactive)
    ld.add_action(from_inactive_to_active)

    ld.add_action(lidar_localization)
    # ld.add_action(lidar_tf) # 已移除
    ld.add_action(to_inactive)

    return ld
