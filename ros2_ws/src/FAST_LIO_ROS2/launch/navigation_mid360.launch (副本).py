import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 0. 获取配置文件路径
    config_file = os.path.join(
        get_package_share_directory('fast_lio'),
        'config',
        'mid360.yaml'
    )

    # 1. 启动 FAST-LIO (使用我们要用的新节点 fastlio_mapping_nav)
    # 这个节点已经修改过 frame_id 和时间戳，完美适配 Nav2
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping_nav',
        name='fastlio_mapping_nav',
        output='screen',
        parameters=[config_file]
    )

    # 2. 静态坐标变换 (Map -> Odom)
    # 假设机器人从地图原点启动，不需要复杂的定位算法时，直接把 map 和 odom 锁死
    tf_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    # 3. 点云转 2D 激光 (用于 Nav2 避障)
    # 任务赛配置：min_height=0.1 过滤掉 50mm 高的减速带
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[
            ('cloud_in', '/cloud_registered'), # 订阅 FAST-LIO 的点云
            ('scan', '/scan')                  # 输出给 Nav2 的激光数据
        ],
        parameters=[{
            'target_frame': 'base_link',
            'transform_tolerance': 0.01,
            'min_height': 0.1,    # !!! 关键：忽略 10cm 以下物体 (减速带通过，箱子阻挡)
            'max_height': 1.0,    # 只看 1m 高度内的障碍
            'angle_min': -3.1415,
            'angle_max': 3.1415,
            'angle_increment': 0.0087,
            'scan_time': 0.1,
            'range_min': 0.2,
            'range_max': 10.0,
            'use_inf': True
        }]
    )

    # 4. 启动 Nav2 (导航全家桶)
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': 'false',   # 实机运行必须是 false
            'params_file': PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'params', 'nav2_params.yaml'
            ])
        }.items()
    )

    return LaunchDescription([
        fast_lio_node,
        tf_map_to_odom,
        pointcloud_to_laserscan,
        nav2_launch
    ])