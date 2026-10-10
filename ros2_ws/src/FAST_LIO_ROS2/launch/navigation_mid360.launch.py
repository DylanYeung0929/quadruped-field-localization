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

    # 1. 启动 FAST-LIO (使用原版节点名 fastlio_mapping)
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',  # <--- 改回了这里
        name='fastlio_mapping',        # <--- 改回了这里
        output='screen',
        parameters=[config_file]
    )

    # 2. 静态坐标变换 (Map -> Odom)
    # 假设机器人从原点启动，锁死 map 和 odom
    tf_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    # 3. 点云转 2D 激光 (用于 Nav2 避障)
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[
            ('cloud_in', '/cloud_registered'), 
            ('scan', '/scan')
        ],
        parameters=[{
            'target_frame': 'base_link',
            'transform_tolerance': 0.01,
            'min_height': 0.1,    # 过滤减速带 (10cm以下忽略)
            'max_height': 1.0,
            'angle_min': -3.1415,
            'angle_max': 3.1415,
            'angle_increment': 0.0087,
            'scan_time': 0.1,
            'range_min': 0.2,
            'range_max': 10.0,
            'use_inf': True
        }]
    )

    # 4. 启动 Nav2
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': 'false',
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