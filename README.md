# 四足场地定位（ROS 2）

本仓库汇集四足机器人场地定位所使用的 ROS 2 源码、配置和驱动：Livox MID-360 点云驱动、Hipnuc IMU/GNSS 驱动、FAST-LIO2 里程计建图，以及基于 NDT 的已知地图定位。

> 本仓库只保存可构建的源码与配置版本，不上传场地地图、点云、录像、rosbag、`build/`、`install/` 和 `log/`。将现场地图放到本地后，再在定位参数中设置其绝对路径。

## 工作流

```text
MID-360 点云 ─┐
               ├─ FAST-LIO2 ──> 实时里程计 / 建图
Hipnuc IMU ────┘

MID-360 点云 + 已知场地 PCD 地图 ──> NDT 定位 ──> map → odom → base_link
```

## 目录

| 路径 | 作用 |
| --- | --- |
| `ros2_ws/src/livox_ros_driver2` | Livox ROS 2 驱动；保留 MID-360 网络与时间戳相关的本地修改。 |
| `ros2_ws/src/hipnuc_*` | Hipnuc IMU / GNSS ROS 2 驱动包。 |
| `ros2_ws/src/FAST_LIO_ROS2` | FAST-LIO2 建图与里程计包，默认活动配置为 `config/mid360.yaml`。 |
| `ros2_ws/src/lidar_localization_ros2` | 基于 NDT 的场地地图定位包。 |
| `ros2_ws/src/ndt_omp_ros2` | 定位包依赖的 OpenMP 加速 NDT 实现。 |
| `docs/` | 代码来源、版本说明和 Hipnuc 原始说明。 |

## 构建

在具备 ROS 2、PCL、Eigen、Livox SDK 与相应依赖的环境中执行：

```bash
cd ros2_ws
source /opt/ros/<你的发行版>/setup.bash
colcon build --symlink-install
source install/setup.bash
```

首次运行前必须检查：

1. `livox_ros_driver2/config/MID360_config.json` 中本机与雷达的 IP 地址、数据端口和 `pcl_data_type`；
2. `FAST_LIO_ROS2/config/mid360.yaml` 中点云、IMU 话题以及外参；
3. `lidar_localization_ros2/param/localization.yaml` 中 `map_path`，将其替换成当前设备上的场地 PCD 地图绝对路径；
4. Hipnuc 对应包的 `config/hipnuc_config.yaml` 中串口 / CAN / GNSS 配置。

## 运行入口

具体 launch 文件与参数参见 [代码组成与版本说明](docs/代码组成与版本说明.md)。建议按“驱动 → FAST-LIO2 或 NDT 定位 → RViz”的顺序启动，并在真实机器人部署前检查 TF 坐标系与时间戳。

## 许可证与来源

各上游包保留其原始 `LICENSE` 与版权声明。集成本身不改变其开源许可证；详情见 [上游代码来源](docs/上游代码来源.md)。
