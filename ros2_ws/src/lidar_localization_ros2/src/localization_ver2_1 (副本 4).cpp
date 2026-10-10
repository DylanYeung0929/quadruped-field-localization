#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <cmath>
#include <functional>
#include <string>

// DYLANYEUNG: v4 - 2026/03/22
// 【新增】外接 HiPNUC 9轴 IMU（HI229 / CH100 等带磁力计型号）提供绝对 yaw
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │  yaw 来源选择                                                        │
// │                                                                     │
// │  模式 A（推荐）: HiPNUC 绝对 yaw                                      │
// │    - 订阅 /hipnuc/imu（或你配置的话题名）                               │
// │    - 直接读取 msg->orientation 四元数 → getRPY → yaw                   │
// │    - 零漂移，无需标定，立即可用                                         │
// │    - 注意坐标系：HiPNUC 默认 NED 输出，需转换到 ROS ENU                  │
// │      NED yaw（指北=0，顺时针正）→ ENU yaw（指东=0，逆时针正）            │
// │      转换公式：yaw_enu = π/2 - yaw_ned                                 │
// │                                                                     │
// │  模式 B（fallback）: Livox 内置 IMU 陀螺仪积分                          │
// │    - 订阅 /livox/imu                                                  │
// │    - 沿用 v3.1 的零偏标定+运行时校正机制                                 │
// │    - 当 HiPNUC 话题不可用时自动降级                                     │
// └─────────────────────────────────────────────────────────────────────┘
//
// ╔══════════════════════════════════════════════════════════════════════╗
// ║  重要：启动前确认你的 HiPNUC 型号和坐标系配置                           ║
// ║                                                                     ║
// ║  1. 型号确认：                                                        ║
// ║     HI226 = 6轴，无磁力计，orientation.z(yaw) 只是积分值，与 Livox 一样  ║
// ║     HI229 / CH100 / CH104 = 9轴，有磁力计，yaw 是真正的绝对磁北参考      ║
// ║     → 只有 HI229/CH100/CH104 等9轴型号才能给你真正的绝对 yaw            ║
// ║                                                                     ║
// ║  2. 坐标系配置（在 HiPNUC CHCenter 或 CONFIG 指令中设置）：              ║
// ║     默认：NED（北东下），yaw=0 指向磁北，顺时针为正                       ║
// ║     可选：ENU（东北上），yaw=0 指向东，逆时针为正（与 ROS 一致）           ║
// ║     → 推荐在 CHCenter 里把输出改为 ENU，省去代码里的转换                  ║
// ║     → 如果用 NED，把 USE_NED_FRAME 设为 true                            ║
// ║                                                                     ║
// ║  3. 话题名：HiPNUC 官方 ROS2 驱动默认发布 /imu/data                     ║
// ║     可在下方 HIPNUC_IMU_TOPIC 修改                                     ║
// ╚══════════════════════════════════════════════════════════════════════╝
//
// 其余定位逻辑（v3 机体系轴分配修复）保持不变。

// ============================================================
// ★ 用户配置区 — 根据你的实际情况修改这里 ★
// ============================================================

// 轮趣 N300 IMU 话题名（wheeltec_n100_imu 驱动发布到 "/imu"）
static constexpr const char* HIPNUC_IMU_TOPIC = "/imu";

// N300 四元数直接使用，不做 NED→ENU 转换（转换反而引入误差）
static constexpr bool USE_NED_FRAME = false;

// N300 启动后磁航向稳定所需时间（秒）
// SPKF 融合滤波器收敛较快，2s 足够
static constexpr double HIPNUC_SETTLE_TIME = 2.0;

// ============================================================

struct GridCell { float min_z, max_z; };

// ============================================================
// 将 NED 四元数转换为 ROS ENU 四元数
//
// NED 坐标系：X=北 Y=东 Z=下，yaw=0 指北，顺时针为正（航空惯例）
// ENU 坐标系：X=东 Y=北 Z=上，yaw=0 指东，逆时针为正（ROS 惯例）
//
// 转换关系（固定旋转）：
//   R_ENU = R_NED2ENU * R_NED_body
//   其中 R_NED2ENU = Rz(+90°) * Rx(180°)
//   对四元数等价为：q_enu = q_ned2enu * q_ned
//
// 等价于欧拉角：
//   roll_enu  =  pitch_ned
//   pitch_enu =  roll_ned
//   yaw_enu   =  π/2 - yaw_ned   （磁北→指东，方向反转）
// ============================================================
static tf2::Quaternion ned_to_enu(const tf2::Quaternion& q_ned)
{
    // NED → ENU 的固定坐标系旋转四元数
    // 等价于先绕 X 转 180°，再绕 Z 转 +90°
    // q = [w, x, y, z] = [cos(π/4)*cos(π/2), sin(π/4)*cos(π/2),
    //                       cos(π/4)*sin(π/2), -sin(π/4)*sin(π/2)]
    // 化简：q_ned2enu = (√2/2, √2/2, 0, 0) * Rz(90°) 分量
    // 直接用轴角：先 Rz(90°) 再 Rx(180°)
    tf2::Quaternion q_rz90, q_rx180;
    q_rz90.setRPY(0, 0,  M_PI / 2.0);
    q_rx180.setRPY(M_PI, 0, 0);
    tf2::Quaternion q_ned2enu = q_rz90 * q_rx180;
    return (q_ned2enu * q_ned).normalized();
}

// ============================================================
// 角度归一化辅助
// ============================================================
static double wrap_pi(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

// ============================================================
// Livox 内置 IMU 积分器（fallback，仅在无 HiPNUC 时使用）
// 沿用 v3.1 的零偏标定+静止检测逻辑，简化版
// ============================================================
struct LivoxYawIntegrator {
    double yaw        = 0.0;
    double timestamp  = -1.0;
    bool   ready      = false;

    std::deque<double> calib_buf;
    double gyro_z_bias = 0.0;
    static constexpr int    CALIB_SAMPLES  = 300;
    static constexpr double STATIC_VAR_MAX = 1e-5;
    static constexpr double TURN_THRESH    = 0.12;

    double yaw_rate   = 0.0;
    bool   is_turning = false;

    bool is_calib_static() const {
        if ((int)calib_buf.size() < CALIB_SAMPLES) return false;
        double s = 0; for (double v : calib_buf) s += v;
        double m = s / calib_buf.size(), var = 0;
        for (double v : calib_buf) var += (v-m)*(v-m);
        return (var / calib_buf.size()) < STATIC_VAR_MAX;
    }

    void feed(double gz, double ts) {
        if (!ready) {
            calib_buf.push_back(gz);
            if ((int)calib_buf.size() > CALIB_SAMPLES * 2) calib_buf.pop_front();
            if ((int)calib_buf.size() < CALIB_SAMPLES) return;
            if (!is_calib_static()) { calib_buf.clear(); return; }
            double s = 0; for (double v : calib_buf) s += v;
            gyro_z_bias = s / calib_buf.size();
            timestamp = ts; ready = true; return;
        }
        double dt = ts - timestamp; timestamp = ts;
        if (dt <= 0.0 || dt > 0.5) return;
        yaw_rate  = gz - gyro_z_bias;
        is_turning = std::abs(yaw_rate) > TURN_THRESH;
        yaw = wrap_pi(yaw + yaw_rate * dt);
    }
};

// ============================================================
// 节点
// ============================================================
class Mid360LocalizationNode : public rclcpp::Node
{
public:
    Mid360LocalizationNode() : Node("mid360_localization_node")
    {
        auto qos = rclcpp::SensorDataQoS();

        // Livox 内置 IMU（roll/pitch 来源，以及 fallback yaw）
        livox_imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/livox/imu", qos,
            std::bind(&Mid360LocalizationNode::livox_imu_cb, this, std::placeholders::_1));

        // HiPNUC 外接 IMU（绝对 yaw 来源）
        hipnuc_imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            HIPNUC_IMU_TOPIC, qos,
            std::bind(&Mid360LocalizationNode::hipnuc_imu_cb, this, std::placeholders::_1));

        // LiDAR
        cloud_sub_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/livox/lidar", qos,
            std::bind(&Mid360LocalizationNode::cloud_cb, this, std::placeholders::_1));

        dbg_pub_    = create_publisher<sensor_msgs::msg::PointCloud2>("/localization/debug_cloud", 10);
        marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/localization/markers", 10);
        tf_br_      = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        grid_.resize(200 * 200);

        RCLCPP_INFO(get_logger(),
            "[v4] 启动 — HiPNUC 绝对 yaw 模式");
        RCLCPP_INFO(get_logger(),
            "[v4] 订阅 HiPNUC: %s  坐标系: %s  收敛等待: %.1fs",
            HIPNUC_IMU_TOPIC,
            USE_NED_FRAME ? "NED（将自动转 ENU）" : "ENU",
            HIPNUC_SETTLE_TIME);
        RCLCPP_INFO(get_logger(),
            "[v4] 若 HiPNUC 话题无数据，将自动降级到 Livox 内置 IMU 积分（需静止标定）");
    }

private:
    // ---- 姿态 ----
    double cur_roll_  = 0.0;
    double cur_pitch_ = 0.0;
    double cur_yaw_   = 0.0;
    bool   imu_ready_ = false;
    bool   is_turning_ = false;

    // ---- HiPNUC 状态 ----
    bool   hipnuc_active_   = false;   // 是否已收到 HiPNUC 数据
    double hipnuc_first_ts_ = -1.0;    // 第一帧时间戳，用于等待收敛
    bool   hipnuc_settled_  = false;   // 滤波器已收敛

    // ---- Livox fallback ----
    LivoxYawIntegrator livox_int_;
    bool livox_warned_ = false;  // 只打一次降级警告

    // ---- 定位结果 ----
    double robot_x_ = 0.0;
    double robot_y_ = 0.0;
    double smooth_x_ = 0.0, smooth_y_ = 0.0;
    bool   first_localize_ = true;

    std::vector<GridCell> grid_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr livox_imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr hipnuc_imu_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dbg_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_br_;

    // ============================================================
    // HiPNUC IMU 回调（绝对 yaw 主来源）
    // ============================================================
    void hipnuc_imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        double ts = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        // 记录首帧时间，等待融合滤波器收敛
        if (!hipnuc_active_) {
            hipnuc_active_   = true;
            hipnuc_first_ts_ = ts;
            RCLCPP_INFO(get_logger(),
                "[v4] ✓ 检测到 HiPNUC 数据，等待 %.1fs 磁航向融合收敛...",
                HIPNUC_SETTLE_TIME);
        }

        // 收敛等待
        if (!hipnuc_settled_) {
            if (ts - hipnuc_first_ts_ < HIPNUC_SETTLE_TIME) {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                    "[v4] HiPNUC 滤波器收敛中... %.1f/%.1fs",
                    ts - hipnuc_first_ts_, HIPNUC_SETTLE_TIME);
                return;
            }
            hipnuc_settled_ = true;
            RCLCPP_INFO(get_logger(), "[v4] ✓ HiPNUC 滤波器已收敛，开始使用绝对 yaw");
        }

        // 读取四元数（HiPNUC 驱动填充 orientation 字段）
        tf2::Quaternion q_raw(
            msg->orientation.x, msg->orientation.y,
            msg->orientation.z, msg->orientation.w);

        // 验证四元数有效（驱动异常时可能全0）
        double norm = q_raw.length();
        if (norm < 0.9 || norm > 1.1) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "[v4] HiPNUC 四元数无效 (norm=%.3f)，跳过本帧", norm);
            return;
        }

        // N300 四元数直接 getRPY 提取 yaw 是准确的（实测 180° 误差仅 3.4°）
        // ned_to_enu() 反而会引入额外误差，直接用原始四元数
        double roll, pitch, yaw;
        tf2::Matrix3x3(q_raw).getRPY(roll, pitch, yaw);

        // roll 在水平放置时约为 ±180°（N300 坐标系特性），
        // 对 roll/pitch 补偿点云时需要修正：实际补偿角 = roll - sign(roll)*180°
        // 但 yaw 是正确的，直接使用
        cur_roll_  = (roll > 0) ? roll - M_PI : roll + M_PI;
        cur_pitch_ = -pitch;
        cur_yaw_   = yaw;

        // 转弯检测：N300 的 HeadingSpeed 映射到 angular_velocity.z
        // 实测逆时针转动时 gz > 0，符合 ROS 约定，直接使用
        is_turning_ = std::abs(msg->angular_velocity.z) > LivoxYawIntegrator::TURN_THRESH;

        imu_ready_ = true;
    }

    // ============================================================
    // Livox 内置 IMU 回调（提供 roll/pitch，以及 fallback yaw）
    // ============================================================
    void livox_imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        double ts = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        // 从 Livox orientation 提取 roll/pitch（短期稳定，无漂移）
        tf2::Quaternion q(msg->orientation.x, msg->orientation.y,
                          msg->orientation.z, msg->orientation.w);
        double r, p, y_dummy;
        tf2::Matrix3x3(q).getRPY(r, p, y_dummy);

        // 如果 HiPNUC 已激活并收敛：roll/pitch/yaw 全部由 hipnuc_imu_cb 维护，不覆盖
        if (hipnuc_settled_) return;

        // ---- Fallback：HiPNUC 不可用，降级到 Livox 积分 yaw ----
        if (!livox_warned_ && !hipnuc_active_) {
            // 启动后 5s 仍无 HiPNUC 数据，打降级警告
            // 用 Livox 时间戳估算（简化：节点启动后第一帧即判断）
            // 实际会在 cloud_cb 里通过 imu_ready_ 的延迟发现
        }

        livox_int_.feed(msg->angular_velocity.z, ts);

        if (livox_int_.ready) {
            cur_roll_   = r;
            cur_pitch_  = p;
            cur_yaw_    = livox_int_.yaw;
            is_turning_ = livox_int_.is_turning;
            imu_ready_  = true;
        } else {
            int n = (int)livox_int_.calib_buf.size();
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "[v4-fallback] HiPNUC 未连接，Livox 零偏标定中 %d/%d (保持静止)...",
                n, LivoxYawIntegrator::CALIB_SAMPLES);
        }
    }

    // ============================================================
    // LiDAR 回调
    // ============================================================
    void cloud_cb(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        if (!imu_ready_) return;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& p : msg->points)
            cloud->push_back({p.x, p.y, p.z});
        process(cloud, msg->header);
    }

    bool is_self(float x, float y) {
        return (x > -0.7f && x < 0.2f && y > -0.3f && y < 0.3f);
    }

    // ============================================================
    // 点云预处理
    // ============================================================
    void process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std_msgs::msg::Header hdr)
    {
        // 姿态对齐（只补偿 roll/pitch）
        Eigen::Affine3f T = Eigen::Affine3f::Identity();
        T.rotate(Eigen::AngleAxisf(-cur_pitch_, Eigen::Vector3f::UnitY()));
        T.rotate(Eigen::AngleAxisf(-cur_roll_,  Eigen::Vector3f::UnitX()));
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud, *aligned, T);

        for (auto& c : grid_) { c.min_z = 999.f; c.max_z = -999.f; }
        for (const auto& p : aligned->points) {
            if (p.z < -0.4f || p.z > 0.4f || is_self(p.x, p.y)) continue;
            int gx = (int)((p.x + 10.f) / 0.1f);
            int gy = (int)((p.y + 10.f) / 0.1f);
            if (gx < 0 || gx >= 200 || gy < 0 || gy >= 200) continue;
            int i = gx * 200 + gy;
            if (p.z < grid_[i].min_z) grid_[i].min_z = p.z;
            if (p.z > grid_[i].max_z) grid_[i].max_z = p.z;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr walls2d(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& p : aligned->points) {
            if (p.z < -0.4f || p.z > 0.4f || is_self(p.x, p.y)) continue;
            int gx = (int)((p.x + 10.f) / 0.1f);
            int gy = (int)((p.y + 10.f) / 0.1f);
            if (gx < 0 || gx >= 200 || gy < 0 || gy >= 200) continue;
            int i = gx * 200 + gy;
            if (grid_[i].max_z - grid_[i].min_z >= 0.15f)
                walls2d->push_back({p.x, p.y, 0.f});
        }
        if (walls2d->size() < 40) return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr clean(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(walls2d);
        vg.setLeafSize(0.05f, 0.05f, 0.05f);
        vg.filter(*clean);
        if (clean->size() < 20) return;

        sensor_msgs::msg::PointCloud2 dbg;
        pcl::toROSMsg(*clean, dbg);
        dbg.header = hdr;
        dbg_pub_->publish(dbg);

        solve(clean, hdr);
    }

    // ============================================================
    // 几何解算（机体系轴分配，与 v3 完全一致）
    // ============================================================
    void solve(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std_msgs::msg::Header hdr)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.05);
        seg.setMaxIterations(200);

        std::vector<pcl::ModelCoefficients> lines;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        int orig_sz = cloud->size();

        while (cloud->size() > 0.05 * orig_sz && lines.size() < 3) {
            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coeff);
            if (inliers->indices.size() < 15) break;

            float x0 = coeff->values[0], y0 = coeff->values[1];
            float dx = coeff->values[3], dy = coeff->values[4];
            float n = std::sqrt(dx*dx + dy*dy);
            dx /= n; dy /= n;
            coeff->values[3] = dx; coeff->values[4] = dy;

            float min_t = 1e9, max_t = -1e9;
            for (int idx : inliers->indices) {
                float px = cloud->points[idx].x - x0;
                float py = cloud->points[idx].y - y0;
                float t = px*dx + py*dy;
                if (t < min_t) min_t = t;
                if (t > max_t) max_t = t;
            }
            if (max_t - min_t < 1.0f) {
                extract.setInputCloud(cloud); extract.setIndices(inliers);
                extract.setNegative(true); extract.filter(*cloud);
                continue;
            }

            bool valid = true;
            if (!lines.empty()) {
                Eigen::Vector2f dc(dx, dy), pc(x0, y0);
                for (const auto& prev : lines) {
                    Eigen::Vector2f dp(prev.values[3], prev.values[4]);
                    Eigen::Vector2f pp(prev.values[0], prev.values[1]);
                    float dot = std::abs(dc.dot(dp));
                    if (dot >= 0.122f && dot <= 0.992f) { valid = false; break; }
                    if (dot > 0.992f) {
                        Eigen::Vector2f np(-dp.y(), dp.x());
                        float dist = std::abs((pc - pp).dot(np));
                        if (!((dist > 3.4f && dist < 4.6f) || (dist > 5.4f && dist < 6.6f))) {
                            valid = false; break;
                        }
                    }
                }
            }

            if (valid) {
                coeff->values.push_back(min_t);
                coeff->values.push_back(max_t);
                lines.push_back(*coeff);
            }

            extract.setInputCloud(cloud); extract.setIndices(inliers);
            extract.setNegative(true); extract.filter(*cloud);
        }

        publish_markers(lines, hdr);
        if (lines.empty()) return;

        // ============================================================
        // 机体系轴分配（v3 核心修复，保持不变）
        // ============================================================
        double cos_y = std::cos(cur_yaw_);
        double sin_y = std::sin(cur_yaw_);

        double sum_x = 0, sum_y = 0;
        int count_x = 0, count_y = 0;

        for (const auto& line : lines) {
            float x0 = line.values[0], y0 = line.values[1];
            float dx = line.values[3], dy = line.values[4];

            float proj = x0*dx + y0*dy;
            float cx = x0 - proj*dx;
            float cy = y0 - proj*dy;
            float dist = std::sqrt(cx*cx + cy*cy);
            if (dist < 0.1f) continue;

            float nx = cx / dist;
            float ny = cy / dist;

            if (std::abs(nx) > 0.8f) {
                double wx = (nx > 0) ? cos_y : -cos_y;
                double wy = (nx > 0) ? sin_y : -sin_y;
                if (std::abs(wx) > std::abs(wy)) {
                    double wall_x = (wx > 0) ? 6.0 : 0.0;
                    sum_x += wall_x - wx * dist; count_x++;
                } else {
                    double wall_y = (wy > 0) ? 2.0 : -2.0;
                    sum_y += wall_y - wy * dist; count_y++;
                }
            } else if (std::abs(ny) > 0.8f) {
                double wx = (ny > 0) ? -sin_y : sin_y;
                double wy = (ny > 0) ?  cos_y : -cos_y;
                if (std::abs(wx) > std::abs(wy)) {
                    double wall_x = (wx > 0) ? 6.0 : 0.0;
                    sum_x += wall_x - wx * dist; count_x++;
                } else {
                    double wall_y = (wy > 0) ? 2.0 : -2.0;
                    sum_y += wall_y - wy * dist; count_y++;
                }
            }
        }

        // 转弯时降低激光权重
        double alpha = is_turning_ ? 0.15 : 0.4;

        if (count_x > 0) {
            double meas_x = sum_x / count_x;
            smooth_x_ = first_localize_ ? meas_x : smooth_x_ * (1.0 - alpha) + meas_x * alpha;
            robot_x_ = smooth_x_;
        }
        if (count_y > 0) {
            double meas_y = sum_y / count_y;
            smooth_y_ = first_localize_ ? meas_y : smooth_y_ * (1.0 - alpha) + meas_y * alpha;
            robot_y_ = smooth_y_;
        }
        if (count_x > 0 || count_y > 0) first_localize_ = false;

        // 发布 TF
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp    = hdr.stamp;
        t.header.frame_id = "map";
        t.child_frame_id  = hdr.frame_id;
        t.transform.translation.x = robot_x_;
        t.transform.translation.y = robot_y_;
        t.transform.translation.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(cur_roll_, cur_pitch_, cur_yaw_);
        t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
        tf_br_->sendTransform(t);

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 100,
            "[v4] X:%5.2f  Y:%5.2f  yaw:%6.1f°  yaw_src:%s  转弯:%s  walls:%zu  cx:%d  cy:%d",
            robot_x_, robot_y_, cur_yaw_ * 180.0 / M_PI,
            hipnuc_settled_ ? "HiPNUC" : "Livox",
            is_turning_ ? "YES" : "no",
            lines.size(), count_x, count_y);
    }

    // ============================================================
    // 可视化（与 v3 相同）
    // ============================================================
    void publish_markers(const std::vector<pcl::ModelCoefficients>& lines,
                         std_msgs::msg::Header hdr)
    {
        visualization_msgs::msg::MarkerArray arr;

        visualization_msgs::msg::Marker mk;
        mk.header = hdr; mk.ns = "lidar_pos"; mk.id = 0;
        mk.type = visualization_msgs::msg::Marker::CYLINDER;
        mk.action = visualization_msgs::msg::Marker::ADD;
        mk.scale.x = mk.scale.y = 0.2; mk.scale.z = 0.1;
        mk.color.r = 1.0; mk.color.a = 1.0;
        mk.pose.position.x = mk.pose.position.y = mk.pose.position.z = 0;
        arr.markers.push_back(mk);

        visualization_msgs::msg::Marker hd;
        hd.header = hdr; hd.ns = "heading"; hd.id = 1;
        hd.type = visualization_msgs::msg::Marker::ARROW;
        hd.action = visualization_msgs::msg::Marker::ADD;
        hd.scale.x = 0.5; hd.scale.y = 0.05; hd.scale.z = 0.05;
        hd.color.r = 1.0; hd.color.g = 0.5; hd.color.a = 1.0;
        hd.pose.position.x = hd.pose.position.y = hd.pose.position.z = 0;
        arr.markers.push_back(hd);

        for (size_t i = 0; i < lines.size(); ++i) {
            visualization_msgs::msg::Marker lm;
            lm.header = hdr; lm.ns = "walls"; lm.id = i + 2;
            lm.type = visualization_msgs::msg::Marker::LINE_LIST;
            lm.action = visualization_msgs::msg::Marker::ADD;
            lm.scale.x = 0.08;
            lm.color.g = 1.0; lm.color.a = 0.9;
            float x0 = lines[i].values[0], y0 = lines[i].values[1];
            float dx = lines[i].values[3], dy = lines[i].values[4];
            float mt = lines[i].values.size() > 6 ? lines[i].values[6] : -3.f;
            float xt = lines[i].values.size() > 7 ? lines[i].values[7] :  3.f;
            geometry_msgs::msg::Point p1, p2;
            p1.x = x0 + dx*mt; p1.y = y0 + dy*mt; p1.z = 0;
            p2.x = x0 + dx*xt; p2.y = y0 + dy*xt; p2.z = 0;
            lm.points.push_back(p1); lm.points.push_back(p2);
            arr.markers.push_back(lm);
        }
        marker_pub_->publish(arr);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Mid360LocalizationNode>());
    rclcpp::shutdown();
    return 0;
}