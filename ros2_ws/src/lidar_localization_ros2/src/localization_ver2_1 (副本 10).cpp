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
#include <tf2/LinearMath/Vector3.h>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>

#include <std_srvs/srv/trigger.hpp>

//问题：改善卡顿（未经验证） 20260517创建此副本

// ============================================================
// ★ 物理安装方向补偿配置 ★ 
// ============================================================
// Mid-360 倒置向前安装 (绕X轴转了180度)，开启此项将在内存中把点云掰正
static constexpr bool LIDAR_INVERTED = true; 

// Hipnuc HI14S2 正置向前安装 (Z朝上，X朝前)
// 因此直接信任它的原始坐标系，无需翻转符号
static constexpr bool   SWAP_ROLL_PITCH = false;
static constexpr double ROLL_SIGN  = 1.0;  
static constexpr double PITCH_SIGN = 1.0;  
static constexpr double YAW_SIGN   = 1.0;    

struct GridCell { float min_z, max_z; };

static double wrap_pi(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

class Mid360LocalizationNode : public rclcpp::Node
{
public:
    Mid360LocalizationNode() : Node("mid360_localization_node")
    {
        start_srv_ = create_service<std_srvs::srv::Trigger>(
            "/start_localization",
            std::bind(&Mid360LocalizationNode::start_srv_cb, this, std::placeholders::_1, std::placeholders::_2));
        // ★ 核心修复：使用 SensorDataQoS 解决与 Hipnuc 驱动的 QoS 冲突警告
        auto imu_qos = rclcpp::SensorDataQoS();
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/imu_state", imu_qos, std::bind(&Mid360LocalizationNode::imu_cb, this, std::placeholders::_1));

        auto sensor_qos = rclcpp::SensorDataQoS();
        cloud_sub_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/livox/lidar", sensor_qos, std::bind(&Mid360LocalizationNode::cloud_cb, this, std::placeholders::_1));

        aligned_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/localization/aligned_cloud", 10);
        dbg_pub_    = create_publisher<sensor_msgs::msg::PointCloud2>("/localization/debug_cloud", 10);
        marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/localization/markers", 10);
        odom_pub_   = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
        tf_br_      = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        grid_.resize(200 * 200);
        RCLCPP_INFO(get_logger(), "[v8.0] 启动 — 适配 Mid-360 倒置 + Hipnuc Z轴纯积分模式！");
    }

private:
    double cur_roll_  = 0.0;
    double cur_pitch_ = 0.0;
    double cur_yaw_   = 0.0;  
    bool   imu_ready_ = false;

    // --- 纯 IMU 速度积分变量 ---
    double imu_vx_ = 0.0; // 世界系 X 速度 (相对于开机航向)
    double imu_vy_ = 0.0; // 世界系 Y 速度
    double imu_vz_ = 0.0;

    // --- 纯积分核心变量 ---
    rclcpp::Time last_imu_time_;
    bool   first_imu_ = true;
    double pure_yaw_ = 0.0; 

    double robot_x_ = 0.0;
    double robot_y_ = 0.0;
    double smooth_x_ = 0.0, smooth_y_ = 0.0;
    bool   first_localize_ = true;
    bool   is_turning_ = false;

    std::vector<GridCell> grid_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dbg_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_; 
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_br_;

    // ★ 新增：休眠锁标志位
    bool is_localization_active_ = false;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_;

    // ★ 新增：服务回调函数（当队友发送请求时触发）
    void start_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        (void)req; // Trigger 请求中没有参数，忽略即可
        
        if (is_localization_active_) {
            res->success = true;
            res->message = "定位节点已经在运行中，无需重复唤醒";
            return;
        }

        is_localization_active_ = true; // 解开休眠锁
        res->success = true;            // 回复
        res->message = "定位引擎正式激活";
        RCLCPP_INFO(get_logger(), "收到电控唤醒指令，定位节点开始运算");
    }

    // ============================================================
    // IMU 回调：获取绝对重力水平 + 纯积分解算 Yaw
    // ============================================================
    void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        if (!is_localization_active_) return;
        rclcpp::Time current_time = msg->header.stamp;

        // 1. 提取 Hipnuc 解算好的完美 Roll 和 Pitch (仅用于重力对齐)
        tf2::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
        tf2::Matrix3x3 m(q);
        double raw_roll, raw_pitch, unused_yaw;
        m.getRPY(raw_roll, raw_pitch, unused_yaw); 

        cur_roll_  = raw_roll * ROLL_SIGN;
        cur_pitch_ = raw_pitch * PITCH_SIGN;

        // 2. 积分时间戳初始化
        if (first_imu_) {
            last_imu_time_ = current_time;
            pure_yaw_ = 0.0; 
            first_imu_ = false;
            imu_ready_ = true; // ★ 核心修复：拿到首帧数据后，必须立刻放行雷达！
            RCLCPP_INFO(get_logger(), "✅ [HI14S2] 成功接收首帧 IMU，Z 轴积分已启动！");
            return;
        }

        // 3. 计算 dt 并更新上一帧时间 (修复时间戳异常导致的死锁)
        double dt = (current_time - last_imu_time_).seconds();
        last_imu_time_ = current_time; 

        if (dt <= 0.0 || dt > 0.1) return;

        // 2. 航向角积分 (用于 Odom 发布)
        double omega_z = msg->angular_velocity.z * YAW_SIGN;
        pure_yaw_ += omega_z * dt;
        pure_yaw_ = wrap_pi(pure_yaw_);
        cur_yaw_ = pure_yaw_;

        // ============================================================
        // ★ 核心逻辑：加速度积分算速度 ★
        // ============================================================
        
        // A. 将本地加速度转换到世界系（即重力对齐后的坐标系）
        tf2::Vector3 acc_local(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        tf2::Vector3 acc_world = tf2::quatRotate(q, acc_local);

        // B. 减去重力加速度 (假设 Z 轴朝上)
        acc_world.setZ(acc_world.z() - 9.80665);

        // C. 积分得到世界系下的瞬时速度
        // 我们这里取水平面速度
        imu_vx_ += acc_world.x() * dt;
        imu_vy_ += acc_world.y() * dt;

        // D. 零速漂移抑制 (Leaky Integrator)
        // 这是一个工程小技巧：如果不给阻尼，IMU 静止时速度也会因为噪声缓慢增长
        if (std::abs(msg->angular_velocity.z) < 0.05 && 
            std::abs(acc_world.x()) < 0.1 && 
            std::abs(acc_world.y()) < 0.1) 
        {
            imu_vx_ *= 0.95; // 自动缓慢归零
            imu_vy_ *= 0.95;
        }

        // 3. 立即发布里程计数据，不受 Lidar 信号限制
        publish_odometry(msg->header.stamp, msg->angular_velocity);
        
        imu_ready_ = true;
    }

    void publish_odometry(const rclcpp::Time & stamp, const geometry_msgs::msg::Vector3 & ang_vel)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "map";
        odom.child_frame_id = "base_link";

        // 速度映射：对应位置映射 (out_x = -robot_y, out_y = robot_x)
        // 所以：out_vx = -vy_imu, out_vy = vx_imu
        double out_vx = -imu_vy_;
        double out_vy = imu_vx_;
        
        // 当前位置 (如果激活了定位就有值，没激活就是0)
        odom.pose.pose.position.x = -robot_y_;
        odom.pose.pose.position.y = robot_x_;
        
        // 姿态
        tf2::Quaternion q;
        q.setRPY(-cur_pitch_, cur_roll_, wrap_pi(cur_yaw_ + M_PI / 2.0));
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        // 写入计算出的速度
        odom.twist.twist.linear.x = out_vx;
        odom.twist.twist.linear.y = out_vy;
        odom.twist.twist.angular.z = ang_vel.z * YAW_SIGN;

        odom_pub_->publish(odom);
    }

    // ============================================================
    // 雷达回调：数字翻转点云 + 坐标系对齐
    // ============================================================
    void cloud_cb(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        if (!is_localization_active_) return;
        if (!imu_ready_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "⏳ 等待 IMU 数据流入...");
            return;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        
        for (const auto& p : msg->points) {
            if (LIDAR_INVERTED) {
                // 核心修复：将倒置的点云强行绕X轴旋转180度掰正
                // x 不变, y 取反, z 取反
                cloud->push_back({p.x, -p.y, -p.z});
            } else {
                cloud->push_back({p.x, p.y, p.z});
            }
        }
        process(cloud, msg->header);
    }

    bool is_self(float x, float y) { return (x > -0.7f && x < 0.2f && y > -0.3f && y < 0.3f); }

    void process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std_msgs::msg::Header hdr)
    {
        // 此时的 cloud 已经是正向的了，可以直接套用正置 IMU 的 Roll 和 Pitch
        Eigen::Affine3f T = Eigen::Affine3f::Identity();
        T.rotate(Eigen::AngleAxisf(-cur_pitch_, Eigen::Vector3f::UnitY()));
        T.rotate(Eigen::AngleAxisf(-cur_roll_,  Eigen::Vector3f::UnitX()));
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud, *aligned, T);

        sensor_msgs::msg::PointCloud2 aligned_msg;
        pcl::toROSMsg(*aligned, aligned_msg);
        aligned_msg.header = hdr;
        aligned_pub_->publish(aligned_msg);

        // Z轴切片滤地 (对齐后的点云，真实的 Z=0 就是物理地面)
        for (auto& c : grid_) { c.min_z = 999.f; c.max_z = -999.f; }
        for (const auto& p : aligned->points) {
            if (p.z < -0.5f || p.z > 0.5f || is_self(p.x, p.y)) continue;
            int gx = (int)((p.x + 10.f) / 0.1f);
            int gy = (int)((p.y + 10.f) / 0.1f);
            if (gx < 0 || gx >= 200 || gy < 0 || gy >= 200) continue;
            int i = gx * 200 + gy;
            if (p.z < grid_[i].min_z) grid_[i].min_z = p.z;
            if (p.z > grid_[i].max_z) grid_[i].max_z = p.z;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr walls2d(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& p : aligned->points) {
            if (p.z < -0.5f || p.z > 0.5f || is_self(p.x, p.y)) continue;
            int gx = (int)((p.x + 10.f) / 0.1f);
            int gy = (int)((p.y + 10.f) / 0.1f);
            if (gx < 0 || gx >= 200 || gy < 0 || gy >= 200) continue;
            int i = gx * 200 + gy;
            if (grid_[i].max_z - grid_[i].min_z >= 0.15f)
                walls2d->push_back({p.x, p.y, 0.f});
        }
        
        if (walls2d->size() < 40) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "⚠️ 墙壁点过少 (剩余: %zu < 40)。原因: 周围空间太大、无墙，或雷达放太低扫到了地板！", walls2d->size());
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr clean(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(walls2d);
        vg.setLeafSize(0.05f, 0.05f, 0.05f);
        vg.filter(*clean);
        
        if (clean->size() < 20) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "⚠️ 滤波后点云过少 (剩余: %zu < 20)，环境杂点太多被清除！", clean->size());
            return;
        }

        sensor_msgs::msg::PointCloud2 dbg;
        pcl::toROSMsg(*clean, dbg);
        dbg.header = hdr;
        dbg_pub_->publish(dbg);

        solve(clean, hdr);
    }

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

        while (cloud->size() > 0.05 * orig_sz && lines.size() < 5) {
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

            // min_t/max_t 的计算，用于确定线段长度和 Marker 显示
            float min_t = 1e9, max_t = -1e9;
            for (int idx : inliers->indices) {
                float px = cloud->points[idx].x - x0;
                float py = cloud->points[idx].y - y0;
                float t = px*dx + py*dy;
                if (t < min_t) min_t = t;
                if (t > max_t) max_t = t;
            }
            
            // 长度限制：如果拟合的直线长度小于 1 米，判定为非墙体障碍物
            if (max_t - min_t < 1.0f) {
                extract.setInputCloud(cloud); 
                extract.setIndices(inliers);
                extract.setNegative(true); // 剔除这些点
                extract.filter(*cloud);
                continue; // 继续寻找下一条足够长的线
            }

            float p_new = -x0 * dy + y0 * dx;

            bool valid = true;
            int replace_idx = -1;

            if (!lines.empty()) {
                Eigen::Vector2f dc(dx, dy);
                for (size_t i = 0; i < lines.size(); ++i) {
                    const auto& prev = lines[i];
                    Eigen::Vector2f dp(prev.values[3], prev.values[4]);
                    float dot = std::abs(dc.dot(dp));

                    if (dot >= 0.122f && dot <= 0.992f) { valid = false; break; }
                    if (dot > 0.992f) { // 平行线判定
                        // 计算已保存直线的带符号距离
                        float p_prev = -prev.values[0] * prev.values[4] + prev.values[1] * prev.values[3];
                        
                        // ★ 修改点3：同侧判定 (符号一致即为同侧)
                        bool same_side = ((p_new > 0 && p_prev > 0) || (p_new < 0 && p_prev < 0));

                        if (same_side) {
                            // 同侧取远者：如果新抽出的线比之前的更远，则标记替换；否则丢弃新线
                            if (std::abs(p_new) > std::abs(p_prev)) {
                                replace_idx = i;
                            } else {
                                valid = false; break;
                            }
                        } else {
                            // 异侧：校验间距是否符合场地尺寸 (4m 或 6m)
                            float dist_between = std::abs(p_new) + std::abs(p_prev);
                            if (!((dist_between > 3.7f && dist_between < 4.3f) || 
                                   (dist_between > 5.6f && dist_between < 6.4f))) {
                                valid = false; break;
                            }
                        }
                    }
                }
            }

            if (valid) {
                coeff->values.push_back(min_t);
                coeff->values.push_back(max_t);
                if (replace_idx != -1) lines[replace_idx] = *coeff;
                else lines.push_back(*coeff);
            }

            extract.setInputCloud(cloud); extract.setIndices(inliers);
            extract.setNegative(true); extract.filter(*cloud);
        }

        publish_markers(lines, hdr);
        
        if (lines.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "⚠️ 提取失败！未能从当前点云中拟合出符合正交规则的直线，场地可能不规整。");
            return;
        }

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


double alpha = is_turning_ ? 0.15 : 0.4;
        double base_jump_threshold = 0.5; // 基础门控阈值
        
        static int reject_count = 0;
        const int MAX_REJECT_FRAMES = 30;

        if (count_x > 0 || count_y > 0) {
            // 【关键修复 1：引入状态预测 (Predict)】
            // 利用 IMU 的测速，推算机器狗在这 0.1 秒内“理论上”走到了哪里。
            // 注意坐标系映射：Map Y (robot_x) 的速度是 imu_vx_；Map X (-robot_y) 的速度是 -imu_vy_
            double dt_lidar = 0.1; // 雷达 10Hz，帧间距 0.1s
            double pred_x = robot_x_ + imu_vx_ * dt_lidar;
            double pred_y = robot_y_ + imu_vy_ * dt_lidar;

            // 获取当前雷达观测值，若某方向无观测则沿用“预测值”而非“被冻结的老数据”
            double meas_x = count_x > 0 ? sum_x / count_x : pred_x;
            double meas_y = count_y > 0 ? sum_y / count_y : pred_y;

            // 计算实际观测与理论预测的偏差（学术上叫 Innovation 新息）
            double dx = count_x > 0 ? meas_x - pred_x : 0.0;
            double dy = count_y > 0 ? meas_y - pred_y : 0.0;
            double dist_jump = std::sqrt(dx * dx + dy * dy);

            // 【关键修复 2：动态膨胀门控 (Dynamic Gating)】
            // 盲走越久，IMU 累计误差越大，对跳变的容忍度应该适度放宽 (每帧放宽 2 厘米)
            double current_threshold = base_jump_threshold + reject_count * 0.02;

            // 欧氏距离门控判定
            if (!first_localize_ && dist_jump > current_threshold && reject_count < MAX_REJECT_FRAMES) {
                reject_count++;
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500, 
                    "跳变拦截,差值: %.3fm (容忍度: %.3fm)。启动 IMU 惯性滑行。拦截数: %d", 
                    dist_jump, current_threshold, reject_count);
                
                // 【关键修复 3：真正的惯性滑行 (Coasting)】
                // 拦截雷达时，坐标绝不能原地冻结，必须强制等于 IMU 推算的预测坐标！
                smooth_x_ = pred_x;
                smooth_y_ = pred_y;
                robot_x_ = smooth_x_;
                robot_y_ = smooth_y_;
            } else {
                if (reject_count >= MAX_REJECT_FRAMES) {
                    RCLCPP_WARN(get_logger(), "达到最大盲走帧数，强制重定位。");
                    first_localize_ = true;
                }

                // 正常观测，用预测值和雷达观测值进行加权平滑
                if (count_x > 0) {
                    smooth_x_ = first_localize_ ? meas_x : pred_x * (1.0 - alpha) + meas_x * alpha;
                    robot_x_ = smooth_x_;
                }
                if (count_y > 0) {
                    smooth_y_ = first_localize_ ? meas_y : pred_y * (1.0 - alpha) + meas_y * alpha;
                    robot_y_ = smooth_y_;
                }
                
                reject_count = 0;
                first_localize_ = false;
            }
        }

        double out_x = -robot_y_;  // 向右 = 负的向左
        double out_y = robot_x_;   // 向前 = 内部的 X
        
        double out_roll  = -cur_pitch_; 
        double out_pitch = cur_roll_;
        double out_yaw   = cur_yaw_ + M_PI / 2.0; // Yaw 补偿 90 度

        geometry_msgs::msg::TransformStamped t;
        t.header.stamp    = hdr.stamp;
        t.header.frame_id = "map";
        t.child_frame_id  = "odom";
        t.transform.translation.x = out_x;
        t.transform.translation.y = out_y;
        t.transform.translation.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(out_roll, out_pitch, out_yaw);
        t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
        tf_br_->sendTransform(t);

        
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 100,
            "🤖 [Hipnuc纯积分] X: %5.2f | Y: %5.2f | 纯净航向角: %5.1f°",
            out_x, out_y, out_yaw * 180.0 / M_PI);
    }

    void publish_markers(const std::vector<pcl::ModelCoefficients>& lines, std_msgs::msg::Header hdr)
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

        for (size_t i = 0; i < lines.size(); ++i) {
            visualization_msgs::msg::Marker lm;
            lm.header = hdr; lm.ns = "walls"; lm.id = i + 2;
            lm.type = visualization_msgs::msg::Marker::LINE_LIST;
            lm.action = visualization_msgs::msg::Marker::ADD;
            lm.scale.x = 0.08; lm.color.g = 1.0; lm.color.a = 0.9;
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