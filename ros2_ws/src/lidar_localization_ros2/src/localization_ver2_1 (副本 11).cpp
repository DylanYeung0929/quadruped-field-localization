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

#include <Eigen/Dense>



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

        // ==========================================================
        // ★ 1. 初始化 H 矩阵（放在构造函数内）
        // ==========================================================
        // H 矩阵大小为 3行7列
        // 建立测量值[x, y, yaw]与状态量[x, y, yaw, vx, vy, bias_ax, bias_ay]的映射
        H_.setZero();
        H_(0, 0) = 1.0; // 测量值 x 对应 状态量 x
        H_(1, 1) = 1.0; // 测量值 y 对应 状态量 y
        H_(2, 2) = 1.0; // 测量值 yaw 对应 状态量 yaw

        // 顺便在这里初始化观测噪声矩阵 R_ (3x3)
        R_.setZero();
        R_(0, 0) = 0.01; // 雷达 X 观测方差（值越小代表越信任雷达位置）
        R_(1, 1) = 0.01; // 雷达 Y 观测方差
        R_(2, 2) = 0.05; // 雷达 Yaw 观测方差

        // 顺便在这里初始化过程噪声矩阵 Q_ (7x7)
        Q_.setIdentity() * 0.001; 
        Q_(5, 5) = 0.0001; // 零偏 bias_ax 的随机游走噪声
        Q_(6, 6) = 0.0001; // 零偏 bias_ay 的随机游走噪声
    }


private:
    double cur_roll_  = 0.0;
    double cur_pitch_ = 0.0;
    double cur_yaw_   = 0.0;  
    bool   imu_ready_ = false;

    // --- 纯积分核心变量 ---
    rclcpp::Time last_imu_time_;
    bool   first_imu_ = true;
    double pure_yaw_ = 0.0; 

    double robot_x_ = 0.0;
    double robot_y_ = 0.0;
    double smooth_x_ = 0.0, smooth_y_ = 0.0;
    bool   first_localize_ = true;
    bool   is_turning_ = false;

    using Vector7d = Eigen::Matrix<double, 7, 1>;
    using Matrix7d = Eigen::Matrix<double, 7, 7>;
    // 7维状态量: x, y, yaw, vx, vy, bias_ax, bias_ay
    Vector7d X_ = Vector7d::Zero(); 
    Matrix7d P_ = Matrix7d::Identity() * 0.1; // 状态协方差
    Matrix7d Q_ = Matrix7d::Zero();           // 过程噪声
    Eigen::Matrix<double, 3, 7> H_ = Eigen::Matrix<double, 3, 7>::Zero(); // 观测矩阵
    Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity() * 0.01; // 观测噪声(雷达非常准，设小点)

    bool is_initialized_ = false;

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
    // IMU 回调：获取绝对重力水平 + 绝对信赖 Hipnuc 自带角度 + 动力学防飘
    // ============================================================
    void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg) {
        imu_ready_ = true; // 打破死锁，通知雷达放行
        
        if (!is_initialized_) {
            last_imu_time_ = msg->header.stamp;
            return;
        }
        
        double dt = (rclcpp::Time(msg->header.stamp) - last_imu_time_).seconds();
        last_imu_time_ = msg->header.stamp;
        if (dt <= 0.0 || dt > 0.1) return;

        // ==========================================
        // 1. ★ 核心修复：从 Hipnuc 硬件四元数中提取精准的绝对姿态 ★
        // ==========================================
        tf2::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
        tf2::Matrix3x3 m(q);
        double raw_roll, raw_pitch, raw_yaw;
        m.getRPY(raw_roll, raw_pitch, raw_yaw);
        
        cur_roll_  = raw_roll * ROLL_SIGN;
        cur_pitch_ = raw_pitch * PITCH_SIGN;
        cur_yaw_   = wrap_pi(raw_yaw * YAW_SIGN); // ★ 赋予 cur_yaw_ 真正的灵魂，不再是 0.0！

        // ==========================================
        // 2. 精准重力扣除
        // ==========================================
        tf2::Vector3 gravity_world(0, 0, 9.80665);
        tf2::Vector3 gravity_local = tf2::quatRotate(q.inverse(), gravity_world);

        double ax_pure = msg->linear_acceleration.x - gravity_local.x() - X_(5);
        double ay_pure = msg->linear_acceleration.y - gravity_local.y() - X_(6);
        double wz = msg->angular_velocity.z * YAW_SIGN;

        // ==========================================
        // 3. 强制令 EKF 的 Yaw 状态量绝对同步于 Hipnuc 硬件直出
        // ==========================================
        X_(2) = cur_yaw_; // 彻底断绝卡尔曼对 Yaw 的漂移和误判

        // ==========================================
        // 4. 映射到全局地图坐标系 (使用绝对准确的 yaw)
        // ==========================================
        double cos_y = std::cos(cur_yaw_);
        double sin_y = std::sin(cur_yaw_);
        double ax_map = cos_y * ax_pure - sin_y * ay_pure;
        double ay_map = sin_y * ax_pure + cos_y * ay_pure;

        // ==========================================
        // 5. 动力学阻尼机制（防止速度积分发散）
        // ==========================================
        if (std::abs(wz) < 0.05 && std::abs(ax_pure) < 0.2 && std::abs(ay_pure) < 0.2) {
            X_(3) *= 0.85; // 静态强刹车
            X_(4) *= 0.85;
        } else {
            X_(3) *= 0.98; // 运动阻尼
            X_(4) *= 0.98;
        }

        // ==========================================
        // 6. EKF 位置与速度状态预测更新
        // ==========================================
        X_(0) += X_(3) * dt + 0.5 * ax_map * dt * dt;
        X_(1) += X_(4) * dt + 0.5 * ay_map * dt * dt;
        X_(3) += ax_map * dt;
        X_(4) += ay_map * dt;

        Matrix7d F = Matrix7d::Identity();
        F(0, 3) = dt; F(1, 4) = dt;
        F(0, 5) = -0.5 * cos_y * dt * dt; F(0, 6) = 0.5 * sin_y * dt * dt;
        F(1, 5) = -0.5 * sin_y * dt * dt; F(1, 6) = -0.5 * cos_y * dt * dt;
        F(3, 5) = -cos_y * dt;            F(3, 6) =  sin_y * dt;
        F(4, 5) = -sin_y * dt;            F(4, 6) = -cos_y * dt;
        
        P_ = F * P_ * F.transpose() + Q_;

        // ==========================================
        // 7. 打印与发布
        // ==========================================
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
            "🤖 [EKF] X: %5.2f | Y: %5.2f | Vx: %5.2f | Vy: %5.2f | Yaw: %5.1f°",
            X_(0), X_(1), X_(3), X_(4), X_(2) * 180.0 / M_PI);

        publish_tf(X_(0), X_(1), X_(2), msg->header.stamp);
        publish_odometry(msg->header.stamp, msg->angular_velocity);
    }

    void publish_tf(double x, double y, double yaw, const rclcpp::Time & stamp)
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = "map";
        t.child_frame_id = "odom"; 

        t.transform.translation.x = -y;
        t.transform.translation.y = x;
        t.transform.translation.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(-cur_pitch_, cur_roll_, wrap_pi(yaw + M_PI / 2.0));
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        tf_br_->sendTransform(t);
    }

    void publish_odometry(const rclcpp::Time & stamp, const geometry_msgs::msg::Vector3 & ang_vel)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "map";
        odom.child_frame_id = "base_link";
        
        // 当前位置 (如果激活了定位就有值，没激活就是0)
        // 位置映射：使用 KF 算出的 X_(0) 和 X_(1)
        odom.pose.pose.position.x = -X_(1);
        odom.pose.pose.position.y = X_(0);
        
        // 姿态映射：使用 KF 算出的 Yaw 角 X_(2)
        tf2::Quaternion q;
        q.setRPY(-cur_pitch_, cur_roll_, wrap_pi(X_(2) + M_PI / 2.0));
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        // 速度映射：使用 KF 算出的无偏速度 X_(3) 和 X_(4)
        odom.twist.twist.linear.x = -X_(4);
        odom.twist.twist.linear.y = X_(3);
        odom.twist.twist.angular.z = ang_vel.z * YAW_SIGN;

        odom_pub_->publish(odom);
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

// ==========================================================
        // ★ 核心替换点：卡尔曼滤波的观测更新 (Measurement Update)
        // ==========================================================
        if (count_x > 0 || count_y > 0) {
            double meas_x = count_x > 0 ? sum_x / count_x : X_(0);
            double meas_y = count_y > 0 ? sum_y / count_y : X_(1);
            double meas_yaw = cur_yaw_;

            if (first_localize_) {
                // 首次定位，直接初始化状态量
                X_ << meas_x, meas_y, meas_yaw, 0.0, 0.0, 0.0, 0.0;
                is_initialized_ = true;
                first_localize_ = false;
                return;
            }

            // 【关键安全门控】计算雷达观测与当前卡尔曼预测的差距
            double dx = meas_x - X_(0);
            double dy = meas_y - X_(1);
            double dist_jump = std::sqrt(dx*dx + dy*dy);

            // 静态计数器：记录连续被拒收的帧数
            static int reject_count = 0;
            const int MAX_REJECT_FRAMES = 30; // 约 3 秒（基于雷达 10Hz）

            // 逻辑反转：大于阈值且没达到上限时，进行拦截
            if (dist_jump > 0.20 && reject_count < MAX_REJECT_FRAMES) { 
                reject_count++;
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500, 
                    "⚠️ 遭遇物块或跳变！隔离雷达数据，进入纯 IMU 盲走 (已拦截: %d 帧)", reject_count);
            } else {
                // 触发长时致盲的强制重定位
                if (reject_count >= MAX_REJECT_FRAMES) {
                    RCLCPP_ERROR(get_logger(), "🚨 达到最大盲走帧数，强制重定位！清除 IMU 累积漂移。");
                    // 强制硬重定位：强行拉回雷达绝对坐标，清空预测速度避免狂奔，但保留已算出的零偏
                    X_(0) = meas_x;
                    X_(1) = meas_y;
                    X_(2) = meas_yaw;
                    X_(3) = 0.0; 
                    X_(4) = 0.0;
                    P_ = Eigen::Matrix<double, 7, 7>::Identity() * 0.1; // 重置协方差矩阵置信度
                    reject_count = 0;
                    return; // 结束本帧，下一帧再恢复标准的 EKF 平滑更新
                }

                // 执行标准卡尔曼更新
                Eigen::Vector3d Z(meas_x, meas_y, meas_yaw);
                Eigen::Vector3d y = Z - H_ * X_; // 计算残差 (Innovation)
                
                // 处理角度环绕问题 (防止偏航角在 +-PI 附近跳变导致矩阵崩掉)
                while (y(2) > M_PI)  y(2) -= 2.0 * M_PI;
                while (y(2) < -M_PI) y(2) += 2.0 * M_PI;

                Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;
                Eigen::Matrix<double, 7, 3> K = P_ * H_.transpose() * S.inverse(); // 计算卡尔曼增益

                X_ = X_ + K * y; // 更新状态量（此时位置、速度、IMU零偏被同时修正！）
                P_ = (Eigen::Matrix<double, 7, 7>::Identity() - K * H_) * P_; // 收缩协方差
                
                reject_count = 0; // 数据正常被接纳，计数器清零
            }
        } // <--- 修复点：补全了 if (count_x > 0 || count_y > 0) 的闭合括号！
        
        // ==========================================================
        // ★ 注意：solve 函数到这里就可以直接结束了！
        // ==========================================================
    } // <--- 这是整个 solve 函数结束的括号
}; // <--- 这是整个类结束的括号

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Mid360LocalizationNode>());
    rclcpp::shutdown();
    return 0;
}