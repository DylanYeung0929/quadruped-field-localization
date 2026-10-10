#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
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
#include <cmath>

// DYLANYEUNG: v6.0 - 2026/03/24
// 【巅峰版：倾角补偿 + 陀螺仪/磁力计互补滤波融合】
// 1. [IMU预测] 利用陀螺仪角速度(Z轴)进行极平滑的高频姿态推演，无惧电机瞬间磁干扰。
// 2. [Mag更新] 利用倾角补偿后的三轴磁力计进行低频牵引，彻底消灭陀螺仪长期温漂。
// 3. [自适应权重] 当检测到剧烈转弯时，自动降低磁力计信任度，高度信任陀螺仪。

struct GridCell { float min_z, max_z; };

// 角度限制在 -PI 到 PI
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
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/imu", 10, std::bind(&Mid360LocalizationNode::imu_cb, this, std::placeholders::_1));

        mag_sub_ = create_subscription<sensor_msgs::msg::MagneticField>(
            "/magnetic_field", 10, std::bind(&Mid360LocalizationNode::mag_cb, this, std::placeholders::_1));

        auto sensor_qos = rclcpp::SensorDataQoS();
        cloud_sub_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/livox/lidar", sensor_qos, std::bind(&Mid360LocalizationNode::cloud_cb, this, std::placeholders::_1));

        dbg_pub_    = create_publisher<sensor_msgs::msg::PointCloud2>("/localization/debug_cloud", 10);
        marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/localization/markers", 10);
        tf_br_      = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        grid_.resize(200 * 200);
        RCLCPP_INFO(get_logger(), "[v6.0] 启动 — 互补滤波传感器融合：陀螺仪平滑预测 + 磁力计绝对校准！");
    }

private:
    double cur_roll_  = 0.0;
    double cur_pitch_ = 0.0;
    double cur_yaw_   = 0.0;  
    bool   imu_ready_ = false;

    // 互补滤波相关变量
    double last_imu_time_ = -1.0;
    double initial_mag_yaw_ = -1000.0; 
    bool   mag_initialized_ = false;

    double robot_x_ = 0.0;
    double robot_y_ = 0.0;

    double smooth_x_ = 0.0, smooth_y_ = 0.0;
    bool   first_localize_ = true;
    bool   is_turning_ = false;

    std::vector<GridCell> grid_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::MagneticField>::SharedPtr mag_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dbg_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_br_;

    // ============================================================
    // ★ 融合步骤 1：IMU 回调 (高频预测) ★
    // ============================================================
    void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        double ts = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        // 1. 提取重力方向 Roll, Pitch (用于后续的磁力计倾角补偿)
        tf2::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
        tf2::Matrix3x3 m(q);
        double dummy_yaw;
        m.getRPY(cur_roll_, cur_pitch_, dummy_yaw);

        // 检测是否在转弯 (用于调整磁力计权重)
        is_turning_ = std::abs(msg->angular_velocity.z) > 0.12;

        // 2. 互补滤波【预测阶段】：利用陀螺仪的极高平滑度进行积分推演
        if (mag_initialized_) {
            if (last_imu_time_ > 0) {
                double dt = ts - last_imu_time_;
                if (dt > 0 && dt < 0.1) {
                    // 角速度积分：完美顺滑，无惧瞬间磁场干扰
                    cur_yaw_ = wrap_pi(cur_yaw_ + msg->angular_velocity.z * dt);
                }
            }
        }
        
        last_imu_time_ = ts;
        imu_ready_ = true; 
    }

    // ============================================================
    // ★ 融合步骤 2：磁力计回调 (低频更新/牵引) ★
    // ============================================================
    void mag_cb(const sensor_msgs::msg::MagneticField::SharedPtr msg)
    {
        if (!imu_ready_) return;

        // 1. 倾角补偿：把由于机器狗倾斜导致的磁场畸变反向补偿回绝对水平面
        Eigen::Vector3d mag_raw(msg->magnetic_field.x, msg->magnetic_field.y, msg->magnetic_field.z);
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.rotate(Eigen::AngleAxisd(-cur_pitch_, Eigen::Vector3d::UnitY()));
        T.rotate(Eigen::AngleAxisd(-cur_roll_,  Eigen::Vector3d::UnitX()));
        Eigen::Vector3d mag_horiz = T * mag_raw; 

        // 2. 计算出当前的绝对磁北航向
        double current_absolute_yaw = -std::atan2(mag_horiz.y(), mag_horiz.x());

        // 3. 初始零点锚定：将开机姿态定义为 0 度
        if (!mag_initialized_) {
            initial_mag_yaw_ = current_absolute_yaw;
            cur_yaw_ = 0.0;
            mag_initialized_ = true;
            RCLCPP_INFO(get_logger(), "[互补滤波] 锚定初始磁方位: %.1f°，启动陀螺仪融合跟踪", initial_mag_yaw_ * 180.0 / M_PI);
            return;
        }

        // 4. 计算当前真实世界的目标相对航向
        double target_yaw = wrap_pi(current_absolute_yaw - initial_mag_yaw_);
        double err = wrap_pi(target_yaw - cur_yaw_);

        // 5. 互补滤波【更新阶段】：动态牵引
        //    如果此时机器狗在疯狂打转，电机电流磁场极大，我们极度降低磁力计信任度 (k=0.005)
        //    如果机器狗平稳行走，使用标准弱牵引 (k=0.02) 消除陀螺仪长期温漂
        double k = is_turning_ ? 0.005 : 0.02;
        cur_yaw_ = wrap_pi(cur_yaw_ + k * err); 
    }

    // ============================================================
    // 雷达回调 (保持原样)
    // ============================================================
    void cloud_cb(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        if (!imu_ready_ || !mag_initialized_) return;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& p : msg->points) cloud->push_back({p.x, p.y, p.z});
        process(cloud, msg->header);
    }

    bool is_self(float x, float y) { return (x > -0.7f && x < 0.2f && y > -0.3f && y < 0.3f); }

    void process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std_msgs::msg::Header hdr)
    {
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
            "[融合引擎] X:%.2f Y:%.2f | 互补滤波平滑偏航:%.1f°",
            robot_x_, robot_y_, cur_yaw_ * 180.0 / M_PI);
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