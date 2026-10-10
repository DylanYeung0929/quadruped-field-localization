#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// PCL 相关头文件
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

//DYLANYEUNG:MODIFIED ON 21/3/2026, RULING OUT THE INTERFERENCE OF GROUND; FATAL: PROJECTION ERROR BITWEEN X& Y AXIS(NO IMU CALCULUS)

// 定义用于高程统计的 2D 网格结构
struct GridCell {
    float min_z;
    float max_z;
};

class Mid360LocalizationNode : public rclcpp::Node
{
public:
    Mid360LocalizationNode() : Node("mid360_localization_node")
    {
        auto sensor_qos = rclcpp::SensorDataQoS();

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/livox/imu", sensor_qos, std::bind(&Mid360LocalizationNode::imu_callback, this, std::placeholders::_1));

        custom_cloud_sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/livox/lidar", sensor_qos, std::bind(&Mid360LocalizationNode::custom_cloud_callback, this, std::placeholders::_1));

        debug_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/localization/debug_cloud", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/localization/markers", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // 初始化 20m x 20m 的高程网格 (分辨率 10cm)
        grid_.resize(200 * 200);

        RCLCPP_INFO(this->get_logger(), "定位节点(栅格高程防地版)已启动");
        RCLCPP_INFO(this->get_logger(), "采用底层网格 Z 轴跨度计算，彻底蒸发地面噪点。");
    }

private:
    double cur_roll_ = 0.0, cur_pitch_ = 0.0, cur_yaw_ = 0.0;
    double robot_x_ = 0.0, robot_y_ = 0.0;
    bool imu_inited_ = false;

    std::vector<GridCell> grid_; // 避免在回调中重复分配内存

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr custom_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    bool is_self_point(float x, float y) {
        return (x > -0.7 && x < 0.2 && y > -0.3 && y < 0.3);
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        tf2::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
        tf2::Matrix3x3 m(q);
        m.getRPY(cur_roll_, cur_pitch_, cur_yaw_);
        imu_inited_ = true;
    }

    void custom_cloud_callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        if (!imu_inited_) return;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto &p : msg->points) {
            cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
        }
        process_localization(cloud, msg->header);
    }

    void process_localization(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std_msgs::msg::Header header)
    {
        // 1. 姿态对齐
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(-cur_pitch_, Eigen::Vector3f::UnitY()));
        transform.rotate(Eigen::AngleAxisf(-cur_roll_, Eigen::Vector3f::UnitX()));
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud, *aligned_cloud, transform);

        // ==========================================
        // 2. 网格化 Z 轴跨度滤波 (物理消灭地面)
        // ==========================================
        // 清空重置网格
        for (auto& cell : grid_) {
            cell.min_z = 999.0f;
            cell.max_z = -999.0f;
        }

        // 第 1 遍遍历：统计每个 10cmx10cm 网格内的最低点和最高点
        for (const auto& p : aligned_cloud->points) {
            if (p.z < -0.4f || p.z > 0.4f || is_self_point(p.x, p.y)) continue;

            int gx = static_cast<int>((p.x + 10.0f) / 0.1f); // 偏移 10m 以防负坐标
            int gy = static_cast<int>((p.y + 10.0f) / 0.1f);

            if (gx >= 0 && gx < 200 && gy >= 0 && gy < 200) {
                int idx = gx * 200 + gy;
                if (p.z < grid_[idx].min_z) grid_[idx].min_z = p.z;
                if (p.z > grid_[idx].max_z) grid_[idx].max_z = p.z;
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr wall_cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        
        // 第 2 遍遍历：只提取高低落差 >= 15cm 的网格内的点（绝对是墙！），并将其降维到 2D
        for (const auto& p : aligned_cloud->points) {
            if (p.z < -0.4f || p.z > 0.4f || is_self_point(p.x, p.y)) continue;

            int gx = static_cast<int>((p.x + 10.0f) / 0.1f);
            int gy = static_cast<int>((p.y + 10.0f) / 0.1f);

            if (gx >= 0 && gx < 200 && gy >= 0 && gy < 200) {
                int idx = gx * 200 + gy;
                // 判断：如果该列点云 Z 轴跨度超过 15cm，确认为垂直墙面
                if (grid_[idx].max_z - grid_[idx].min_z >= 0.15f) {
                    pcl::PointXYZ p2d = p;
                    p2d.z = 0.0f; // 安全降维
                    wall_cloud_2d->push_back(p2d);
                }
            }
        }

        if (wall_cloud_2d->size() < 40) return;

        // ==========================================
        // 3. 2D 体素滤波，使墙面线条纤细平滑
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr clean_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(wall_cloud_2d);
        vg.setLeafSize(0.05f, 0.05f, 0.05f); // 5cm 分辨率
        vg.filter(*clean_cloud);

        if (clean_cloud->size() < 20) return;

        // 发布绝对纯净的墙面供 RViz 查看（此时画面应该只剩干净的墙体轮廓，毫无杂草）
        sensor_msgs::msg::PointCloud2 debug_msg;
        pcl::toROSMsg(*clean_cloud, debug_msg);
        debug_msg.header = header;
        debug_cloud_pub_->publish(debug_msg);

        solve_geometry(clean_cloud, header);
    }

    void solve_geometry(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std_msgs::msg::Header header)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE); 
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.05); 
        seg.setMaxIterations(200); 

        std::vector<pcl::ModelCoefficients> lines;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        int original_size = cloud->size();
        int extract_attempts = 0;

        // 在绝对纯净的数据上，提取 3 条直线轻而易举
        while (cloud->size() > 0.05 * original_size && lines.size() < 3 && extract_attempts < 6) {
            extract_attempts++;
            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
            
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coeff);
            
            if (inliers->indices.size() < 15) break; 
            
            // --- 2D 跨度计算 ---
            float min_t = 999.0, max_t = -999.0;
            float x0 = coeff->values[0], y0 = coeff->values[1];
            float dx = coeff->values[3], dy = coeff->values[4];
            
            float norm = std::sqrt(dx*dx + dy*dy);
            dx /= norm; dy /= norm;
            coeff->values[3] = dx; coeff->values[4] = dy;

            for (int idx : inliers->indices) {
                float px = cloud->points[idx].x - x0;
                float py = cloud->points[idx].y - y0;
                float t = px * dx + py * dy; 
                if (t < min_t) min_t = t;
                if (t > max_t) max_t = t;
            }
            float line_span = max_t - min_t;

            bool is_valid = true;
            if (line_span < 1.0) {  // 墙壁至少1米长
                is_valid = false;
            }

            // 矩形拓扑验证 (严格的 7 度角与 4m/6m 距离限制)
            if (is_valid && !lines.empty()) {
                Eigen::Vector2f dir_curr(dx, dy);
                Eigen::Vector2f pt_curr(x0, y0);

                for (const auto& prev_line : lines) {
                    Eigen::Vector2f dir_prev(prev_line.values[3], prev_line.values[4]);
                    Eigen::Vector2f pt_prev(prev_line.values[0], prev_line.values[1]);

                    float dot = std::abs(dir_curr.dot(dir_prev));
                    
                    // 1. 角度约束：必须平行(夹角<7度, dot>0.992)或垂直(夹角83~97度, dot<0.122)
                    if (dot >= 0.122 && dot <= 0.992) {
                        is_valid = false;
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                            "剔除干扰直线：角度不符合矩形正交特征 (偏角>7度, 点积=%.3f)", dot);
                        break;
                    }

                    // 2. 距离约束：若是平行线，距离必须约为 4m 或 6m
                    if (dot > 0.992) {
                        Eigen::Vector2f normal_prev(-dir_prev.y(), dir_prev.x());
                        float distance = std::abs((pt_curr - pt_prev).dot(normal_prev));
                        
                        // 允许 ±0.6m 的测量与拟合误差
                        bool is_4m = (distance > 3.4 && distance < 4.6);
                        bool is_6m = (distance > 5.4 && distance < 6.6);
                        
                        if (!is_4m && !is_6m) {
                            is_valid = false;
                            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                "剔除干扰直线：平行但距离错误 (%.2fm，非 4m/6m)", distance);
                            break;
                        }
                    }
                }
            }

            if (is_valid) {
                pcl::ModelCoefficients final_coeff = *coeff;
                final_coeff.values.push_back(min_t); // Index 6
                final_coeff.values.push_back(max_t); // Index 7
                lines.push_back(final_coeff);
            }
            
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud);
        }

        if (lines.empty()) return;

        // --- 画图模块 ---
        visualization_msgs::msg::MarkerArray marker_array;
        
        visualization_msgs::msg::Marker lidar_marker;
        lidar_marker.header = header; lidar_marker.ns = "lidar_position"; lidar_marker.id = 0;
        lidar_marker.type = visualization_msgs::msg::Marker::CYLINDER; lidar_marker.action = visualization_msgs::msg::Marker::ADD;
        lidar_marker.scale.x = 0.2; lidar_marker.scale.y = 0.2; lidar_marker.scale.z = 0.1; 
        lidar_marker.color.r = 1.0; lidar_marker.color.g = 0.0; lidar_marker.color.b = 0.0; lidar_marker.color.a = 1.0; 
        lidar_marker.pose.position.x = 0.0; lidar_marker.pose.position.y = 0.0; lidar_marker.pose.position.z = 0.0;
        marker_array.markers.push_back(lidar_marker);

        visualization_msgs::msg::Marker heading_marker;
        heading_marker.header = header; heading_marker.ns = "lidar_heading"; heading_marker.id = 1;
        heading_marker.type = visualization_msgs::msg::Marker::ARROW; heading_marker.action = visualization_msgs::msg::Marker::ADD;
        heading_marker.scale.x = 0.5; heading_marker.scale.y = 0.05; heading_marker.scale.z = 0.05; 
        heading_marker.color.r = 1.0; heading_marker.color.g = 0.5; heading_marker.color.b = 0.0; heading_marker.color.a = 1.0; 
        heading_marker.pose.position.x = 0.0; heading_marker.pose.position.y = 0.0; heading_marker.pose.position.z = 0.0;
        marker_array.markers.push_back(heading_marker);

        for (size_t i = 0; i < lines.size(); ++i) {
            visualization_msgs::msg::Marker line_marker;
            line_marker.header = header; line_marker.ns = "fitted_walls"; line_marker.id = i + 2;
            line_marker.type = visualization_msgs::msg::Marker::LINE_LIST; line_marker.action = visualization_msgs::msg::Marker::ADD;
            line_marker.scale.x = 0.08; line_marker.color.r = 0.0; line_marker.color.g = 1.0; line_marker.color.b = 0.0; line_marker.color.a = 0.9;
            
            float x0 = lines[i].values[0], y0 = lines[i].values[1];
            float dx = lines[i].values[3], dy = lines[i].values[4];
            
            float min_t = lines[i].values.size() > 6 ? lines[i].values[6] : -3.0;
            float max_t = lines[i].values.size() > 7 ? lines[i].values[7] : 3.0;
            
            geometry_msgs::msg::Point p1, p2;
            p1.x = x0 + dx * min_t; p1.y = y0 + dy * min_t; p1.z = 0;
            p2.x = x0 + dx * max_t; p2.y = y0 + dy * max_t; p2.z = 0;
            line_marker.points.push_back(p1); line_marker.points.push_back(p2);
            marker_array.markers.push_back(line_marker);
        }

        marker_pub_->publish(marker_array);

        double sum_x = 0, sum_y = 0;
        int count_x = 0, count_y = 0;
        float cos_y = std::cos(cur_yaw_), sin_y = std::sin(cur_yaw_);

        for (const auto& line : lines) {
            float x0 = line.values[0], y0 = line.values[1];
            float dx = line.values[3], dy = line.values[4];
            
            float proj = x0 * dx + y0 * dy;
            float c_x = x0 - proj * dx;
            float c_y = y0 - proj * dy;
            float dist = std::sqrt(c_x * c_x + c_y * c_y);

            if (dist < 0.05) continue;

            float world_c_x = c_x * cos_y - c_y * sin_y;
            float world_c_y = c_x * sin_y + c_y * cos_y;

            float dir_x = world_c_x / dist;
            float dir_y = world_c_y / dist;

            if (std::abs(dir_x) > 0.8) {
                if (dir_x > 0) sum_x += (6.0 - dist); else sum_x += dist;
                count_x++;
            } else if (std::abs(dir_y) > 0.8) {
                if (dir_y > 0) sum_y += (2.0 - dist); else sum_y += (-2.0 + dist);
                count_y++;
            }
        }

        if (count_x > 0) robot_x_ = robot_x_ * 0.5 + (sum_x / count_x) * 0.5;
        if (count_y > 0) robot_y_ = robot_y_ * 0.5 + (sum_y / count_y) * 0.5;

        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = header.stamp;
        t.header.frame_id = "map";
        t.child_frame_id = header.frame_id; 
        t.transform.translation.x = robot_x_;
        t.transform.translation.y = robot_y_;
        t.transform.translation.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(cur_roll_, cur_pitch_, cur_yaw_);
        t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(t);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
            "[绝对定位] X: %5.2f, Y: %5.2f | 墙数: %ld", robot_x_, robot_y_, lines.size());
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Mid360LocalizationNode>());
    rclcpp::shutdown();
    return 0;
}