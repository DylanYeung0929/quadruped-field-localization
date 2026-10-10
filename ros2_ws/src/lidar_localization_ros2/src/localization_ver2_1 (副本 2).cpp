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

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/Dense>

//DYLAN YEUNG: MODIFIED ON 18/3/2026, USABLE WHEN INVERTED& HORIZONTAL

class Mid360LocalizationNode : public rclcpp::Node
{
public:
    Mid360LocalizationNode() : Node("mid360_localization_node")
    {
        auto sensor_qos = rclcpp::SensorDataQoS();

        // 1. 订阅
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/livox/imu", sensor_qos, std::bind(&Mid360LocalizationNode::imu_callback, this, std::placeholders::_1));

        custom_cloud_sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/livox/lidar", sensor_qos, std::bind(&Mid360LocalizationNode::custom_cloud_callback, this, std::placeholders::_1));

        // 2. 发布
        debug_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/localization/debug_cloud", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/localization/markers", 10);
        
        // 3. TF 广播器
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(this->get_logger(), "🚀 定位节点(稳定2D版)已启动！");
        RCLCPP_INFO(this->get_logger(), "💡 高度过滤：绝对世界 [20cm, 70cm] | 拟合上限：3条直线");
    }

private:
    double cur_roll_ = 0.0, cur_pitch_ = 0.0, cur_yaw_ = 0.0;
    double robot_x_ = 0.0, robot_y_ = 0.0;
    bool imu_inited_ = false;

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
        // ==========================================
        // 1. 姿态对齐 (消除颠簸并恢复重力朝向)
        // 雷达倒置时，IMU Roll约180度，逆旋转后Z轴朝上
        // ==========================================
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(-cur_pitch_, Eigen::Vector3f::UnitY()));
        transform.rotate(Eigen::AngleAxisf(-cur_roll_, Eigen::Vector3f::UnitX()));
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud, *aligned_cloud, transform);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (auto &p : aligned_cloud->points) {
            // ==========================================
            // 2. 绝对高度过滤
            // 对齐后Z轴朝上，雷达原点高度为绝对世界0.5m
            // 目标绝对高度：[0.2m, 0.7m]
            // 相对于雷达Z：[0.2-0.5, 0.7-0.5] = [-0.3m, 0.2m]
            // ==========================================
            if (p.z > -0.30 && p.z < 0.20 && !is_self_point(p.x, p.y)) {
                float dist_sq = p.x*p.x + p.y*p.y;
                if (dist_sq > 0.09 && dist_sq < 64.0) {
                    pcl::PointXYZ p2d = p; 
                    p2d.z = 0.0; // 降维打击，压扁到2D平面
                    filtered_cloud->push_back(p2d);
                }
            }
        }

        if (filtered_cloud->size() < 40) return;

        sensor_msgs::msg::PointCloud2 debug_msg;
        pcl::toROSMsg(*filtered_cloud, debug_msg);
        debug_msg.header = header;
        debug_cloud_pub_->publish(debug_msg);

        solve_geometry(filtered_cloud, header);
    }

    void solve_geometry(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std_msgs::msg::Header header)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE); 
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.06); 
        seg.setMaxIterations(200);

        std::vector<pcl::ModelCoefficients> lines;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        int original_size = cloud->size();
        int extract_attempts = 0;

        // ==========================================
        // 3. 约束拟合数量：结合 6x4m 矩形先验，提取最多 3 条有效直线
        // ==========================================
        // 允许 RANSAC 最多尝试提取 6 次，从中筛选出 3 条符合矩形约束的线
        while (cloud->size() > 0.05 * original_size && lines.size() < 3 && extract_attempts < 6) {
            extract_attempts++;
            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coeff);
            
            if (inliers->indices.size() < 30) break; 
            
            // --- 几何先验约束判断 (6m x 4m 矩形特征) ---
            bool is_valid = true;
            if (!lines.empty()) {
                Eigen::Vector2f dir_curr(coeff->values[3], coeff->values[4]);
                dir_curr.normalize();
                Eigen::Vector2f pt_curr(coeff->values[0], coeff->values[1]);

                // 与已提取的真实墙壁进行比对
                for (const auto& prev_line : lines) {
                    Eigen::Vector2f dir_prev(prev_line.values[3], prev_line.values[4]);
                    dir_prev.normalize();
                    Eigen::Vector2f pt_prev(prev_line.values[0], prev_line.values[1]);

                    // 计算方向向量的内积 (绝对值代表 cos(夹角))
                    float dot = std::abs(dir_curr.dot(dir_prev));
                    
                    // 1. 角度约束：必须平行(夹角<7度, dot>0.992)或垂直(夹角83~97度, dot<0.122)
                    if (dot >= 0.122 && dot <= 0.992) {
                        is_valid = false;
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                            "⚠️ 剔除干扰直线：角度不符合矩形正交特征 (偏角>7度, 点积=%.3f)", dot);
                        break;
                    }

                    // 2. 距离约束：若是平行线，距离必须约为 4m 或 6m
                    if (dot > 0.992) {
                        // 求出平行线的法向量
                        Eigen::Vector2f normal_prev(-dir_prev.y(), dir_prev.x());
                        // 计算两条平行线之间的垂直距离
                        float distance = std::abs((pt_curr - pt_prev).dot(normal_prev));
                        
                        // 允许 ±0.6m 的测量与拟合误差
                        bool is_4m = (distance > 3.4 && distance < 4.6);
                        bool is_6m = (distance > 5.4 && distance < 6.6);
                        
                        if (!is_4m && !is_6m) {
                            is_valid = false;
                            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                "⚠️ 剔除干扰直线：平行但距离错误 (%.2fm，非 4m/6m)", distance);
                            break;
                        }
                    }
                }
            }

            // 如果满足所有约束，将其加入墙壁集合
            if (is_valid) {
                lines.push_back(*coeff);
            }
            
            // 无论这条线是否被判定为墙壁，都把它的内点剔除，防止 RANSAC 卡死在这条线上
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud);
        }

        if (lines.empty()) return;

        visualization_msgs::msg::MarkerArray marker_array;
        
        // --- 新增：绘制雷达当前位置 (红色圆柱体) ---
        visualization_msgs::msg::Marker lidar_marker;
        lidar_marker.header = header;
        lidar_marker.ns = "lidar_position";
        lidar_marker.id = 0;
        lidar_marker.type = visualization_msgs::msg::Marker::CYLINDER;
        lidar_marker.action = visualization_msgs::msg::Marker::ADD;
        lidar_marker.scale.x = 0.2; // 圆柱体直径 20cm
        lidar_marker.scale.y = 0.2;
        lidar_marker.scale.z = 0.1; // 厚度 10cm
        lidar_marker.color.r = 1.0; 
        lidar_marker.color.g = 0.0; 
        lidar_marker.color.b = 0.0; 
        lidar_marker.color.a = 1.0; // 红色实体
        lidar_marker.pose.position.x = 0.0; // 雷达永远位于自身坐标系的原点
        lidar_marker.pose.position.y = 0.0;
        lidar_marker.pose.position.z = 0.0;
        marker_array.markers.push_back(lidar_marker);

        // --- 新增：绘制雷达前方朝向 (橙色箭头) ---
        visualization_msgs::msg::Marker heading_marker;
        heading_marker.header = header;
        heading_marker.ns = "lidar_heading";
        heading_marker.id = 1;
        heading_marker.type = visualization_msgs::msg::Marker::ARROW;
        heading_marker.action = visualization_msgs::msg::Marker::ADD;
        heading_marker.scale.x = 0.5; // 箭头指示前方 50cm
        heading_marker.scale.y = 0.05; // 箭头杆的宽度
        heading_marker.scale.z = 0.05; // 箭头杆的高度
        heading_marker.color.r = 1.0; 
        heading_marker.color.g = 0.5; 
        heading_marker.color.b = 0.0; 
        heading_marker.color.a = 1.0; // 橙色
        heading_marker.pose.position.x = 0.0;
        heading_marker.pose.position.y = 0.0;
        heading_marker.pose.position.z = 0.0;
        marker_array.markers.push_back(heading_marker);

        // --- 可视化拟合出的墙壁直线 ---
        for (size_t i = 0; i < lines.size(); ++i) {
            visualization_msgs::msg::Marker line_marker;
            line_marker.header = header;
            line_marker.ns = "fitted_walls";
            line_marker.id = i;
            line_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            line_marker.action = visualization_msgs::msg::Marker::ADD;
            line_marker.scale.x = 0.05; // 线宽
            line_marker.color.r = 0.0; line_marker.color.g = 1.0; line_marker.color.b = 0.0; line_marker.color.a = 0.8;
            
            float x0 = lines[i].values[0], y0 = lines[i].values[1];
            float dx = lines[i].values[3], dy = lines[i].values[4];
            geometry_msgs::msg::Point p1, p2;
            p1.x = x0 - dx * 5.0; p1.y = y0 - dy * 5.0; p1.z = 0;
            p2.x = x0 + dx * 5.0; p2.y = y0 + dy * 5.0; p2.z = 0;
            line_marker.points.push_back(p1);
            line_marker.points.push_back(p2);
            marker_array.markers.push_back(line_marker);
        }

        // --- 计算交点（墙角）并可视化 ---
        int corner_id = 0;
        for (size_t i = 0; i < lines.size(); ++i) {
            for (size_t j = i + 1; j < lines.size(); ++j) {
                float dx1 = lines[i].values[3], dy1 = lines[i].values[4];
                float dx2 = lines[j].values[3], dy2 = lines[j].values[4];
                float dot = dx1 * dx2 + dy1 * dy2;
                if (std::abs(dot) < 0.3) {
                    float x1 = lines[i].values[0], y1 = lines[i].values[1];
                    float x2 = lines[j].values[0], y2 = lines[j].values[1];
                    float det = dx2 * dy1 - dx1 * dy2;
                    if (std::abs(det) > 1e-4) {
                        float t = (dx2 * (y2 - y1) - dy2 * (x2 - x1)) / det;
                        visualization_msgs::msg::Marker corner_marker;
                        corner_marker.header = header;
                        corner_marker.ns = "corners";
                        corner_marker.id = corner_id++;
                        corner_marker.type = visualization_msgs::msg::Marker::SPHERE;
                        corner_marker.scale.x = 0.2; corner_marker.scale.y = 0.2; corner_marker.scale.z = 0.2;
                        corner_marker.color.r = 1.0; corner_marker.color.g = 1.0; corner_marker.color.b = 0.0; corner_marker.color.a = 1.0;
                        corner_marker.pose.position.x = x1 + t * dx1;
                        corner_marker.pose.position.y = y1 + t * dy1;
                        marker_array.markers.push_back(corner_marker);
                    }
                }
            }
        }
        marker_pub_->publish(marker_array);

        // --- 位置解算 ---
        double sum_x = 0, sum_y = 0;
        int count_x = 0, count_y = 0;
        float cos_y = std::cos(cur_yaw_), sin_y = std::sin(cur_yaw_);

        for (const auto& line : lines) {
            float x0 = line.values[0], y0 = line.values[1];
            float dx = line.values[3], dy = line.values[4];
            
            // 计算原点(雷达)到直线的垂足向量，该向量绝对始终从雷达指向墙壁！
            float proj = x0 * dx + y0 * dy;
            float c_x = x0 - proj * dx;
            float c_y = y0 - proj * dy;
            float dist = std::sqrt(c_x * c_x + c_y * c_y);

            if (dist < 0.05) continue; // 防御极端贴脸情况

            // 将指向墙壁的垂足向量旋转到绝对世界坐标系
            float world_c_x = c_x * cos_y - c_y * sin_y;
            float world_c_y = c_x * sin_y + c_y * cos_y;

            // 归一化用于方向判定
            float dir_x = world_c_x / dist;
            float dir_y = world_c_y / dist;

            // 根据向量指向，100% 准确地反推雷达坐标
            if (std::abs(dir_x) > 0.8) {
                // 垂足向量平行于世界 X 轴
                if (dir_x > 0) { 
                    // 向量朝正向 -> 说明这是前墙 (X=6)
                    sum_x += (6.0 - dist); 
                } else {         
                    // 向量朝负向 -> 说明这是后墙 (X=0)
                    sum_x += dist;         
                }
                count_x++;
            } else if (std::abs(dir_y) > 0.8) {
                // 垂足向量平行于世界 Y 轴
                if (dir_y > 0) { 
                    // 向量朝正向 -> 说明这是左墙 (Y=2)
                    sum_y += (2.0 - dist); 
                } else {         
                    // 向量朝负向 -> 说明这是右墙 (Y=-2)
                    sum_y += (-2.0 + dist); 
                }
                count_y++;
            }
        }

        if (count_x > 0) robot_x_ = sum_x / count_x;
        if (count_y > 0) robot_y_ = sum_y / count_y;

        // --- 广播 TF 变换 (map -> livox_frame) ---
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
            "🤖 [场地绝对坐标] X: %5.2f, Y: %5.2f | 墙数: %ld | Yaw: %6.1f°", robot_x_, robot_y_, lines.size(), cur_yaw_ * 180.0 / M_PI);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Mid360LocalizationNode>());
    rclcpp::shutdown();
    return 0;
}