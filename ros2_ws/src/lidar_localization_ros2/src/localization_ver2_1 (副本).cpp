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

        RCLCPP_INFO(this->get_logger(), "🚀 定位节点(可视化版)已启动！");
        RCLCPP_INFO(this->get_logger(), "💡 请在 RViz2 中将 Fixed Frame 设置为 'map'");
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
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(-cur_pitch_, Eigen::Vector3f::UnitY()));
        transform.rotate(Eigen::AngleAxisf(-cur_roll_, Eigen::Vector3f::UnitX()));
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud, *aligned_cloud, transform);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (auto &p : aligned_cloud->points) {
            if (p.z > 0.0 && p.z < 0.4 && !is_self_point(p.x, p.y)) {
                float dist_sq = p.x*p.x + p.y*p.y;
                if (dist_sq > 0.09 && dist_sq < 64.0) {
                    pcl::PointXYZ p2d = p; p2d.z = 0.0; 
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

        while (cloud->size() > 0.1 * original_size && lines.size() < 4) {
            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coeff);
            if (inliers->indices.size() < 30) break; 
            lines.push_back(*coeff);
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud);
        }

        if (lines.empty()) return;

        visualization_msgs::msg::MarkerArray marker_array;
        
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
            float dx = line.values[3], dy = line.values[4];
            Eigen::Vector2f local_normal(-dy, dx); local_normal.normalize();
            float dist = std::abs(line.values[0] * local_normal.x() + line.values[1] * local_normal.y());
            Eigen::Vector2f world_normal;
            world_normal.x() = local_normal.x() * cos_y - local_normal.y() * sin_y;
            world_normal.y() = local_normal.x() * sin_y + local_normal.y() * cos_y;

            if (std::abs(world_normal.x()) > 0.8) {
                if (world_normal.x() > 0) sum_x += (6.0 - dist); else sum_x += dist;
                count_x++;
            } else if (std::abs(world_normal.y()) > 0.8) {
                if (world_normal.y() > 0) sum_y += (2.0 - dist); else sum_y += (-2.0 + dist);
                count_y++;
            }
        }

        if (count_x > 0) robot_x_ = sum_x / count_x;
        if (count_y > 0) robot_y_ = sum_y / count_y;

        // --- 广播 TF 变换 (map -> livox_frame) ---
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = header.stamp;
        t.header.frame_id = "map";
        t.child_frame_id = header.frame_id; // 通常是 livox_frame
        t.transform.translation.x = robot_x_;
        t.transform.translation.y = robot_y_;
        t.transform.translation.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(cur_roll_, cur_pitch_, cur_yaw_);
        t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(t);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
            "🤖 [场地绝对坐标] X: %5.2f, Y: %5.2f | Yaw: %6.1f°", robot_x_, robot_y_, cur_yaw_ * 180.0 / M_PI);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Mid360LocalizationNode>());
    rclcpp::shutdown();
    return 0;
}