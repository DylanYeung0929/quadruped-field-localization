#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"

// 创建一个简单的节点类
class Mid360TestNode : public rclcpp::Node {
public:
    Mid360TestNode() : Node("mid360_test_node") {
        RCLCPP_INFO(this->get_logger(), "测试节点已启动！正在等待 Mid-360 数据...");

        // 订阅 /livox/imu 话题
        // 这里的 10 是队列大小
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/livox/imu", 
            10, 
            std::bind(&Mid360TestNode::imu_callback, this, std::placeholders::_1));
    }

private:
    // 当收到 IMU 数据时，会执行这个函数
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        // 打印接收到的加速度数据
        RCLCPP_INFO(this->get_logger(), 
            "收到雷达数据! [加速度 X: %.2f, Y: %.2f, Z: %.2f]", 
            msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z);
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
};

int main(int argc, char **argv) {
    // 初始化 ROS2
    rclcpp::init(argc, argv);
    // 启动节点并自旋（保持运行）
    rclcpp::spin(std::make_shared<Mid360TestNode>());
    // 关闭
    rclcpp::shutdown();
    return 0;
}