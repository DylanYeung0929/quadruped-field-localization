#include <lidar_localization/lidar_localization_component.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  
  // 使用完整的命名空间
  std::shared_ptr<lidar_localization::PCLLocalization> pcl_l = 
    std::make_shared<lidar_localization::PCLLocalization>(options);
  
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(pcl_l->get_node_base_interface());
  executor.spin();
  rclcpp::shutdown();
  
  return 0;
}