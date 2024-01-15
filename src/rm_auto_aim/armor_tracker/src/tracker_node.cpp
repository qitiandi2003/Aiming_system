// Copyright 2022 Chen Jun
#include "armor_tracker/tracker_node.hpp"
// STD
#include <memory>
#include <vector>

namespace rm_auto_aim
{
// EKF   定义了一个用于状态更新的过程函数 f，该函数用于更新目标状态的预测值
  // xa = x_armor, xc = x_robot_center
  // 状态信息: xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r
  // 测量信息: xa, ya, za, yaw
  // f - Process function
  //速度的更新是通过将速度的变化率乘以时间步长 dt_ 添加到当前速度上实现的
Eigen::VectorXd processFunction(const Eigen::VectorXd &x, double dt_) {
  Eigen::VectorXd x_new = x;
  x_new(0) += x(1) * dt_;
  x_new(2) += x(3) * dt_;
  x_new(4) += x(5) * dt_;
  x_new(6) += x(7) * dt_;
  return x_new;
}

 //过程函数的雅可比矩阵  用于在非线性优化
Eigen::MatrixXd jacobianProcessFunction(double dt_) {
  Eigen::MatrixXd f(9, 9);
    f <<  1,   dt_, 0,   0,   0,   0,   0,   0,   0,
          0,   1,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   1,   dt_, 0,   0,   0,   0,   0, 
          0,   0,   0,   1,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   1,   dt_, 0,   0,   0,
          0,   0,   0,   0,   0,   1,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   1,   dt_, 0,
          0,   0,   0,   0,   0,   0,   0,   1,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   1;

  
  return f;
}

//观测函数  将系统的状态映射到观测空间，以便与实际观测值进行比较
Eigen::VectorXd observationFunction(const Eigen::VectorXd &x) {
  Eigen::VectorXd z(4);
  double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
  z(0) = xc - r * cos(yaw);
  z(1) = yc - r * sin(yaw);
  z(2) = x(4);
  z(3) = x(6);
  return z;
}

//观测函数雅可比  用于表示观测函数对状态向量的偏导数
Eigen::MatrixXd jacobianObservationFunction(const Eigen::VectorXd &x) {
  Eigen::MatrixXd h(4, 9);
  double yaw = x(6), r = x(8);

    //    xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
    h <<  1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
          0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
          0,   0,   0,   0,   1,   0,   0,          0,   0,
          0,   0,   0,   0,   0,   0,   1,          0,   0;

  return h;
}


ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions & options)
: Node("armor_tracker", options)
{ 

  RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");

  // 声明参数
  max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);

  // Tracker 声明参数
  double max_match_distance = this->declare_parameter("tracker.max_match_distance", 0.20);
  double max_match_yaw_diff = this->declare_parameter("tracker.max_match_yaw_diff", 1.4);
  tracker_ = std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff);
  tracker_->tracking_thres = this->declare_parameter("tracker.tracking_thres", 5);
  lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 0.5);

  auto f = [this](const Eigen::VectorXd &x) { return processFunction(x, dt_); };
  auto j_f = [this](const Eigen::VectorXd &) { return jacobianProcessFunction(dt_); };
  auto h = [](const Eigen::VectorXd &x) { return observationFunction(x); };
  auto j_h = [](const Eigen::VectorXd &x) { return jacobianObservationFunction(x); };

  //表示过程噪声协方差矩阵 Q 的生成函数  用于描述系统模型中的不确定性或未建模的影响
  s2qxyz_ = declare_parameter("ekf.sigma2_q_xyz", 20.0);
  s2qyaw_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
  s2qr_ = declare_parameter("ekf.sigma2_q_r", 800.0);
  auto u_q = [this]() 
  {
    Eigen::MatrixXd q(9, 9);
    double t = dt_, x = s2qxyz_, y = s2qyaw_, r = s2qr_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * y;
    double q_r = pow(t, 4) / 4 * r;
   
    //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
    q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
          q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
          0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
          0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
          0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
          0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
          0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
          0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
          0,      0,      0,      0,      0,      0,      0,      0,      q_r;
 
    return q;
  };


  //更新测量噪声协方差矩阵（R）的函数  用于量化测量值的不确定性
  r_xyz_factor = declare_parameter("ekf.r_xyz_factor", 0.05);
  r_yaw = declare_parameter("ekf.r_yaw", 0.02);
  auto u_r = [this](const Eigen::VectorXd & z) {
    Eigen::DiagonalMatrix<double, 4> r;
    double x = r_xyz_factor;
    r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw;
    return r;
  };


  // 初始化卡尔曼滤波器中的状态估计误差协方差矩阵（P）
  Eigen::DiagonalMatrix<double, 9> p0;
  p0.setIdentity();
  tracker_->ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};

  //  用于重置跟踪器
  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  reset_tracker_srv_ = this->create_service<std_srvs::srv::Trigger>(
    "/tracker/reset", [this](
                        const std_srvs::srv::Trigger::Request::SharedPtr,
                        std_srvs::srv::Trigger::Response::SharedPtr response) {
      tracker_->tracker_state = Tracker::LOST;
      response->success = true;
      RCLCPP_INFO(this->get_logger(), "Tracker reset!");
      return;
    });

  
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());  //用于管理tf2中的坐标变换。
  

  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
    this->get_node_base_interface(), this->get_node_timers_interface());  
  tf2_buffer_->setCreateTimerInterface(timer_interface);  //用于设置 tf2_buffer_ 的计时器接口
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);  //用于监听tf2的坐标变换
 
 
  armors_sub_.subscribe(this, "/detector/armors", rmw_qos_profile_sensor_data);  //该话题用于接收装甲板信息
  target_frame_ = this->declare_parameter("target_frame", "odom");  //设置了目标坐标系的名称，默认为 "odom"
  tf2_filter_ = std::make_shared<tf2_filter>(
    armors_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
    this->get_node_clock_interface(), std::chrono::duration<int>(1));  //对接收到的装甲板信息进行坐标变换
  
 
  tf2_filter_->registerCallback(&ArmorTrackerNode::armorsCallback, this);  //在坐标变换可用时被调用

  //  创建了一个发布器，用于发布 auto_aim_interfaces::msg::TrackerInfo 类型的信息到 "/tracker/info" 话题
  info_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/tracker/info", 10);

  
  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
    "/tracker/target", rclcpp::SensorDataQoS());  //用于发布 auto_aim_interfaces::msg::Target 类型的信息到 "/tracker/target" 话题

  //  创建可视化标记的发布器
  // See http://wiki.ros.org/rviz/DisplayTypes/Marker
  position_marker_.ns = "position";
  position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
  position_marker_.color.a = 1.0;
  position_marker_.color.g = 1.0;
  linear_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  linear_v_marker_.ns = "linear_v";
  linear_v_marker_.scale.x = 0.03;
  linear_v_marker_.scale.y = 0.05;
  linear_v_marker_.color.a = 1.0;
  linear_v_marker_.color.r = 1.0;
  linear_v_marker_.color.g = 1.0;
  angular_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  angular_v_marker_.ns = "angular_v";
  angular_v_marker_.scale.x = 0.03;
  angular_v_marker_.scale.y = 0.05;
  angular_v_marker_.color.a = 1.0;
  angular_v_marker_.color.b = 1.0;
  angular_v_marker_.color.g = 1.0;
  armor_marker_.ns = "armors";
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.03;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.r = 1.0;
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracker/marker", 10);
}

void ArmorTrackerNode::armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
  //将装甲板的位置从图像坐标系转换到世界坐标系
  for (auto & armor : armors_msg->armors) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = armors_msg->header;
    ps.pose = armor.pose;
    try {
      armor.pose = tf2_buffer_->transform(ps, target_frame_).pose;
    } catch (const tf2::ExtrapolationException & ex) {
      RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
      return;
    }
  }

  //用于删除容器中满足某个条件的元素  目的是过滤掉异常的装甲板
  armors_msg->armors.erase(
    std::remove_if(
      armors_msg->armors.begin(), armors_msg->armors.end(),
      [this](const auto_aim_interfaces::msg::Armor & armor) {
        return abs(armor.pose.position.z) > 1.2 ||
               Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() >
                 max_armor_distance_;
      }),
    armors_msg->armors.end());

  // 初始化message
  auto_aim_interfaces::msg::TrackerInfo info_msg;
  auto_aim_interfaces::msg::Target target_msg;
  rclcpp::Time time = armors_msg->header.stamp;
  target_msg.header.stamp = time;
  target_msg.header.frame_id = target_frame_;

  // 更新追踪器
  if (tracker_->tracker_state == Tracker::LOST) {
    tracker_->init(armors_msg);
    target_msg.tracking = false;  //如果丢失目标，调用 init 函数来初始化跟踪器
  } else {
    dt_ = (time - last_time_).seconds();  //计算当前时间与上一次更新时间的时间差，以便在后续的状态更新中使用
    tracker_->lost_thres = static_cast<int>(lost_time_thres_ / dt_);  //用于设定在临时丢失状态下的阈值
    tracker_->update(armors_msg);

    // 发布信息
    info_msg.position_diff = tracker_->info_position_diff;
    info_msg.yaw_diff = tracker_->info_yaw_diff;
    info_msg.position.x = tracker_->measurement(0);
    info_msg.position.y = tracker_->measurement(1);
    info_msg.position.z = tracker_->measurement(2);
    info_msg.yaw = tracker_->measurement(3);
    info_pub_->publish(info_msg);

    if (tracker_->tracker_state == Tracker::DETECTING) {
      target_msg.tracking = false;  //表示没有追踪的目标
    } else if (
      tracker_->tracker_state == Tracker::TRACKING ||
      tracker_->tracker_state == Tracker::TEMP_LOST) {
      target_msg.tracking = true;
      // 储存target目标信息
      const auto & state = tracker_->target_state;
      target_msg.id = tracker_->tracked_id;
      target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
      target_msg.position.x = state(0);
      target_msg.velocity.x = state(1);
      target_msg.position.y = state(2);
      target_msg.velocity.y = state(3);
      target_msg.position.z = state(4);
      target_msg.velocity.z = state(5);
      target_msg.yaw = state(6);
      target_msg.v_yaw = state(7);
      target_msg.radius_1 = state(8);
      target_msg.radius_2 = tracker_->another_r;
      target_msg.dz = tracker_->dz;
    }
  }

  last_time_ = time;

  target_pub_->publish(target_msg);

  publishMarkers(target_msg);
}

rclcpp::Rate rate(10);


//用于发布可视化标记的函数
void ArmorTrackerNode::publishMarkers(const auto_aim_interfaces::msg::Target & target_msg)
{
  position_marker_.header = target_msg.header;//目标位置
  linear_v_marker_.header = target_msg.header;//线速度
  angular_v_marker_.header = target_msg.header;//角速度
  armor_marker_.header = target_msg.header;  //意味着这四个标记将与 target_msg 具有相同的时间戳和坐标系

  visualization_msgs::msg::MarkerArray marker_array;
  if (target_msg.tracking) {
    double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
    double xc = target_msg.position.x, yc = target_msg.position.y, za = target_msg.position.z;
    double vx = target_msg.velocity.x, vy = target_msg.velocity.y, vz = target_msg.velocity.z;
    double dz = target_msg.dz;

    position_marker_.action = visualization_msgs::msg::Marker::ADD;
    position_marker_.pose.position.x = xc;
    position_marker_.pose.position.y = yc;
    position_marker_.pose.position.z = za + dz / 2;

    linear_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    linear_v_marker_.points.clear();
    linear_v_marker_.points.emplace_back(position_marker_.pose.position);
    geometry_msgs::msg::Point arrow_end = position_marker_.pose.position;
    arrow_end.x += vx;
    arrow_end.y += vy;
    arrow_end.z += vz;
    linear_v_marker_.points.emplace_back(arrow_end);

    angular_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    angular_v_marker_.points.clear();
    angular_v_marker_.points.emplace_back(position_marker_.pose.position);
    arrow_end = position_marker_.pose.position;
    arrow_end.z += target_msg.v_yaw / M_PI;
    angular_v_marker_.points.emplace_back(arrow_end);

    armor_marker_.action = visualization_msgs::msg::Marker::ADD;
    armor_marker_.scale.y = tracker_->tracked_armor.type == "small" ? 0.135 : 0.23;
    bool is_current_pair = true;
    size_t a_n = target_msg.armors_num;
    geometry_msgs::msg::Point p_a;
    double r = 0;
    for (size_t i = 0; i < a_n; i++) {
      double tmp_yaw = yaw + i * (2 * M_PI / a_n);
      // Only 4 armors has 2 radius and height
      if (a_n == 4) {
        r = is_current_pair ? r1 : r2;
        p_a.z = za + (is_current_pair ? 0 : dz);
        is_current_pair = !is_current_pair;
      } else {
        r = r1;
        p_a.z = za;
      }
      p_a.x = xc - r * cos(tmp_yaw);
      p_a.y = yc - r * sin(tmp_yaw);

      armor_marker_.id = i;
      armor_marker_.pose.position = p_a;
      tf2::Quaternion q;
      q.setRPY(0, target_msg.id == "outpost" ? -0.26 : 0.26, tmp_yaw);
      armor_marker_.pose.orientation = tf2::toMsg(q);
      marker_array.markers.emplace_back(armor_marker_);
    }
  } else {
    position_marker_.action = visualization_msgs::msg::Marker::DELETE;
    linear_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
    angular_v_marker_.action = visualization_msgs::msg::Marker::DELETE;

    armor_marker_.action = visualization_msgs::msg::Marker::DELETE;
    marker_array.markers.emplace_back(armor_marker_);
  }

  marker_array.markers.emplace_back(position_marker_);
  marker_array.markers.emplace_back(linear_v_marker_);
  marker_array.markers.emplace_back(angular_v_marker_);
  marker_pub_->publish(marker_array);
  rate.sleep();
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)
