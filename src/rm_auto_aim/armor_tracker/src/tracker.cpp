#include "armor_tracker/tracker.hpp"

#include <angles/angles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

#include <rclcpp/logger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <cfloat>
#include <memory>
#include <string>


namespace rm_auto_aim
{
Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
    : tracker_state(LOST), 
    tracked_id(""),
    measurement(Eigen::VectorXd::Zero(4)),//将成员变量measurement初始化为一个四维的、所有元素为0的eigen向量   用于储存与测量相关信息
    target_state(Eigen::VectorXd::Zero(9)),//储存跟踪对象的状态信息
    max_match_distance_(max_match_distance),
    max_match_yaw_diff_(max_match_yaw_diff)
{

}


//目标追踪器的初始化函数，主要用于在收到装甲板信息后选择一个初始追踪的目标
void Tracker::init(const Armors::SharedPtr &armors_msg)
{
    if (armors_msg->armors.empty()) {
        return;
    }

    // 用于追踪距离图像中心最近的装甲板的距离  
    double min_distance = DBL_MAX;
    tracked_armor = armors_msg->armors[0];  //将tracker_armor设为armors_msg中的第一个装甲板作为初始选择
    for (const auto &armor : armors_msg->armors) {
        if (armor.distance_to_image_center < min_distance) {
            min_distance = armor.distance_to_image_center;
            tracked_armor = armor;
        }
    }

    initEKF(tracked_armor);
    //target_state << tracked_armor.pose.position.x, 0, tracked_armor.pose.position.y, 0, tracked_armor.pose.position.z, 0, orientationToYaw(tracked_armor.pose.orientation), 0, 0;


    RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF!");

    tracked_id = tracked_armor.number;
    tracker_state = DETECTING;

    updateArmorsNum(tracked_armor);  // 更新装甲板id
}


//更新装甲板信息
void Tracker::update(const Armors::SharedPtr &armors_msg)
{
    // KF predict
    Eigen::VectorXd ekf_prediction = ekf.predict();
    RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");

    // 初始化变量以跟踪最近的装甲?
    bool matched = false; // 初始化匹配值为false，用于标志是否找到了匹配的装甲板
    Armor closest_armor;   //!
    double min_distance = DBL_MAX;
    double min_yaw_diff = DBL_MAX;

    // 从状态中提取预测位置  !
    Eigen::Vector3d predicted_position = getArmorPositionFromState(ekf_prediction);

    // 迭代检测到的装甲板以找到具有相同 ID 的最接近的装甲板
    int same_id_armors_count = 0; // 修复：添加 same_id_armors_count 的初始化
    for (const auto &armor : armors_msg->armors)
    {
        // 仅考虑相同id的装甲板
        if (armor.number == tracked_id)
        {
            // 计算预测位置与当前装甲位置的差值
            auto armor_position = Eigen::Vector3d(armor.pose.position.x, armor.pose.position.y, armor.pose.position.z);
            double position_diff = (predicted_position - armor_position).norm();
            
            double yaw_diff = std::abs(orientationToYaw(armor.pose.orientation) - ekf_prediction(6));
            
            // 如果最接近的装甲板更匹配，则更新它   !
            if (position_diff < min_distance || (position_diff == min_distance && yaw_diff < min_yaw_diff))
            {
                min_distance = position_diff;
                min_yaw_diff = yaw_diff;
                closest_armor = armor;
                matched = true;
            }

            // 修复：在匹配的情况下增加相同 ID 装甲板的计数
            same_id_armors_count++;
        }
    }

    // 根据匹配结果更新目标状态
    if (matched)
    {
        RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Position Difference: %f, Yaw Difference: %f", min_distance, min_yaw_diff); //输出匹配的装甲板的相关信息
        tracked_armor = closest_armor;
        target_state = ekf_prediction;
        const auto &p = tracked_armor.pose.position; 

        // 更新EKF
        double measured_yaw = orientationToYaw(tracked_armor.pose.orientation);
        measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
        target_state = ekf.update(measurement);
        RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
        tracked_armor = closest_armor;
        // 存储跟踪器信息
        info_position_diff = min_distance;
        info_yaw_diff = min_yaw_diff;
    }
    else if (same_id_armors_count == 1 && min_yaw_diff > max_match_yaw_diff_)
    {
        // 未找到匹配的装甲板，但只有一件具有相同 id 的装甲板
        // 并且偏航已经跳跃，将这种情况视为目标正在旋转并且装甲跳跃
        handleArmorJump(closest_armor);
    }
    
    else
    {
        // 如果没有找到匹配的装甲板，则保持预测状态
        tracked_armor = Armor(); // 重置tracked_armor
        // target_state = ekf_prediction;  改为
        target_state = Eigen::VectorXd::Zero(9);
        // 没有找到匹配的装甲板
        RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "No matched armor found!");
    }

    // 防止半径扩散
    const double min_radius = 0.12;
    const double max_radius = 0.4;

    if (target_state(8) < min_radius) {
    target_state(8) = min_radius;
    ekf.setState(target_state);
  } else if (target_state(8) > max_radius) {
    target_state(8) = max_radius;
    ekf.setState(target_state);
  }

    // 检测模式
    if (tracker_state == DETECTING)
    {
        if (matched)
        {
            detect_count_++;
            if (detect_count_ > tracking_thres)
            {
                detect_count_ = 0;
                tracker_state = TRACKING;
            }
        }
        else
        {
            detect_count_ = 0;
            tracker_state = LOST;
        }
    }
    //追踪模式
    else if (tracker_state == TRACKING)
    {
        if (!matched)
        {
            tracker_state = TEMP_LOST;
            lost_count_++;
        }
    }
    //短暂丢失
    else if (tracker_state == TEMP_LOST)
    {
        if (!matched)
        {
            lost_count_++;
            if (lost_count_ > lost_thres)
            {
                lost_count_ = 0;
                tracker_state = LOST;
            }
        }
    //完全丢失
        else
        {
            tracker_state = TRACKING;
            lost_count_ = 0;
        }
    }
}



//初始化EKF
void Tracker::initEKF(const Armor &armor) {
  // 提取装甲位置和方向信息
  double xa = armor.pose.position.x;
  double ya = armor.pose.position.y;
  double za = armor.pose.position.z;
  double yaw = orientationToYaw(armor.pose.orientation);

  // 初始化目标状态
  last_yaw_ = 0;  // 记录最后一个装甲的yaw角
  target_state = Eigen::VectorXd::Zero(9);

  // 设置初始位置在目标后面0.2m处
  double r = 0.26;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  dz = 0, another_r = r;

  // 用计算出的位置信息设置初始状态向量
  target_state << xc, 0, yc, 0, za, 0, yaw, 0, r;

  // 设置扩展卡尔曼滤波器的初始状态
  ekf.setState(target_state);
}

void Tracker::updateArmorsNum(const Armor & armor)
{
  if (armor.type == "large" && (tracked_id == "3" || tracked_id == "4" || tracked_id == "5")) {
    tracked_armors_num = ArmorsNum::BALANCE_2;
  } else if (tracked_id == "outpost") {
    tracked_armors_num = ArmorsNum::OUTPOST_3;
  } else {
    tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

//处理目标发生跳变的逻辑
void Tracker::handleArmorJump(const Armor & current_armor)
{
  double yaw = orientationToYaw(current_armor.pose.orientation);
  target_state(6) = yaw;
  updateArmorsNum(current_armor);
  
  //如果为普通步兵则更新目标z上的变化
  if (tracked_armors_num == ArmorsNum::NORMAL_4) {
    dz = target_state(4) - current_armor.pose.position.z;
    target_state(4) = current_armor.pose.position.z;
    std::swap(target_state(8), another_r);   
  }
  RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Armor jump!");

// 如果位置差大于 max_match_distance_,
// 将此情况视为 ekf 发散，重置状态
  auto p = current_armor.pose.position;
  Eigen::Vector3d current_p(p.x, p.y, p.z);
  Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);  //获取目标状态预测的位置
  if ((current_p - infer_p).norm() > max_match_distance_) {
    double r = target_state(8);
    target_state(0) = p.x + r * cos(yaw);  // xc
    target_state(1) = 0;                   // vxc
    target_state(2) = p.y + r * sin(yaw);  // yc
    target_state(3) = 0;                   // vyc
    target_state(4) = p.z;                 // za
    target_state(5) = 0;                   // vza
    RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State!");
  }

  ekf.setState(target_state);
}


//将位姿信息转化为欧拉角
double Tracker::orientationToYaw(const geometry_msgs::msg::Quaternion &q)
{
  tf2::Quaternion tf_q;
  // 将geometry_msgs::四元数转换为tf2::四元数
  tf2::fromMsg(q, tf_q);
  
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);

  // 使偏航角变化连续（-pi~pi 到 -inf~inf）
  yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;

  return yaw;
}


//计算预测位置的函数
Eigen::Vector3d Tracker::getArmorPositionFromState(const Eigen::VectorXd &x)
{
// 从状态向量中提取参数
  double xc = x(0);
  double yc = x(2);
  double za = x(4);
  double yaw = x(6);
  double r = x(8);

// 计算当前装甲的预测位置  
  double xa = xc - r * std::cos(yaw);
  double ya = yc - r * std::sin(yaw);

  return Eigen::Vector3d(xa, ya, za);
}


}

