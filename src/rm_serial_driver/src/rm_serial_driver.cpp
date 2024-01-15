// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#include <tf2/LinearMath/Quaternion.h>

#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/utilities.hpp>
#include <serial_driver/serial_driver.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// C++ system
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rm_serial_driver/crc.hpp"
#include "rm_serial_driver/packet.hpp"
#include "rm_serial_driver/rm_serial_driver.hpp"

namespace rm_serial_driver
{
RMSerialDriver::RMSerialDriver(const rclcpp::NodeOptions & options)
: Node("rm_serial_driver", options),
  owned_ctx_{new IoContext(2)},
  serial_driver_{new drivers::serial_driver::SerialDriver(*owned_ctx_)}
{
  RCLCPP_INFO(get_logger(), "Start RMSerialDriver!");

  getParams();

  // TF broadcaster
  timestamp_offset_ = this->declare_parameter("timestamp_offset", 0.0);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // Create Publisher
  latency_pub_ = this->create_publisher<std_msgs::msg::Float64>("/latency", 10);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/aiming_point", 10);
  debug_target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Aim>(
    "/debug_target", rclcpp::SensorDataQoS());
  // Detect parameter client
  detector_param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, "armor_detector");

  // Tracker reset service client
  reset_tracker_client_ = this->create_client<std_srvs::srv::Trigger>("/tracker/reset");

  try {
    serial_driver_->init_port(device_name_, *device_config_);
    if (!serial_driver_->port()->is_open()) {
      serial_driver_->port()->open();
      receive_thread_ = std::thread(&RMSerialDriver::receiveData, this);
    }
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(
      get_logger(), "Error creating serial port: %s - %s", device_name_.c_str(), ex.what());
    throw ex;
  }

  aiming_point_.header.frame_id = "odom";
  aiming_point_.ns = "aiming_point";
  aiming_point_.type = visualization_msgs::msg::Marker::SPHERE;
  aiming_point_.action = visualization_msgs::msg::Marker::ADD;
  aiming_point_.scale.x = aiming_point_.scale.y = aiming_point_.scale.z = 0.12;
  aiming_point_.color.r = 1.0;
  aiming_point_.color.g = 1.0;
  aiming_point_.color.b = 1.0;
  aiming_point_.color.a = 1.0;
  aiming_point_.lifetime = rclcpp::Duration::from_seconds(0.1);

  // Create Subscription
  target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Aim>(
    "/target", rclcpp::SensorDataQoS(),
    std::bind(&RMSerialDriver::sendData, this, std::placeholders::_1));
  armor_sub_ = this->create_subscription<auto_aim_interfaces::msg::Armors>(
    "/tracker/target", rclcpp::SensorDataQoS(),
    std::bind(&RMSerialDriver::armorCB, this, std::placeholders::_1));
    }

RMSerialDriver::~RMSerialDriver()
{
  if (receive_thread_.joinable()) {
    receive_thread_.join();
  }

  if (serial_driver_->port()->is_open()) {
    serial_driver_->port()->close();
  }

  if (owned_ctx_) {
    owned_ctx_->waitForExit();
  }
}

void RMSerialDriver::receiveData()
{
  std::vector<uint8_t> header(1);
  std::vector<uint8_t> data;
  data.reserve(sizeof(ReceivePacket));

  while (rclcpp::ok()) {
    try {
      serial_driver_->port()->receive(header);

      if (header[0] == 0x5A) {
        data.resize(sizeof(ReceivePacket) - 1);
        serial_driver_->port()->receive(data);


        
        data.insert(data.begin(), header[0]);
        ReceivePacket packet = fromVector(data);

        //by duang
        // for(long unsigned int i = 0; i< sizeof(ReceivePacket); i++)
        // {
        //   RCLCPP_INFO(get_logger(), "serial data is %d", data[i]);
        // }
        // RCLCPP_INFO(get_logger(), "over!");

        // bool crc_ok =
        // crc16::Verify_CRC16_Check_Sum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (1) {
          if (!initial_set_param_ || packet.detect_color != previous_receive_color_) {
            setParam(rclcpp::Parameter("detect_color", packet.detect_color)); // packet.detect_color
            previous_receive_color_ = packet.detect_color;
          }
          // packet.reset_tracker = false;
          // if (packet.reset_tracker) {
          //   resetTracker();
          // }

          geometry_msgs::msg::TransformStamped t;
          timestamp_offset_ = this->get_parameter("timestamp_offset").as_double();
          t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
          t.header.frame_id = "odom";
          t.child_frame_id = "gimbal_link";
          pitch_ = packet.pitch;
          yaw_ = packet.yaw;
          tf2::Quaternion q;
          q.setRPY(-packet.roll/180*3.1415, (packet.pitch)/180*3.1415, (packet.yaw)/180*3.1415);
          t.transform.rotation = tf2::toMsg(q);
          tf_broadcaster_->sendTransform(t);
          
          //by duang(successful shit)
          RCLCPP_INFO(get_logger(), "pitch:%f , yaw:%f, detect_color%02X", pitch_, yaw_, packet.detect_color);


          // if (abs(packet.aim_x) > 0.01) {
          //   aiming_point_.header.stamp = this->now();
          //   aiming_point_.pose.position.x = packet.aim_x;
          //   aiming_point_.pose.position.y = packet.aim_y;
          //   aiming_point_.pose.position.z = packet.aim_z;
          //   marker_pub_->publish(aiming_point_);
          // }
        } else {
          RCLCPP_ERROR(get_logger(), "CRC error!");
        }
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 20, "Invalid header: %02X", header[0]);
      }
    } catch (const std::exception & ex) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 20, "Error while receiving data: %s", ex.what());
      reopenPort();
    }
  }
}

void RMSerialDriver::armorCB(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
  // for(int  i = 0; i <= 3; i++) {
  //     if(armors_msg->armors[i].pose.position.x < 0.07)
  //     {
  //       // armor_time = armors_msg->header;
  //       auto_shoot = true;
  //     }
  // }
  // armor_type = 
}

void RMSerialDriver::sendData(const auto_aim_interfaces::msg::Aim::SharedPtr msg)
{
  const static std::map<std::string, uint8_t> id_unit8_map{
    {"", 0},  {"outpost", 0}, {"1", 1}, {"1", 1},     {"2", 2},
    {"3", 3}, {"4", 4},       {"5", 5}, {"guard", 6}, {"base", 7}};

  try {
    SendPacket packet;
    // packet.tracking = msg->tracking;
    // packet.id = id_unit8_map.at(msg->id);
    // packet.armors_num = msg->armors_num;
    // packet.x = msg->position.x;
    // packet.y = msg->position.y;
    // packet.z = msg->position.z;
    // packet.yaw = msg->yaw;
    // packet.vx = msg->velocity.x;
    // packet.vy = msg->velocity.y;
    // packet.vz = msg->velocity.z;
    // packet.v_yaw = msg->v_yaw;
    // packet.r1 = msg->radius_1;
    // packet.r2 = msg->radius_2;
    // packet.dz = msg->dz;

    packet.enemy = msg->enemy;
    packet.pitch = msg->pitch_angle* 180 / 3.1415926535 - pitch_  ;

    if(abs(msg->v_yaw) > 5 || abs(msg->v_yaw) < 2 )
       {
         packet.yaw = msg->yaw_angle* 180 / 3.1415926535 - yaw_  ;
         packet.pitch -= 1;
       }
     else 
     {
     packet.yaw = msg->yaw_center * 180 / 3.1415926535 - yaw_ ;
     }
    packet.robot_id = msg->robot_id;
    if(msg->robot_id == 0 )
    {
      packet.yaw = msg->yaw_center * 180 / 3.1415926535 - yaw_ ;
    }
    //by myself
    // std::cout << msg->enemy;
    // std::cout << msg->pitch_angle;

      // if(msg->yaw * 0.13 < 0.07)  aim_msg.auto_shoot = true;
      // if(msg->v_yaw > 10) aim_msg.auto_shoot = true;
      // packet.auto_shoot = false;

        // packet.yaw = 23;
    // packet.auto_aim = msg->auto_shoot;

    // packet.auto_aim = false;
    // if(armor_dit < 0.07) 
    // packet.auto_aim = true;
    // else packet.auto_aim = false;

    // msg->header == armor_time &&
    // if( auto_shoot ) packet.auto_aim = true;
    // else packet.auto_aim = false;


    float auto_shoot_num = tan(( msg->yaw_angle * 180 / 3.1415926535 - yaw_ )*3.1415926/180 )  * msg->distance;
    packet.auto_aim =false;   
    // if(msg->v_x > 2 ) 
    // {
    //   fire_count--;
    //   if(fire_count < 0){
    //     fire_count = 5;
    //     packet.auto_aim = true;
    //     packet.yaw*=2.0;
    //   }
    // }

    // else 


     if( (auto_shoot_num  <= 0.17 && auto_shoot_num  >= -0.17 && msg->type)
         || (auto_shoot_num  <= 0.10 && auto_shoot_num  >= -0.10 && !msg->type) )
      {
        packet.auto_aim = true;
      }
    //  if( abs(packet.yaw) < 4 )
    //   {
    //     packet.auto_aim = true;
    //   }
    //   if(abs(msg->v_yaw) > 0.5)
    //   {
    //     packet.yaw *= 2.5;
    //   }

    // else packet.auto_aim = false;
      // packet.yaw += 1;
    // packet.auto_aim = auto_shoot;
    // if(!packet.enemy)  packet.auto_aim = 0;
    //crc16::Append_CRC16_Check_Sum(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));

    std::vector<uint8_t> data = toVector(packet);

    serial_driver_->port()->send(data);

    auto_aim_interfaces::msg::Aim aim_msg;
    aim_msg.enemy = packet.enemy;
    aim_msg.pitch_angle = packet.pitch;
    aim_msg.yaw_angle =  packet.yaw;
    aim_msg.auto_shoot = packet.auto_aim;
    aim_msg.distance = msg->yaw_angle* 180 / 3.1415926535;
    aim_msg.auto_shoot_num = yaw_;
    debug_target_pub_->publish(aim_msg);

    std_msgs::msg::Float64 latency;
    // latency.data = (this->now() - msg->header.stamp).seconds() * 1000.0;
    RCLCPP_DEBUG_STREAM(get_logger(), "Total latency: " + std::to_string(latency.data) + "ms");
    latency_pub_->publish(latency);
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(get_logger(), "Error while sending data: %s", ex.what());
    reopenPort();
  }
}

void RMSerialDriver::getParams()
{
  using FlowControl = drivers::serial_driver::FlowControl;
  using Parity = drivers::serial_driver::Parity;
  using StopBits = drivers::serial_driver::StopBits;

  uint32_t baud_rate{};
  auto fc = FlowControl::NONE;
  auto pt = Parity::NONE;
  auto sb = StopBits::ONE;

  try {
    device_name_ = declare_parameter<std::string>("device_name", "");
  } catch (rclcpp::ParameterTypeException & ex) {
    RCLCPP_ERROR(get_logger(), "The device name provided was invalid");
    throw ex;
  }

  try {
    baud_rate = declare_parameter<int>("baud_rate", 0);
  } catch (rclcpp::ParameterTypeException & ex) {
    RCLCPP_ERROR(get_logger(), "The baud_rate provided was invalid");
    throw ex;
  }

  try {
    const auto fc_string = declare_parameter<std::string>("flow_control", "");

    if (fc_string == "none") {
      fc = FlowControl::NONE;
    } else if (fc_string == "hardware") {
      fc = FlowControl::HARDWARE;
    } else if (fc_string == "software") {
      fc = FlowControl::SOFTWARE;
    } else {
      throw std::invalid_argument{
        "The flow_control parameter must be one of: none, software, or hardware."};
    }
  } catch (rclcpp::ParameterTypeException & ex) {
    RCLCPP_ERROR(get_logger(), "The flow_control provided was invalid");
    throw ex;
  }

  try {
    const auto pt_string = declare_parameter<std::string>("parity", "");

    if (pt_string == "none") {
      pt = Parity::NONE;
    } else if (pt_string == "odd") {
      pt = Parity::ODD;
    } else if (pt_string == "even") {
      pt = Parity::EVEN;
    } else {
      throw std::invalid_argument{"The parity parameter must be one of: none, odd, or even."};
    }
  } catch (rclcpp::ParameterTypeException & ex) {
    RCLCPP_ERROR(get_logger(), "The parity provided was invalid");
    throw ex;
  }

  try {
    const auto sb_string = declare_parameter<std::string>("stop_bits", "");

    if (sb_string == "1" || sb_string == "1.0") {
      sb = StopBits::ONE;
    } else if (sb_string == "1.5") {
      sb = StopBits::ONE_POINT_FIVE;
    } else if (sb_string == "2" || sb_string == "2.0") {
      sb = StopBits::TWO;
    } else {
      throw std::invalid_argument{"The stop_bits parameter must be one of: 1, 1.5, or 2."};
    }
  } catch (rclcpp::ParameterTypeException & ex) {
    RCLCPP_ERROR(get_logger(), "The stop_bits provided was invalid");
    throw ex;
  }

  device_config_ =
    std::make_unique<drivers::serial_driver::SerialPortConfig>(baud_rate, fc, pt, sb);
}

void RMSerialDriver::reopenPort()
{
  RCLCPP_WARN(get_logger(), "Attempting to reopen port");
  try {
    if (serial_driver_->port()->is_open()) {
      serial_driver_->port()->close();
    }
    serial_driver_->port()->open();
    RCLCPP_INFO(get_logger(), "Successfully reopened port");
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(get_logger(), "Error while reopening port: %s", ex.what());
    if (rclcpp::ok()) {
      rclcpp::sleep_for(std::chrono::seconds(1));
      reopenPort();
    }
  }
}

void RMSerialDriver::setParam(const rclcpp::Parameter & param)
{
  if (!detector_param_client_->service_is_ready()) {
    RCLCPP_WARN(get_logger(), "Service not ready, skipping parameter set");
    return;
  }

  if (
    !set_param_future_.valid() ||
    set_param_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    RCLCPP_INFO(get_logger(), "Setting detect_color to %ld...", param.as_int());
    set_param_future_ = detector_param_client_->set_parameters(
      {param}, [this, param](const ResultFuturePtr & results) {
        for (const auto & result : results.get()) {
          if (!result.successful) {
            RCLCPP_ERROR(get_logger(), "Failed to set parameter: %s", result.reason.c_str());
            return;
          }
        }
        RCLCPP_INFO(get_logger(), "Successfully set detect_color to %ld!", param.as_int());
        initial_set_param_ = true;
      });
  }
}

void RMSerialDriver::resetTracker()
{
  if (!reset_tracker_client_->service_is_ready()) {
    RCLCPP_WARN(get_logger(), "Service not ready, skipping tracker reset");
    return;
  }

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  reset_tracker_client_->async_send_request(request);
  RCLCPP_INFO(get_logger(), "Reset tracker!");
}

}  // namespace rm_serial_driver

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_serial_driver::RMSerialDriver)
