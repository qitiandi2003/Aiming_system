#ifndef __SOLVETRAJECTORY_H__
#define __SOLVETRAJECTORY_H__

#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
// #include </home/wyt/Downloads/ros2_humble/src/transport_drivers-humble/serial_driver/include/serial_driver/serial_driver.hpp>
#include <std_msgs/msg/float64.hpp>
// #include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>

// C++ system
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/aim.hpp"
#include<string> 
#ifndef PI
#define PI 3.1415926535f
#endif
#define GRAVITY 9.78
// typedef unsigned char uint8_t;


 using namespace std;
 
namespace robot_tf
{
    class  RoboTtfNode : public rclcpp::Node
    {
    public :
    explicit RoboTtfNode(const rclcpp::NodeOptions &options);

   // 低通滤波器参数?
   static constexpr float alpha_pitch = 0.1; // 调整此值以控制滤波强度
   
   void targetCB(auto_aim_interfaces::msg::Target::SharedPtr msg);
   //单方向空气阻力模型
   float monoDirectionalAirResistanceModel(float s, float v, float angle);
   //完全空气阻力模型
   float completeAirResistanceModel(float s, float v, float angle);
   //pitch弹道补偿
   float pitchTrajectoryCompensation(float s, float y, float v);
   //根据最优决策得出被击打装甲板 自动解算弹道
   float autoSolveTrajectory(float *pitch, float *yaw, float *aim_x, float *aim_y, float *aim_z );

    // ~RobotTf() override;

    private:

      rclcpp::Publisher<auto_aim_interfaces::msg::Aim>::SharedPtr target_pub_;
    rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr target_sub_;

        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

          visualization_msgs::msg::Marker shoot_point_;
    float t = 0.18f; // 飞行时间

    float aim_x = 0, aim_y = 0, aim_z = 0; // aim point 落点，传回上位机用于可视化
    float pitch = 0; //输出控制量 pitch绝对角度 弧度
    float yaw = 0;   //输出控制量 yaw绝对角度 弧度
    float pitch_filtered = 0.0; // 用于存储滤波后的 pitch

    // tar_pos tar_position[4]; //最多只有四块装甲板
    enum ARMOR_ID
{
    ARMOR_OUTPOST = 0,
    ARMOR_HERO = 1,
    ARMOR_ENGINEER = 2,
    ARMOR_INFANTRY3 = 3,
    ARMOR_INFANTRY4 = 4,
    ARMOR_INFANTRY5 = 5,
    ARMOR_GUARD = 6,
    ARMOR_BASE = 7
};

enum ARMOR_NUM
{
    ARMOR_NUM_BALANCE = 2,
    ARMOR_NUM_OUTPOST = 3,
    ARMOR_NUM_NORMAL = 4
};

enum BULLET_TYPE
{
    BULLET_17 = 0,
    BULLET_42 = 1
};

    float k{0.012};             //弹道系数

    //自身参数
    enum BULLET_TYPE bullet_type{BULLET_17};  //自身机器人类型 0-步兵 1-英雄
    float current_v{28};      //当前弹速
    float current_pitch{0};  //当前pitch
    float current_yaw{0};    //当前yaw

    //目标参数
    string id;
    float x{1.66568};             //ROS坐标系下的x
    float y{0.0159};             //ROS坐标系下的y
    float z{-0.2898};             //ROS坐标系下的z
    float vx{0};            //ROS坐标系下的vx
    float vy{0};            //ROS坐标系下的vy
    float vz{0};            //ROS坐标系下的vz
    float tar_yaw{0};        //目标yaw
    float v_yaw{0};          //目标yaw速度
    float r1{0.5};             //目标中心到前后装甲板的距离
    float r2{0.5};             //目标中心到左右装甲板的距离
    float dz{0.1};             //另一对装甲板的相对于被跟踪装甲板的高度差


    int bias_time{140};        //偏置时间
    
    
    float s_bias{0.074};         //枪口前推的距离
    float z_bias{0.058};        //yaw轴电机到枪口水平面的垂直距离
    enum ARMOR_ID armor_id{ARMOR_INFANTRY3};     //装甲板类型  0-outpost 6-guard 7-base
                                //1-英雄 2-工程 3-4-5-步兵 
    enum ARMOR_NUM armor_num{ARMOR_NUM_NORMAL};   //装甲板数字  2-balance 3-outpost 4-normal
    uint8_t armors_num;
    bool enemy{false};
    bool auto_shoot{false};
    float distance{0};
    float yaw_center{0};
//用于存储目标装甲板的信息
struct tar_pos
{
    float x;           //装甲板在世界坐标系下的x
    float y;           //装甲板在世界坐标系下的y
    float z;           //装甲板在世界坐标系下的z
    float yaw;         //装甲板坐标系相对于世界坐标系的yaw角
} tar_position[4];
    };
}




#endif /*__SOLVETRAJECTORY_H__*/