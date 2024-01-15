/*
@brief: 弹道解算 适配陈君的rm_vision
@author: CodeAlan  华南师大Vanguard战队
*/
// 近点只考虑水平方向的空气阻力



//TODO 完整弹道模型
//TODO 适配英雄机器人弹道解算


#include <math.h>
#include <stdio.h>

#include "/home/zhanqi/ros3_ws/src/robot_tf/include/SolveTrajectory.hpp"



namespace robot_tf
{
    RoboTtfNode::RoboTtfNode(const rclcpp::NodeOptions & options)
    : Node("robot_tf", options )
    {
        //   RCLCPP_INFO(get_logger(), "Start RMSerialDriver!");
          target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Aim>(
    "/target", rclcpp::SensorDataQoS());
        
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/shoot_point", 10);
        target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS(),
        std::bind(&RoboTtfNode::targetCB, this, std::placeholders::_1));

    shoot_point_.header.frame_id = "odom";
  shoot_point_.ns = "aiming_point";
  shoot_point_.type = visualization_msgs::msg::Marker::SPHERE;
  shoot_point_.action = visualization_msgs::msg::Marker::ADD;
  shoot_point_.scale.x = shoot_point_.scale.y = shoot_point_.scale.z = 0.12;
  shoot_point_.color.r = 1.0;
  shoot_point_.color.g = 1.0;
  shoot_point_.color.b = 1.0;
  shoot_point_.color.a = 1.0;
  shoot_point_.lifetime = rclcpp::Duration::from_seconds(0.1);
    }
    
void RoboTtfNode::targetCB(const auto_aim_interfaces::msg::Target::SharedPtr msg)
{


    const static std::map<std::string, uint8_t> id_unit8_map{
    {"", 0},  {"outpost", 0}, {"1", 1}, {"1", 1},     {"2", 2},
    {"3", 3}, {"4", 4},       {"5", 5}, {"guard", 6}, {"base", 7}};

    x = msg->position.x;             //ROS坐标系下的x
    y = msg->position.y;             //ROS坐标系下的y
    z = msg->position.z;             //ROS坐标系下的z
    vx = msg->velocity.x;            //ROS坐标系下的vx
    vy = msg->velocity.y;            //ROS坐标系下的vy
    vz = msg->velocity.z;            //ROS坐标系下的vz
    tar_yaw = msg->yaw;        //目标yaw
    v_yaw = msg->v_yaw;          //目标yaw速度
    r1 = msg->radius_1;             //目标中心到前后装甲板的距离
    r2 = msg->radius_2;             //目标中心到左右装甲板的距离
    dz = msg->dz;             //另一对装甲板的相对于被跟踪装甲板的高度差
    armors_num = msg->armors_num; //
    id = msg->id;
    if(armors_num != 0){
    //   enemy = true;
      if(armors_num == 2) {   armor_num = ARMOR_NUM_BALANCE; } 
      else if(armors_num == 3){ armor_num = ARMOR_NUM_OUTPOST; }
      else { armor_num = ARMOR_NUM_NORMAL; }
    // armor_id = msg->id; //
    
       distance = autoSolveTrajectory(&pitch, &yaw, &aim_x, &aim_y, &aim_z );
       
       // 应用低通滤波器?
       pitch_filtered = alpha_pitch * pitch + (1.0 - alpha_pitch) * pitch_filtered;

       auto_aim_interfaces::msg::Aim aim_msg;
       aim_msg.pitch_angle = pitch_filtered;
       aim_msg.yaw_angle = yaw;
       aim_msg.yaw_center = yaw_center;
    //    if(msg->yaw * 0.13 < 0.07)  aim_msg.auto_shoot = true;
    //    if(msg->v_yaw > 10) aim_msg.auto_shoot = true;
    //    aim_msg.auto_shoot = false;
    if(id == "1" || armors_num == 2)
    aim_msg.type = 1;//大装甲
    else
    aim_msg.type = 0;//小装甲
       aim_msg.enemy = true;
       aim_msg.auto_shoot = auto_shoot;
       aim_msg.distance = distance;
       aim_msg.v_x = msg->velocity.x;
       aim_msg.v_yaw = v_yaw;
       aim_msg.robot_id = std::stoi(id);
       target_pub_->publish(aim_msg);
       aim_msg.enemy = false;

            shoot_point_.header.stamp = this->now();
            shoot_point_.pose.position.x = aim_x;
            shoot_point_.pose.position.y = aim_y;
            shoot_point_.pose.position.z = aim_z;
            marker_pub_->publish(shoot_point_);
    //      printf("main pitch:%f° yaw:%f° ", pitch * 180 / PI, yaw * 180 / PI);
    // printf("\npitch:%frad yaw:%frad aim_x:%f aim_y:%f aim_z:%f", pitch, yaw, aim_x, aim_y, aim_z);
    }
    else 
    {
        auto_aim_interfaces::msg::Aim aim_msg;
        aim_msg.enemy = false;
        // aim_msg.distance = 99;
        // aim_msg.yaw_angle = 99;
        target_pub_->publish(aim_msg);
        visualization_msgs::msg::Marker shoot_point_;
        marker_pub_->publish(shoot_point_);
    }
}
    /*
@brief 单方向空气阻力弹道模型
@param s:m 距离
@param v:m/s 速度
@param angle:rad 角度
@return z:m
*/
float RoboTtfNode::monoDirectionalAirResistanceModel(float s, float v, float angle)
{
    float z;
    //t为给定v与angle时的飞行时间
    t = (float)((exp(k * s) - 1) / (k * v * cos(angle)));
    //z为给定v与angle时的高度
    z = (float)(v * sin(angle) * t - GRAVITY * t * t / 2);
    // printf("model %f %f\n", t, z);
    return z;
}


/*
@brief 完整弹道模型
@param s:m 距离
@param v:m/s 速度
@param angle:rad 角度
@return z:m
*/
//TODO 完整弹道模型
float RoboTtfNode::completeAirResistanceModel(float s, float v, float angle)
{
    // continue;
    return 0 ;

}



/*
@brief pitch轴解算
@param s:m 距离
@param z:m 高度
@param v:m/s
@return angle_pitch:rad
*/
float RoboTtfNode::pitchTrajectoryCompensation(float s, float z, float v)
{
    float z_temp, z_actual, dz;
    float angle_pitch;
    int i = 0;
    z_temp = z;
    // iteration
    for (i = 0; i < 20; i++)
    {
        angle_pitch = atan2(z_temp, s); // rad
        z_actual = monoDirectionalAirResistanceModel(s, v, angle_pitch);
        dz = 0.3*(z - z_actual);
        z_temp = z_temp + dz;
        // printf("iteration num %d: angle_pitch %f, temp target z:%f, err of z:%f, s:%f\n",
        //     i + 1, angle_pitch * 180 / PI, z_temp, dz,s);
        if (fabsf(dz) < 0.00001)
        {
            break;
        }
    }
    return angle_pitch;
}

/*
@brief 根据最优决策得出被击打装甲板 自动解算弹道
@param pitch:rad  传出pitch
@param yaw:rad    传出yaw
@param aim_x:传出aim_x  打击目标的x
@param aim_y:传出aim_y  打击目标的y
@param aim_z:传出aim_z  打击目标的z
*/
float RoboTtfNode::autoSolveTrajectory(float *pitch, float *yaw, float *aim_x, float *aim_y, float *aim_z)
{


    float dis = sqrt((*aim_x) * (*aim_x) + (*aim_y) * (*aim_y));
    t = dis / current_v ;//0.55 +
    // 线性预测
    float timeDelay = bias_time/1000.0 + t;
    tar_yaw += v_yaw * timeDelay;

    //计算四块装甲板的位置
	int use_1 = 1;
	int i = 0;
    int idx = 0; // 选择的装甲板
    //armor_type = 1 为平衡步兵
    if (armor_num == ARMOR_NUM_BALANCE) {
        for (i = 0; i<2; i++) {
            float tmp_yaw = tar_yaw + i * PI;
            float r = r1;
            tar_position[i].x = x - r*cos(tmp_yaw);
            tar_position[i].y = y - r*sin(tmp_yaw);
            tar_position[i].z = z;
            tar_position[i].yaw = tar_yaw + i * PI;
        }

        float yaw_diff_min = fabsf(*yaw - tar_position[0].yaw);

        //因为是平衡步兵 只需判断两块装甲板即可
        float temp_yaw_diff = fabsf(*yaw - tar_position[1].yaw);
        if (temp_yaw_diff < yaw_diff_min)
        {
            yaw_diff_min = temp_yaw_diff;
            idx = 1;
        }


    } else {

        for (i = 0; i<4; i++) {
            float tmp_yaw = tar_yaw + i * PI/2.0;
            float r = use_1 ? r1 : r2;
            tar_position[i].x = x - r*cos(tmp_yaw);
            tar_position[i].y = y - r*sin(tmp_yaw);
            tar_position[i].z = use_1 ? z : z + dz;
            tar_position[i].yaw = tar_yaw + i * PI/2.0;
            use_1 = !use_1;
        }

            //2种常见决策方案：
            //1.计算枪管到目标装甲板yaw最小的那个装甲板
            //2.计算距离最近的装甲板


            if(v_yaw <=0.5 && v_yaw >= -0.5)
            {
                            //计算枪管到目标装甲板yaw最小的那个装甲板
        float yaw_diff_min = fabsf(*yaw - tar_position[0].yaw);
        for (i = 1; i<4; i++) {
            float temp_yaw_diff = fabsf(*yaw - tar_position[i].yaw);
            if (temp_yaw_diff < yaw_diff_min)
            {
                yaw_diff_min = temp_yaw_diff;
                idx = i;
            }
        }
            }
else
{  //计算距离最近的装甲板
            	float dis_diff_min = sqrt(tar_position[0].x * tar_position[0].x + tar_position[0].y * tar_position[0].y);
        	// int idx = 0;
        	for (i = 1; i<4; i++)
        	{
        		float temp_dis_diff = sqrt(tar_position[i].x * tar_position[0].x + tar_position[i].y * tar_position[0].y);
        		if (temp_dis_diff < dis_diff_min)
        		{
        			dis_diff_min = temp_dis_diff;
        			idx = i;
        		}
        	}
}
        



    }

	

    *aim_z = (tar_position[idx].z + tar_position[idx % 4 + 1].z ) / 2 + vz * timeDelay;
    *aim_x = tar_position[idx].x + vx * timeDelay;
    *aim_y = tar_position[idx].y + vy * timeDelay;

    *pitch = -pitchTrajectoryCompensation(sqrt((*aim_x) * (*aim_x) + (*aim_y) * (*aim_y)) - s_bias,
            *aim_z + z_bias, current_v);

    yaw_center = atan (y / x);
    *yaw = (float)(atan2(*aim_y, *aim_x));
  
    return sqrt(tar_position[idx].x*tar_position[idx].x + tar_position[idx].y*tar_position[idx].y ) ;
    // return sqrt((*aim_x) * (*aim_x) + (*aim_y) * (*aim_y)) - s_bias;

}
}

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(robot_tf::RoboTtfNode)


// 从坐标轴正向看向原点，逆时针方向为正

// int main()
// {
//     float aim_x = 0, aim_y = 0, aim_z = 0; // aim point 落点，传回上位机用于可视化
//     float pitch = 0; //输出控制量 pitch绝对角度 弧度
//     float yaw = 0;   //输出控制量 yaw绝对角度 弧度

//     //定义参数
//     st.k = 0.092;
//     st.bullet_type =  BULLET_17;
//     st.current_v = 18;
//     st.current_pitch = 0;
//     st.current_yaw = 0;
//     st.xw = 1.66568;
//     st.yw = 0.0159;
//     st.zw = -0.2898;
//     st.vxw = 0;
//     st.vyw = 0;
//     st.vzw = 0;
//     st.v_yaw = 0;
//     st.tar_yaw = 0.09131;
//     st.r1 = 0.5;
//     st.r2 = 0.5;
//     st.dz = 0.1;
//     st.bias_time = 100;
//     st.s_bias = 0.19133;
//     st.z_bias = 0.21265;
//     st.armor_id = ARMOR_INFANTRY3;
//     st.armor_num = ARMOR_NUM_NORMAL;


//     autoSolveTrajectory(&pitch, &yaw, &aim_x, &aim_y, &aim_z);


//     printf("main pitch:%f° yaw:%f° ", pitch * 180 / PI, yaw * 180 / PI);
//     printf("\npitch:%frad yaw:%frad aim_x:%f aim_y:%f aim_z:%f", pitch, yaw, aim_x, aim_y, aim_z);

//     return 0;
// }
