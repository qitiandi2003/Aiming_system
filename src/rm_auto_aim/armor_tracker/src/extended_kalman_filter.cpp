#include "armor_tracker/extended_kalman_filter.hpp"

namespace rm_auto_aim
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const VecVecFunc & f, const VecVecFunc & h, const VecMatFunc & j_f, const VecMatFunc & j_h,
  const VoidMatFunc & u_q, const VecMatFunc & u_r, const Eigen::MatrixXd & P0)
: 
//初始化函数
  f(f),
  h(h),
  jacobian_f(j_f),
  jacobian_h(j_h),
  update_Q(u_q),
  update_R(u_r),

//将初始状态协方差矩阵赋值给成员变量 P_post。
  P_post(P0),

//获取初始状态协方差矩阵的行数，赋值给成员变量 n。
  n(P0.rows()),

//创建一个单位矩阵，行数和列数为 n，然后赋值给成员变量 I。
  I(Eigen::MatrixXd::Identity(n, n)),

//分别初始化预测状态和后验状态的向量，长度为 n。
  x_pri(n),
  x_post(n)
{
}

//设置滤波器的初始状态   作用是将传入的状态向量 x0 设置为扩展卡尔曼滤波器的当前状态 x_post。
void ExtendedKalmanFilter::setState(const Eigen::VectorXd & x0) { x_post = x0; }


Eigen::MatrixXd ExtendedKalmanFilter::predict()
{

  //计算状态转移函数的雅可比矩阵 F，该矩阵在卡尔曼滤波中用于预测状态。
  F = jacobian_f(x_post),
  //调用函数 update_Q() 来获取过程噪声协方差矩阵 Q，用于考虑系统中的不确定性和噪声
  Q = update_Q();

  //使用状态转移函数 f 预测当前状态 x_post，结果存储在 x_pri 中。
  x_pri = f(x_post);

  //利用状态转移函数的雅可比矩阵和卡尔曼滤波的协方差更新方程，预测当前状态的协方差矩阵。
  P_pri = F * P_post * F.transpose() + Q;

  //在某些情况下，可能没有可用的测量值，因此将预测值作为后验值，更新 x_post 和 P_post。
  x_post = x_pri;
  P_post = P_pri;

  return x_pri;
}


Eigen::MatrixXd ExtendedKalmanFilter::update(const Eigen::VectorXd & z)
{

  //计算测量函数的雅可比矩阵 H，该矩阵在卡尔曼滤波中用于估计测量与状态之间的关系。
  H = jacobian_h(x_pri), 
  //调用函数 update_R(z) 来获取更新后的测量噪声协方差矩阵 R，用于考虑测量中的不确定性和噪声。
  R = update_R(z);
  
  //计算卡尔曼增益 K，该增益用于权衡预测值和测量值，校正状态估计。
  K = P_pri * H.transpose() * (H * P_pri * H.transpose() + R).inverse();

  //使用卡尔曼增益和测量残差（测量值与预测值之间的差异）更新后验状态 x_post。
  x_post = x_pri + K * (z - h(x_pri));

  //使用卡尔曼增益更新后验协方差矩阵 P_post。
  P_post = (I - K * H) * P_pri;

  return x_post;
}

}  // namespace rm_auto_aim
