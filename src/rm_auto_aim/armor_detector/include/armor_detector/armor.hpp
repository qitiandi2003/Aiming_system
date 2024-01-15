// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim {

// 常量定义
const int RED = 0;  // 红色标识
const int BLUE = 1;  // 蓝色标识

// 装甲类型枚举
enum class ArmorType { SMALL, LARGE, INVALID };  // 小型、大型和无效装甲类型
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};  // 装甲类型字符串表示

// 灯条结构表示装甲板上的灯条
struct Lamp : public cv::RotatedRect {
  Lamp() = default;
  explicit Lamp(cv::RotatedRect box) : cv::RotatedRect(box) {
    // 计算灯条的额外属性
    cv::Point2f p[4];
    box.points(p);
    std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
    top = (p[0] + p[1]) / 2;
    bottom = (p[2] + p[3]) / 2;

    length = cv::norm(top - bottom);
    width = cv::norm(p[0] - p[1]);

    tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    tilt_angle = tilt_angle / CV_PI * 180;
  }

  int color;             // 灯条颜色，可以是红色或蓝色
  cv::Point2f top, bottom; // 灯条的顶部和底部点
  double length;          // 灯条的长度
  double width;           // 灯条的宽度
  float tilt_angle;       // 灯条的倾斜角
};

// 装甲结构表示检测到的装甲
struct Armor {
  Armor() = default;
  Armor(const Lamp & l1, const Lamp & l2) {
    // 根据 x 坐标排列灯条
    if (l1.center.x < l2.center.x) {
      left_lamp = l1, right_lamp = l2;
    } else {
      left_lamp = l2, right_lamp = l1;
    }
    center = (left_lamp.center + right_lamp.center) / 2;
  }

  // 灯条对部分
  Lamp left_lamp, right_lamp; // 左灯条和右灯条
  cv::Point2f center;            // 检测到的装甲的中心
  ArmorType type;                // 装甲的类型

  // 数字部分
  cv::Mat number_img;            // 检测到数字的图像
  std::string number;            // 检测到数字的字符串表示
  float confidence;               // 检测的置信度
  std::string classification_result; // 分类结果
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_

