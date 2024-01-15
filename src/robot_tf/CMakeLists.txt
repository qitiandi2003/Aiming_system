cmake_minimum_required(VERSION 3.10)
project(robot_tf)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE DEBUG)
## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(
#  fcl_INCLUDE_DIRS
#  "/home/csy/ros_ws/src/robot_tf/include"
# ) 


#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
ament_auto_find_build_dependencies()

# INCLUDE_DIRECTORIES(
#   ${/home/csy/ros_ws/src/robot_tf}/robot_tf/include/
# )


###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  # DIRECTORY src
  include
  src/SolveTrajectory.cpp
  #DIRECTORY src
  )

  #add_executable (robot_tf src/SolveTrajectory.cpp)
#target_link_libraries (robot_tf $ {catkin_LIBRARIES})


target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN robot_tf::RoboTtfNode
  EXECUTABLE robot_tf_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest)


endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  # model
)
