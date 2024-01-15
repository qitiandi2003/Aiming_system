# rm_auto_aim

#### ubuntu版本 22 
#### ros版本    humble


## License



创建 ROS 工作空间后 clone 项目，使用 rosdep 安装依赖后编译代码

	cd ros_ws/src
	git clone https://github.com/chenjunnn/rm_auto_aim.git
	cd ..
	rosdep install --from-paths src --ignore-src -r -y
	colcon build --symlink-install --packages-up-to auto_aim_bringup

## Usage



```sh
sudo apt-get install ros-humble-camera-info-manager
sudo apt-get install ros-humble-cv-bridge
更改cv_bridge .hpp为.h
so 绝对路径
