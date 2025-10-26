#  SmartCar 巡检机器人

一款基于 ROS 的自动巡检机器人，旨在实现工业环境中的自主巡检任务。

---

##  项目简介

SmartCar 是一个开源项目，利用 ROS（Robot Operating System）框架，结合 Python、C++ 和 Shell 脚本，实现自动巡检和自主导航功能。项目支持机器人在复杂环境下进行避障、路径规划及数据采集。

---


##  功能特性

🤖 机器人仿真: 基于Gazebo的完整机器人仿真环境

🗺️ 自主导航: 使用Navigation2进行路径规划和避障

🔄 自动巡逻: 可配置的多点循环巡逻功能

📷 图像记录: 在巡逻点自动拍照记录

🗣️ 语音播报: 到达目标点时的语音提示

⚙️ 可配置化: 通过参数文件灵活配置巡逻路径

## 安装和编译

### 1.克隆项目
```bash
git clone https://github.com/lijing0416/smartcar.git
cd smartcar
```
### 2.编译项目
```bash
colcon build
source install/setup.bash
```

## 使用说明

### 1.启动仿真环境
```bash
# 终端1: 启动Gazebo仿真
ros2 launch fishbot_description gazebo_sim.launch.py

# 终端2: 启动导航系统
ros2 launch fishbot_navigation2 navigation2.launch.py

```
### 2.配置巡逻参数
编辑`src/autopatrol_robot/autopatrol_robot/patrol_node.py`
```python
self.declare_parameter('target_points', [
    0.0, 0.0, 0.0,  #(x,y,z)
    3.0, 1.0, 0.0,
    0.0, 2.0, 0.0
])
```
### 3. 启动自动巡逻
```bash
# 终端3: 启动巡逻节点
ros2 launch autopatrol_robot autopatrol.launch.py
```
# 功能模块说明
## 1. 自动巡逻 (autopatrol_robot)
### patrol_node.py: 主要的巡逻控制节点

* 继承 BasicNavigator 提供导航能力
* 支持多点循环巡逻
* 在每个巡逻点进行图像记录
* 集成语音播报功能
* 通过 TF 获取实时位置信息
### speaker.py: 语音播报模块
- 提供语音合成服务
- 支持到达目标点时的语音提示
## 2. 机器人描述 (smartcar_description)
- 完整的机器人 URDF 模型
- 包含激光雷达、摄像头、IMU 等传感器
- Gazebo 物理仿真配置
- ROS2 控制器集成
## 3. 导航系统 (smartcar_navigation2)
- 基于 Navigation2 的完整导航栈
- AMCL 定位
- 全局和局部路径规划
- 动态避障
- 地图服务
## 4. 基础应用 (fishbot_application)
- 位姿初始化和获取
- 单点导航功能
- 多点路径跟随
- TF 坐标变换工具
# 演示
机器人主体 <img width="2593" height="1754" alt="机器人主体" src="https://github.com/user-attachments/assets/71579d94-843b-4d64-b9d3-22e9199fd2b7" />
Gazebo中地图和机器人 <img width="3265" height="1866" alt="截图 2025-10-26 17-47-00" src="https://github.com/user-attachments/assets/a308b173-8a86-4ae5-90bc-69345c713ab1" />
Slam建图 <img width="2718" height="1638" alt="截图 2025-10-26 17-49-34" src="https://github.com/user-attachments/assets/93c6ec11-d2f5-4beb-a861-7f9a82efbe8f" />
Nav2生成代价地图 <img width="2718" height="1638" alt="截图 2025-10-26 17-49-47" src="https://github.com/user-attachments/assets/3bd3e404-7de5-48f0-bf5e-e10651563cd2" />



