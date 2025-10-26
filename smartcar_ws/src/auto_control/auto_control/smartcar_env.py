import threading
import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from gymnasium import spaces

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty


class SmartCarEnv:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('smartcar_env')

        # ---------------- 状态变量 ----------------
        self.vel_x, self.ang_z = 0.0, 0.0
        self.laser_state = np.zeros(36, dtype=np.float32)
        self.scan_ready = False

        # Gazebo 坐标系下小车位姿
        self.x_g, self.y_g, self.yaw_g = 0.0, 0.0, 0.0
        #目标点
        self.goal_x_g, self.goal_y_g = 12.0, 7.0
        self.prev_dist = None
        self.model_state_ready = False  # 标记是否收到第一条消息

        # ---------------- 动作/状态空间 ----------------
        self.action_dim = 1
        self.max_forward_speed = 2.0
        self.max_turn_speed = 0.5
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self.state_dim = 36 + 4
        high = np.full(self.state_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # ---------------- ROS 发布/订阅 ----------------
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.node.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_callback, 10)

        # reset 服务
        self.reset_world = self.node.create_client(Empty, 'reset_world')
        self.reset_sim = self.node.create_client(Empty, 'reset_simulation')

        # 等待激光数据
        print("[env] 等待激光数据...")
        start_time = time.time()
        while not self.scan_ready and time.time() - start_time < 10.0:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        print("[env] 激光数据已就绪")

        # 等待第一条 model_states
        print("[env] 等待 Gazebo 位姿数据...")
        start_time = time.time()
        while not self.model_state_ready and time.time() - start_time < 10.0:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        print("[env] Gazebo 位姿已就绪")

        self.collision_threshold = 0.2

    # ---------------- 回调函数 ----------------
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
        ranges = np.where(np.isnan(ranges), msg.range_max, ranges)
        self.laser_state = ranges[::10][:36].astype(np.float32)
        self.scan_ready = True

    def model_states_callback(self, msg: ModelStates):
        try:
            idx = msg.name.index('smartcar')
            pose = msg.pose[idx]
            self.x_g = pose.position.x
            self.y_g = pose.position.y

            q = pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
            self.yaw_g = math.atan2(siny_cosp, cosy_cosp)

            self.model_state_ready = True
        except ValueError:
            pass

    # ---------------- 奖励函数 ----------------
    def compute_reward(self, v, w, dist_to_goal, rel_yaw, min_laser):
        reward = 0.0
        # 前进奖励
        reward += 5.0 * (self.prev_dist - dist_to_goal)
        # 朝向奖励
        reward += 2.0 * math.cos(rel_yaw) * (v / self.max_forward_speed)
        reward -= 1.5 * (1 - math.cos(rel_yaw)) * (self.max_forward_speed - v) / self.max_forward_speed

        # 激光碰撞惩罚
        if min_laser < 0.5:
            reward -= (0.5 - min_laser) * 10.0
        # 碰撞终止
        done = False
        if min_laser < self.collision_threshold:
            reward -= 20.0
            done = True
        # 到达目标
        if dist_to_goal < 0.3:
            reward += 50.0
            done = True
            print("小车到达目标点！")

        return reward, done

    # ---------------- step ----------------
    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        dead_zone = 0.05
        if abs(action_val) < dead_zone:
            v, w = 0.0, 0.0
        else:
            v = abs(action_val) * self.max_forward_speed
            w = np.sign(action_val) * abs(action_val) * self.max_turn_speed * 0.5

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

        time.sleep(0.05)
        rclpy.spin_once(self.node, timeout_sec=0.01)

        dx = self.goal_x_g - self.x_g
        dy = self.goal_y_g - self.y_g
        dist_to_goal = math.sqrt(dx**2 + dy**2)
        angle_to_goal = math.atan2(dy, dx)
        rel_yaw = math.atan2(math.sin(angle_to_goal - self.yaw_g),
                             math.cos(angle_to_goal - self.yaw_g))

        min_laser = np.min(self.laser_state)
        state = np.concatenate([self.laser_state, [self.vel_x, self.ang_z, dist_to_goal, rel_yaw]])
        reward, done = self.compute_reward(v, w, dist_to_goal, rel_yaw, min_laser)
        return state, reward, done

    # ---------------- reset ----------------
    def reset(self):
        self.stop()
        self.scan_ready = False
        self.laser_state = np.ones(36, dtype=np.float32) * 10.0
        self.prev_dist = None

        if self.reset_world.wait_for_service(timeout_sec=3.0):
            self.reset_world.call_async(Empty.Request())
        time.sleep(0.5)

        # 等待 model_states 更新
        start_time = time.time()
        while not self.model_state_ready and time.time() - start_time < 5.0:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        dx = self.goal_x_g - self.x_g
        dy = self.goal_y_g - self.y_g
        self.prev_dist = math.sqrt(dx**2 + dy**2)

        state = np.concatenate([self.laser_state, [self.vel_x, self.ang_z, self.prev_dist, 0.0]])
        print(f"[env] 重置完成: Gazebo初始位置({self.x_g:.2f},{self.y_g:.2f}) | 目标({self.goal_x_g:.2f},{self.goal_y_g:.2f}) | 距离: {self.prev_dist:.3f}")
        return state

    # ---------------- 辅助方法 ----------------
    def stop(self):
        self.cmd_pub.publish(Twist())
        print("[env] 小车停止。")

    def cleanup(self):
        self.stop()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[env] 环境已清理。")

    def start_pose_printer(self, interval=3.0):
        """每隔 interval 秒打印小车 Gazebo 坐标"""
        def printer():
            while rclpy.ok():
                # 强制触发回调
                if self.model_state_ready:
                    print(f"[pose] x: {self.x_g:.2f}, y: {self.y_g:.2f}, yaw: {math.degrees(self.yaw_g):.1f}°")
                time.sleep(interval)
        t = threading.Thread(target=printer, daemon=True)
        t.start()