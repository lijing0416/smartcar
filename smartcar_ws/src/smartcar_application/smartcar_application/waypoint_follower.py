from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
import rclpy
from rclpy.duration import Duration

def main():
    rclpy.init()
    nav = BasicNavigator()
    nav.waitUntilNav2Active() # 等待导航可用

    goal_poses = []
    goal_pose1 = PoseStamped()
    goal_pose1.header.frame_id = "map"
    goal_pose1.header.stamp = nav.get_clock().now().to_msg()
    goal_pose1.pose.position.x = 0.0
    goal_pose1.pose.position.y = 0.0
    goal_pose1.pose.orientation.w = 1.0
    goal_poses.append(goal_pose1)
    goal_pose2 = PoseStamped()
    goal_pose2.header.frame_id = "map"
    goal_pose2.header.stamp = nav.get_clock().now().to_msg()
    goal_pose2.pose.position.x = 2.0
    goal_pose2.pose.position.y = 0.0
    goal_pose2.pose.orientation.w = 1.0
    goal_poses.append(goal_pose2)
    goal_pose3 = PoseStamped()
    goal_pose3.header.frame_id = "map"
    goal_pose3.header.stamp = nav.get_clock().now().to_msg()
    goal_pose3.pose.position.x = 2.0
    goal_pose3.pose.position.y = 2.0
    goal_pose3.pose.orientation.w = 1.0
    goal_poses.append(goal_pose3)
    # 调用路点服务
    nav.followWaypoints(goal_poses)
    # 判断结束及获取反馈
    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        nav.get_logger().info(f'当前目标编号：{feedback.current_waypoint}')
        #nav.cancelTask()
    result = nav.getResult()
    nav.get_logger().info(f'导航结果：{result}')
    
    # rclpy.spin()
    # rclpy.shutdown()
    