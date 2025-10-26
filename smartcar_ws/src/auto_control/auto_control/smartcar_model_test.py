import rclpy
import torch
import numpy as np
from time import sleep

# 导入环境和 SAC 类
from .smartcar_env import SmartCarEnv
from .SAC import SAC


def rl_net_test(env):
    # ------------------- 超参数 -------------------
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    hidden_dim = 64
    gamma = 0.95
    tau = 0.005
    target_entropy = -0.5

    # 动作上限（训练时定义的）
    action_bound = env.action_space.high[0]

    # 状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"[env] state_dim = {state_dim}, action_dim = {action_dim}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # 初始化 SAC agent
    agent = SAC(state_dim, hidden_dim, action_dim, action_bound,
                actor_lr, critic_lr, alpha_lr,
                target_entropy, tau, gamma, device)

    # 加载训练好的 actor 模型
    actor_path = "actor_final_1026_0008.pth"
    try:
        state_dict = torch.load(actor_path, map_location=device)
        agent.actor.load_state_dict(state_dict, strict=False)
        agent.actor.eval()
        print(f"[test] 成功加载模型：{actor_path}")
    except Exception as e:
        print(f"[error] 模型加载失败：{e}")
        return

    # ------------------- 测试循环 -------------------
    state = env.reset()
    episode_reward = 0
    episode = 1
    print("开始测试智能小车...")

    while rclpy.ok():
        # 用 actor 输出动作（去掉 evaluate 参数）
        with torch.no_grad():
            action = agent.take_action(state)  # 注意：去掉 evaluate=True

        # 限制动作在动作空间
        action = np.clip(action, -action_bound, action_bound)

        # 与环境交互
        next_state, reward, done = env.step(action)
        episode_reward += reward

        print(f"[Episode {episode}] reward: {episode_reward:.3f}")

        if done:
            print(f"✅ Episode {episode} 结束，总奖励: {episode_reward:.2f}\n")
            state = env.reset()
            episode_reward = 0
            episode += 1
        else:
            state = next_state

        sleep(0.05)  # 控制执行频率


def main(args=None):
    rclpy.init(args=args)
    env = SmartCarEnv()

    try:
        rl_net_test(env)
    except KeyboardInterrupt:
        print("\n测试中断。")
    finally:
        # 安全销毁节点
        if hasattr(env, "destroy_node"):
            env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
