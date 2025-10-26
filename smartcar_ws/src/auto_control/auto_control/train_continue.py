import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from auto_control.smartcar_env import SmartCarEnv
from collections import deque
import random
import time


# ============ 定义网络结构 ============
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ============ Replay Buffer ============
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s_),
            torch.FloatTensor(d).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# ============ SAC Agent ============
class SACAgent:
    def __init__(self, state_dim, action_dim, device):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.buffer = ReplayBuffer()
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # entropy 温度

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def soft_update(self, net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        s, a, r, s_, d = self.buffer.sample(batch_size)
        s, a, r, s_, d = s.to(self.device), a.to(self.device), r.to(self.device), s_.to(self.device), d.to(self.device)

        with torch.no_grad():
            next_action = self.actor(s_)
            next_q1 = self.critic1_target(s_, next_action)
            next_q2 = self.critic2_target(s_, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * torch.log(torch.abs(next_action) + 1e-6).mean(dim=-1, keepdim=True)
            target_q = r + (1 - d) * self.gamma * next_q

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        # 更新 actor
        action_pred = self.actor(s)
        actor_loss = -(self.critic1(s, action_pred)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft update target
        self.soft_update(self.critic1_target, self.critic1, self.tau)
        self.soft_update(self.critic2_target, self.critic2, self.tau)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic1_optim': self.critic1_optim.state_dict(),
            'critic2_optim': self.critic2_optim.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'], strict=False)
        print(f"[SAC] 已加载 actor 权重：{path}")


# ============ 主训练逻辑 ============
def train_continue():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SmartCarEnv()
    agent = SACAgent(env.state_dim, env.action_dim, device)

    # === 加载已有模型 ===
    actor_path = "/home/ljyyds/smartcar/smartcar_ws/checkpoint/best_actor.pth"
    if os.path.exists(actor_path):
        agent.load(actor_path)
        print("[续训] 成功加载已有模型。")
    else:
        print("[续训] 未找到旧模型，将重新开始训练。")

    max_episodes = 2000
    batch_size = 64

    for ep in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(500):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)
            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"[Ep {ep}] 总奖励: {episode_reward:.2f}")

        # 每隔100轮保存模型
        if ep % 100 == 0:
            save_path = f"/home/ljyyds/smartcar/smartcar_ws/checkpoint/sac_continue_ep{ep}.pth"
            agent.save(save_path)
            print(f"[保存] 模型已保存至 {save_path}")

    env.cleanup()

def main(args=None):
    train_continue()

if __name__ == "__main__":
    train_continue()
