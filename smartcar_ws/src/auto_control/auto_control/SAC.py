import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    """Actor网络：输入状态，输出动作（连续值）"""
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出动作维度（1）
        self.action_bound = action_bound  # 动作边界（1.0）

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 输出动作范围：[-action_bound, action_bound]（用tanh缩放）
        action = torch.tanh(self.fc3(x)) * self.action_bound
        return action

class Critic(nn.Module):
    """Critic网络：输入状态+动作，输出Q值（双Q网络之一）"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        # 输入维度 = 状态维度 + 动作维度（40 + 1 = 41）
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出Q值（单值）

    def forward(self, state, action):
        # 拼接状态和动作（维度：batch×(40+1)=batch×41）
        cat = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class SAC:
    """Soft Actor-Critic 算法"""
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr=3e-4, critic_lr=1e-3, alpha_lr=1e-4,
                 target_entropy=-0.5, tau=0.005, gamma=0.95, device='cpu'):
        # 设备（CPU/GPU）
        self.device = device

        # 1. 初始化Actor和Critic网络
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)

        # 2. 初始化目标Critic网络（延迟更新）
        self.target_critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)
        # 目标网络参数初始化为与当前网络一致
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 3. 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # 4. 熵系数α（自动学习）
        self.target_entropy = target_entropy  # 目标熵
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()

        # 5. 其他超参数
        self.tau = tau  # 软更新系数
        self.gamma = gamma  # 折扣因子
        self.action_bound = action_bound  # 动作边界

    def take_action(self, state):
        """选择动作（带高斯噪声探索）"""
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32, device=self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        # 添加高斯噪声（增强探索，均值0，标准差0.1）
        noise = np.random.normal(0, 0.1, size=action.shape)
        action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return action

    def soft_update(self, net, target_net):
        """软更新目标网络参数"""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self, states, actions, rewards, next_states, dones):
        """更新SAC的Actor、Critic和α"""
        # 转换为Tensor
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # ---------------- 1. 更新Critic网络 ----------------
        # 计算目标Q值（双Q网络取最小值，降低过估计）
        next_actions = self.actor(next_states)
        target_q1 = self.target_critic_1(next_states, next_actions)
        target_q2 = self.target_critic_2(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        # 目标Q值：r + γ * (1-done) * (target_q - α*log_prob)（熵正则化）
        # 一维动作无log_prob，简化为 r + γ*(1-done)*target_q
        target_q = rewards + self.gamma * (1 - dones) * target_q

        # 计算当前Q值
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        # 计算Critic损失（MSE）
        critic_1_loss = F.mse_loss(current_q1, target_q.detach())
        critic_2_loss = F.mse_loss(current_q2, target_q.detach())

        # 优化Critic
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ---------------- 2. 更新Actor网络 ----------------
        # 冻结Critic，避免更新时影响
        for param in self.critic_1.parameters():
            param.requires_grad = False
        for param in self.critic_2.parameters():
            param.requires_grad = False

        # 计算Actor损失（最大化Q值 - α*熵，即最小化负Q值）
        new_actions = self.actor(states)
        q1 = self.critic_1(states, new_actions)
        q2 = self.critic_2(states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = -torch.mean(q)  # 一维动作简化，无熵项

        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 解冻Critic
        for param in self.critic_1.parameters():
            param.requires_grad = True
        for param in self.critic_2.parameters():
            param.requires_grad = True

        # ---------------- 3. 更新熵系数α ----------------
        # 一维动作简化，α损失设为0（或根据需求调整）
            alpha_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()  # 现在是张量，可正常调用 backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # ---------------- 4. 软更新目标Critic网络 ----------------
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)