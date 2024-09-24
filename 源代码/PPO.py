# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import sys
import time
import random
import collections
from tqdm import *  # 用于显示进度条

# 策略模型，给定状态生成各个动作的概率
class PolicyModel(nn.Module):
    def __init__(self, grid_size, output_dim):
        super(PolicyModel, self).__init__()
        self.grid_size = grid_size

        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1),  # 使用1x1卷积升维
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=1),  # 使用1x1卷积降维
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        # 使用全连接层构建一个简单的神经网络。ReLU作为激活函数
        # 最后用一个Softmax层，使得输出可以看作是概率分布
        self.fc = nn.Sequential(
            nn.Flatten(),  # 将特征图展平成一维向量
            nn.Linear(grid_size * grid_size * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=1)
        )

    # 定义前向传播，输出动作概率分布
    def forward(self, x):
        x = x.view(-1, 3, self.grid_size, self.grid_size)
        x = self.conv(x)
        action_prob = self.fc(x)
        return action_prob


# 价值模型，给定状态评估其价值
class ValueModel(nn.Module):
    def __init__(self, grid_size):
        super(ValueModel, self).__init__()
        self.grid_size = grid_size

        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1),  # 使用1x1卷积升维
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=1),  # 使用1x1卷积降维
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        # 构造简单的全连接层网络，输出维度为动作空间的维度
        self.fc = nn.Sequential(
            nn.Flatten(),  # 将特征展平为一维向量
            nn.Linear(grid_size * grid_size * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 最终输出是一个标量，表示状态的价值
        )

    # 定义前向传播，输出价值估计
    def forward(self, x):
        x = x.view(-1, 3, self.grid_size, self.grid_size)
        x = self.conv(x)
        value = self.fc(x)
        return value

class PPO:
    # 构造函数，参数包含环境，学习率，折扣因子，优势计算参数，clip参数，训练轮数
    def __init__(self, env, learning_rate=0.00002, gamma=0.99, lamda=0.95, clip_eps=0.1, epochs=10):
        self.env = env
        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.epochs = epochs

        # 判断可用的设备是 CPU 还是 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 根据环境的观测空间和动作空间，定义策略模型和价值模型，并将模型移动到指定设备上
        self.policy_model = PolicyModel(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.value_model = ValueModel(env.observation_space.shape[0]).to(self.device)

        # 定义Adam优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=learning_rate)

    # 使用策略模型生成动作概率分布并采样动作
    def choose_action(self, state):
        # 将状态转换为Tensor输入模型
        state = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            action_prob = self.policy_model(state)

        # 生成分布后采样返回动作
        c = torch.distributions.Categorical(action_prob)
        action = c.sample()
        return action.item()

    # 计算优势估计
    def calc_advantage(self, td_delta):
        # 将TD误差转换为numpy数组
        td_delta = td_delta.cpu().detach().numpy()

        # 初始化优势值和优势估计的列表
        advantage = 0
        advantage_list = []

        # 反向遍历，从最后一步开始推
        for r in td_delta[::-1]:
            # 将当前的TD误差和下一步的优势加权值作为当前步的优势
            advantage = r + self.gamma * self.lamda * advantage
            # 将当前的优势值插入到列表的最前端
            advantage_list.insert(0, advantage)

        # 转换为Tensor后返回
        return torch.FloatTensor(np.array(advantage_list)).to(self.device)

    # 模型更新
    def update(self, buffer):
        # 取出数据，并将其转换为numpy数组
        # 然后进一步转换为Tensor，并将数据移动到指定计算资源设备上
        states, actions, rewards, next_states, dones = zip(*buffer)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        # actions = torch.tensor(np.array(actions)).view(-1, 1).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(self.device)

        with torch.no_grad():
            # 计算旧策略下的动作概率
            old_action_prob = torch.log(self.policy_model(states).gather(1, actions))

            # 计算TD目标
            td_target = rewards + (1 - dones) * self.gamma * self.value_model(next_states)
            td_delta = td_target - self.value_model(states)

        # 优势估计
        advantage = self.calc_advantage(td_delta)

        # 多步更新策略
        for i in range(self.epochs):
            # 计算新的动作概率
            action_prob = torch.log(self.policy_model(states).gather(1, actions))
            # 计算策略的相对变化率
            ratio = torch.exp(action_prob - old_action_prob)

            # CLIP损失
            part1 = ratio * advantage
            part2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            # 计算策略损失
            policy_loss = -torch.min(part1, part2).mean()

            # 计算价值损失
            value_loss = F.mse_loss(self.value_model(states), td_target).mean()

            # 优化过程，反向传播，更新参数
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()