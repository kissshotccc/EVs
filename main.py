import json
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import *  # 用于显示进度条

from env import GymHelper, EVs_Env
from PPO import PPO
from DQN import DQN



def init_data():
    # 50个节点
    data_50_temp = pd.read_csv('data_set/data_50.csv', header=None)
    data = []
    for i in range(0, 50):
        data.append((data_50_temp.iloc[i][0], data_50_temp.iloc[i][1]))

    # 节点间距离
    distance_50_temp = pd.read_csv('data_set/distance_50.csv', header=None)
    distance = distance_50_temp.values

    # 两节点之间是否为充电路段
    roads_50_temp = pd.read_csv('data_set/roads_50.csv', header=None)
    roads = roads_50_temp.values

    # 两个节点之间的行驶速度
    speed_50_temp = pd.read_csv('data_set/speed_50.csv', header=None)
    speed = speed_50_temp.values

    # 每辆电车的信息
    EVs_50_temp = pd.read_csv('data_set/EVs_50.csv', header=None)
    EVs = {}
    for i in range(0, 50):
        EVs[i] = {}
        EVs[i]['start'] = data.index((EVs_50_temp.iloc[i][0], EVs_50_temp.iloc[i][1]))
        EVs[i]['end'] = data.index((EVs_50_temp.iloc[i][2], EVs_50_temp.iloc[i][3]))
        EVs[i]['init_power'] = EVs_50_temp.iloc[i][4]
        EVs[i]['max_power'] = EVs_50_temp.iloc[i][5]
        EVs[i]['dead_line'] = EVs_50_temp.iloc[i][6]
        EVs[i]['consumption'] = EVs_50_temp.iloc[i][7]

    node_road = np.ones((50, 50))
    # 将主对角线元素设为0
    np.fill_diagonal(node_road, 0)

    env_info = {
        'data': data,
        'distance': distance,
        'speed': speed,
        'roads' : roads,
        'node_road': node_road
    }
    return EVs, env_info

def DQN_test(env):
    # 定义超参数
    max_episodes = 2000  # 训练episode数量
    max_steps = 500  # 每个回合的最大步数
    batch_size = 32  # 采样数量

    # 创建DQN对象
    agent = DQN(env)

    # 定义保存每个回合奖励的列表
    episode_rewards = []

    # 开始循环，tqdm用于显示进度条并评估任务时间开销
    for episode in tqdm(range(max_episodes), file=sys.stdout):
        # 重置环境并获取初始状态
        state, _ = env.reset()
        # 当前回合的奖励
        episode_reward = 0

        # 循环进行每一步操作
        for step in range(max_steps):
            # 根据当前状态选择动作
            action = agent.choose_action(state)
            # 执行动作，获取新的信息
            next_state, reward, terminated, truncated, info = env.step(action)
            # 判断是否达到终止状态
            done = terminated or truncated

            # 将这个五元组加入到缓冲区中
            agent.replay_buffer.add(state, action, reward, next_state, done)
            # 累计奖励
            episode_reward += reward

            # 如果经验回放缓冲区已经有足够数据，就更新网络参数
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            # 更新当前状态
            state = next_state

            if done:
                break

        # 记录当前回合奖励值
        episode_rewards.append(episode_reward)

        # 打印中间值
        if episode % 40 == 0:
            tqdm.write("Episode " + str(episode) + ": " + str(episode_reward))

def PPO_test(env):
    # 定义超参数
    max_episodes = 10000  # 训练episode数
    max_steps = 200  # 每个回合的最大步数

    # 创建PPO对象
    agent = PPO(env)
    # 定义保存每个回合奖励的列表
    episode_rewards = []

    # 开始训练，tqdm用于显示进度条并评估任务时间开销
    for episode in tqdm(range(max_episodes), file=sys.stdout):
        # 重置环境并获取初始状态
        state, _ = env.reset()
        # 当前回合的奖励
        episode_reward = 0
        # 记录每个episode的信息
        buffer = []

        # 循环进行每一步操作
        for step in range(max_steps):
            # 根据当前状态选择动作
            action = agent.choose_action(state)
            # 执行动作，获取反馈信息
            next_state, reward, terminated, truncated, info = env.step(action)
            # 判断是否达到终止状态
            done = terminated or truncated

            # 将这个五元组添加到buffer中
            buffer.append((state, action, reward, next_state, done))
            # 累计奖励
            episode_reward += reward

            # 更新当前状态
            state = next_state

            if done:
                break

        # 更新策略
        agent.update(buffer)
        # 记录当前回合奖励值
        episode_rewards.append(episode_reward)

        # 打印中间值
        if episode % (max_episodes // 10) == 0:
            tqdm.write("Episode " + str(episode) + ": " + str(episode_reward))

    # 使用Matplotlib绘制奖励值的曲线图
    plt.plot(episode_rewards[:50])
    plt.title("reward")
    plt.show()


if __name__ == '__main__':
    EVs, env_info = init_data()
    # # print(json.dumps(EVs_50, indent=2))
    #
    # # 环境测试
    env = EVs_Env(EVs[0], env_info)
    env.reset()
    # # gym_helper = GymHelper(env)
    #
    # for i in range(10):
    #     # gym_helper.render(title=str(i))
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print([observation, reward, terminated, truncated, info])

    # env.close()
    DQN_test(env)





