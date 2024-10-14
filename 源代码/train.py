import numpy as np
import pandas as pd
import torch
from DQN import DQN
from tqdm import *  # 用于显示进度条
from matplotlib import pyplot as plt
import csv
import sys
from env import EVs_Env

def init_data():
    # 50个节点
    data_50_temp = pd.read_csv('数据集/data_50.csv', header=None)
    data = []
    for i in range(0, 50):
        data.append((data_50_temp.iloc[i][0], data_50_temp.iloc[i][1]))

    # 节点间距离
    distance_50_temp = pd.read_csv('数据集/distance_50.csv', header=None)
    distance = distance_50_temp.values

    # 两节点之间是否为充电路段
    roads_50_temp = pd.read_csv('数据集/roads_50.csv', header=None)
    roads = roads_50_temp.values

    # 两个节点之间的行驶速度
    speed_50_temp = pd.read_csv('数据集/speed_50.csv', header=None)
    speed = speed_50_temp.values

    # 每辆电车的信息
    EVs_50_temp = pd.read_csv('数据集/EVs_50.csv', header=None)
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
        'roads': roads,
        'node_road': node_road
    }
    return EVs, env_info


def DQN_train(env, num):
    # 定义超参数
    max_episodes = 2000  # 训练episode数量
    max_steps = 500  # 每个回合的最大步数
    batch_size = 32  # 采样数量

    # 创建DQN对象
    agent = DQN(env)

    # 定义保存每个回合奖励的列表
    episode_rewards = [num]

    # 保存每个回合的剩余电量
    episode_power = []

    # 开始循环，tqdm用于显示进度条并评估任务时间开销
    for episode in tqdm(range(max_episodes), file=sys.stdout):
        # 重置环境并获取初始状态
        state, _ = env.reset()
        # 当前回合的奖励
        episode_reward = 0
        path = []

        # 循环进行每一步操作
        for step in range(max_steps):
            path.append(env.current)
            # 根据当前状态选择动作
            action = agent.choose_action(state)
            # 执行动作，获取新的信息
            next_state, reward, terminated, info = env.step(action)
            # 判断是否达到终止状态
            done = terminated

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
                path.append(env.current)
                break

        # 记录当前回合奖励值
        episode_rewards.append(episode_reward)

        # 训练完一幕后剩余电量
        if env.current == env.end:
            episode_power.append(env.current_power)
        else:
            episode_power.append(0)

        # 打印中间值
        if episode % 40 == 0:
            tqdm.write("Episode " + str(episode) + ": " + str(episode_reward) + "====>path:" + str(path))


    # 以追加模式 'a' 打开 CSV 文件
    with open('../输出结果/rewards.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据（追加一行）
        writer.writerow(episode_rewards)

    # 每训练完一辆小车，保存网络模型参数
    model_path = f'模型/dqn_model_car_{num}.pth'
    torch.save(agent.model.state_dict(), model_path)

    # 绘制最终剩余电量
    plt.plot(episode_power)
    plt.title("remain_power")
    plt.show()

if __name__ == '__main__':
    EVs, env_info = init_data()
    # print(json.dumps(EVs_50, indent=2))

    # 训练50辆车
    for i in range(len(EVs)):
        env = EVs_Env(EVs[i], env_info)
        env.reset()
        DQN_train(env, i)
