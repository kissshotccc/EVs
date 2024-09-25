import csv
import json
import sys

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import *  # 用于显示进度条

from env import GymHelper, EVs_Env
from PPO import PPO
from DQN import DQN

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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

def DQN_train(env, num):
    # 定义超参数
    max_episodes = 2000  # 训练episode数量
    max_steps = 500  # 每个回合的最大步数
    batch_size = 32  # 采样数量

    training_time = []

    # 创建DQN对象
    agent = DQN(env)

    # 定义保存每个回合奖励的列表
    episode_rewards = [num]

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

        # 打印中间值
        if episode % 40 == 0:
            tqdm.write("Episode " + str(episode) + ": " + str(episode_reward) + "====>path:" + str(path))


    # 以追加模式 'a' 打开 CSV 文件
    with open('../输出结果/rewards.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据（追加一行）
        writer.writerow(episode_rewards)

    # 每训练完一辆小车，保存网络模型参数
    model_path = f'./model/dqn_model_car_{num}.pth'
    torch.save(agent.model.state_dict(), model_path)

    # 使用Matplotlib绘制奖励值的曲线图
    plt.plot(episode_rewards)
    plt.title("reward")
    plt.show()

def DQN_test(env, num, max_steps=500):
    # 创建DQN对象
    agent = DQN(env)

    # 加载保存的模型权重
    model_path = f'./model/dqn_model_car_{num}.pth'
    agent.model.load_state_dict(torch.load(model_path, weights_only=True))
    agent.model.eval()  # 设置模型为评估模式

    # [电动汽车序号、起点序号、终点序号、初始电量、电池容量、截止时间]
    info_list = [num, env.start, env.end, env.init_power, env.max_power, env.dead_line]

    # 电车调度路径 [(s1,s2),(s2,s3)...]
    path_list = []

    # [[s1-s2的剩余电量, s1->s2的剩余截止时间]...]
    power_time_list = []

    # 汇总表 [序号, 到终点时剩余电量, 到终点时剩余时间]
    summary = []

    state, _ = env.reset()

    # 循环执行每一步操作
    for step in range(max_steps):
        pre = env.current
        # 使用模型选择动作（不使用epsilon-greedy策略，直接用模型的输出）
        action = agent.choose_action(state)
        next_state, reward, terminated, info = env.step(action)

        path_list.append((pre,env.current))
        power_time_list.append([env.current_power,env.remain_time])

        state = next_state

        if terminated:
            if env.current == env.end:
                summary.append([num, env.current_power, env.remain_time])
            else:
                summary.append([num, 0, env.remain_time])
            break

    # 保存到csv
    with open('../输出结果/results_detail.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据（追加一行）
        writer.writerow(info_list)
        writer.writerow(path_list)
        writer.writerow(power_time_list)

    with open('../输出结果/results_all.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据（追加一行）
        for row in summary:
            writer.writerow(row)



if __name__ == '__main__':
    EVs, env_info = init_data()
    # print(json.dumps(EVs_50, indent=2))

    # 训练50辆车
    # for i in range(len(EVs)):
    #     env = EVs_Env(EVs[i], env_info)
    #     env.reset()
    #     DQN_train(env, i)


    # 生成50轮结果，即50轮总电量，最后取平均值
    res = 0
    for k in range(50):
        # 打开 CSV 文件，使用写模式 'w' 清空文件内容
        with open('../输出结果/results_all.csv', 'w', newline='', encoding='utf-8') as file:
            pass  # 不写入任何内容，相当于清空文件
        # 打开 CSV 文件，使用写模式 'w' 清空文件内容
        with open('../输出结果/results_detail.csv', 'w', newline='', encoding='utf-8') as file:
            pass  # 不写入任何内容，相当于清空文件

        # 测试50辆车
        for i in range(len(EVs)):
            env = EVs_Env(EVs[i], env_info)
            env.reset()
            DQN_test(env, i)

        # 读取 CSV 文件
        df = pd.read_csv('../输出结果/results_all.csv', header=None)

        # 获取第二列并计算和（假设第二列的索引为1）
        total_power = df.iloc[:, 1].sum()
        res += total_power

    print(res/50)


    # 单辆测试
    # env = EVs_Env(EVs[0], env_info)
    # env.reset()
    # DQN_train(env, 0)
    # DQN_test(env, 0)