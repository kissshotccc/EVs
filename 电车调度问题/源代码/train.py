import copy
import csv
import json
import sys
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import *  # 用于显示进度条

from env import EVs_Env
from DQN import DQN

from data import get_data, get_model_name, generate_random_car

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# def DQN_train(env, num):
#     # 定义超参数
#     max_episodes = 2000  # 训练episode数量
#     max_steps = 500  # 每个回合的最大步数
#     batch_size = 32  # 采样数量
#
#     training_time = []
#
#     # 创建DQN对象
#     agent = DQN(env)
#
#     # 定义保存每个回合奖励的列表
#     episode_rewards = [num]
#
#     '''最大奖励'''
#     window_size = 30 #这里将动态更新窗口设置为20
#     success_window = [0] * window_size  #用于维护success——times
#     power_window = np.zeros(window_size)   #设置一个窗口，大小固定
#     best_window =  np.zeros(window_size)    #最佳窗口，用于探索最优网络
#     current_successes = 0    #哨兵作用
#     success_times = 0 #用于记录当前窗口中成功到达终点的次数
#
#     best_power = 0
#     best_current_power = 0
#     best_reward = 0
#     best_power_path = []
#     best_reward_path = []
#     best_window_path = []
#
#     # 开始循环，tqdm用于显示进度条并评估任务时间开销
#     for episode in tqdm(range(max_episodes), file=sys.stdout):
#         # # 重置环境并获取初始状态
#         # state, _ = env.reset()
#         state, _ = env.refresh()
#         # 当前回合的奖励
#         episode_reward = 0
#
#         step_loss = []
#
#         # 循环进行每一步操作
#         for step in range(max_steps):
#
#             # 根据当前状态选择动作
#             action = agent.choose_action(state)
#             # 执行动作，获取新的信息
#             next_state, reward, terminated, info = env.step(action)
#             # 判断是否达到终止状态
#             done = terminated
#
#             # 将这个五元组加入到缓冲区中
#             agent.replay_buffer.add(state, action, reward, next_state, done)
#             # 累计奖励
#             episode_reward += reward
#
#             # 如果经验回放缓冲区已经有足够数据，就更新网络参数
#             if len(agent.replay_buffer) > batch_size:
#                 loss = agent.update(batch_size)
#                 step_loss.append(loss)
#
#             # 更新当前状态
#             state = next_state
#
#             if done:
#
#                 break
#
#         # 记录当前回合奖励值
#         episode_rewards.append(episode_reward)
#
#         #env
#         env.episode = episode
#
#         #维护最优奖励值,以及最优窗口，以及最优路径
#         if episode_reward >= best_reward and env.success == True:    #如果当前奖励比最好奖励还大，更新
#             best_reward = episode_reward    #维护最优奖励值
#             best_reward_path = copy.deepcopy(env.path)
#
#         #在保证又到了终点的情况下，电量的最大
#         if env.current_power >= best_power and env.success == True:
#             best_power = env.current_power
#             best_power_path = copy.deepcopy(env.path)
#             best_power_reward = episode_reward
#
#         #滑动窗口，同时对最优窗口进行维护
#         if episode < window_size:   #添柴火
#             power_window[episode] = env.current_power
#             success_window[episode] = int(env.success == True)#终点和路径最后位置相同，说明成功到达终点
#
#             if success_window[episode] == 1:
#                 current_successes += 1
#         else:   #只有当窗口中数值积累到一定才开始进行滑动
#             # 检查并移除被滑出窗口的成功状态
#             current_successes -= success_window[0]
#
#             # 通过左移元素实现窗口滑动
#             power_window[:-1] = power_window[1:]  # 所有元素向左移动一个单位
#             success_window[:-1] = success_window[1:]  # 滑动成功状态窗口
#
#             # 更新最后一个元素
#             power_window[-1] = env.current_power
#             success_window[-1] = int(env.success == True)  # 记录新的是否到达终点
#
#             # 如果新加入的元素成功到达终点，增加 current_successes
#             current_successes += success_window[-1]
#
#             #如果同时满足：1、成功到达终点的数量比最优的大；
#             #2、当前episode的power比之前最优窗口的高(但这样不一定能得到奖励值最高的)
#
#             if current_successes >= success_times and env.current_power >= best_current_power and env.success == True:
#                 np.copyto(best_window,power_window)
#                 success_times = current_successes
#                 best_current_power = env.current_power
#                 best_window_path = copy.deepcopy(env.path)
#
#                 try:#只有当最佳窗口变化的时候，才会进行保存
#                     model_dir = f"./model/best_power_path{num+1}"
#                     os.makedirs(model_dir, exist_ok=True)
#                     model_path_name = f'./model/best_power_path{num+1}/model_{episode}.pth'
#                     torch.save(agent.model.state_dict(), model_path_name)
#                     current_episode = episode
#                 except Exception as e:
#                     print("失败")
#         #每200 次检查一下，如果当前最大电量不等于汽车的最大电量，说明当前可能陷入了局部麻烦，需要重新调整epsilon的大小
#         if episode % 200 == 0 and (best_current_power != env.max_power or episode_reward < 310) :
#             # if current_successes < best_reward:
#             #     break
#             agent.epsilon = 0.2 + episode/200 * 0.02
#             agent.coe = 0.0002 + episode/200 * 0.00002
#         # 打印中间值
#         if episode % 200 == 0:
#             tqdm.write("Episode " + str(episode) + ": " + str(episode_reward) + "====>path:" + str(env.path))
#
#
#     # 使用Matplotlib绘制奖励值的曲线图
#     plt.plot(episode_rewards)
#     plt.title("reward")
#     plt.show()
#
#     # print("best_reward: %f best_power: %f best_power_reward: %f best_current_power: %f" %(best_reward,best_power,best_power_reward,best_current_power))
#     # print("best_reward_path:{} best_power_path: {} best_window_path:{}".format(best_reward_path, best_power_path,best_window_path))
#     # print(best_window,success_times)
#     #print(current_episode)
#
#     return best_current_power

def DQN_train(env, model_name):
    # 定义超参数
    max_episodes = 2000  # 训练episode数量
    max_steps = 500  # 每个回合的最大步数
    batch_size = 64  # 采样数量


    parts = model_name.split('_')
    # 提取出 end, max_power, consumption
    end = int(parts[1])  # 将字符串转换为整数
    max_power = int(parts[2])
    consumption = int(parts[3]) / 100

    # 创建DQN对象
    agent = DQN(env)

    # total_rewards = [num]

    # 开始循环，tqdm用于显示进度条并评估任务时间开销
    for episode in tqdm(range(max_episodes), file=sys.stdout):
        # 重置环境并获取初始状态
        state, _ = env.refresh()
        env.end = end
        env.max_power = max_power
        env.consumption = consumption
        # 当前回合的奖励
        episode_reward = 0

        for step in range(max_steps):

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
                break

        # 记录当前回合奖励值
        # total_rewards.append(episode_reward)

        # 打印中间值
        if episode % 40 == 0:
            tqdm.write("Episode " + str(episode) + ": " + str(episode_reward) + "======>" + str(env.path) + "======>" + "start:" + str(env.start) + ' ' + 'end:' + str(env.end))

    model_path_name = f'./model/{model_name}.pth'
    torch.save(agent.model.state_dict(), model_path_name)



if __name__ == '__main__':
    Evs = generate_random_car()
    env_info = get_data()
    model_name = get_model_name()

    for model in model_name[0:2]:
        env = EVs_Env(Evs, env_info)
        env.reset()
        DQN_train(env, model)

    # sum_list = []
    # #训练50辆车
    # for _ in range(1):
    #     best_powerlist = []
    #     loss_list = []
    #     # for i in range(len(EVs)):
    #     test_list = [1,2]
    #     for i in  test_list:
    #         i -= 1
    #         env = EVs_Env(EVs[i], env_info)
    #         env.reset()
    #         best_power, episode_loss =DQN_train(env,i)
    #         best_powerlist.append(best_power)
    #
    #     sum_list.append(sum(best_powerlist))
    # print(sum_list)
