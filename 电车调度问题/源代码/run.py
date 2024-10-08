import re
import shutil
import numpy as np
import pandas as pd
import torch

from env import EVs_Env
from DQN import DQN
import csv

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


def get_max_model(dir_path):
    pattern = re.compile(r"model_(\d+)\.pth")

    # Initialize the maximum model number
    max_num = -1

    # Traverse through all subdirectories of the parent directory
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            match = pattern.match(file)

            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

    return max_num

def get_best_model(env,num):
    dir_path = f'./model/best_power_path{num+1}'
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            model_path = os.path.join(root,file)
            model_power = DQN_test(env,num,model_path)
            if model_power == env.max_power:
                save_best_model(model_path,num)
                print("car{}保存成功！".format(num+1))
                return -1
                break
        if model_power != env.max_power:
            return num+1

def save_best_model(model_path,num):
    destination_folder = './model/best_power_path'  #保存模型的位置
    #检查目标文件夹是否存在，如果不存在则创建一个
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)


    file_extension = os.path.splitext(model_path)[1]

    #构造新的文件名
    new_model_path = f'car_{num+1}{file_extension}'
    destination_path = os.path.join(destination_folder,new_model_path)

    #复制文件到目标路径
    if not os.path.isfile(new_model_path):
        shutil.copy(model_path,destination_path)


def DQN_test(env, num, model_path, max_steps=500):
    # 创建DQN对象
    agent = DQN(env)

    # # 加载保存的模型权重
    # model_num = get_max_model(f'./model/best_power_path{num+1}')
    # print(model_num)
    # model_path = f'./model/best_power_path{num+1}/model_{model_num}.pth'
    # # model_path = f'./model/best_power_path/model_car_{num+1}.pth'
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()  # 设置模型为评估模式

    # [电动汽车序号、起点序号、终点序号、初始电量、电池容量、截止时间]
    info_list = [num, env.start, env.end, env.init_power, env.max_power, env.dead_line]

    # 电车调度路径 [(s1,s2),(s2,s3)...]
    path_list = []

    # [[s1-s2的剩余电量, s1->s2的剩余截止时间]...]
    power_time_list = []

    # 汇总表 [序号, 到终点时剩余电量, 到终点时剩余时间]
    summary = []
    last_power = 0

    state, _ = env.reset()
    # 循环执行每一步操作
    for step in range(max_steps):
        pre = env.current
        # 使用模型选择动作（不使用epsilon-greedy策略，直接用模型的输出）
        action = agent.take_action(state, 1)
        if action == pre:
            action = agent.take_action(state, 2)

        next_state, reward, terminated, info = env.step(action)

        path_list.append((pre,env.current))
        power_time_list.append([env.current_power,env.remain_time])

        state = next_state
        if terminated:
            if env.current == env.end:
                summary.append([num, env.current_power, env.remain_time])
                last_power = env.current_power
            else:
                summary.append([num, 0, env.remain_time])
                last_power = 0
            break
    #保存到csv
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

    return last_power


if __name__ == "__main__":
    EVs, env_info = init_data()
    # 生成50轮结果，即50轮总电量，最后取平均值
    res = 0
    # loss_car_list = []
    # # for i in range(50):
    # test_list = [40]
    # for i in test_list:
    #     i -= 1
    #     env = EVs_Env(EVs[i], env_info)
    #     env.reset()
    #     if get_best_model(env,i) != -1:
    #         loss_car_list.append(get_best_model(env,i))
    # print(loss_car_list)
    for k in range(1):
        # 打开 CSV 文件，使用写模式 'w' 清空文件内容
        with open('../输出结果/results_all.csv', 'w', newline='', encoding='utf-8') as file:
            pass  # 不写入任何内容，相当于清空文件
        # 打开 CSV 文件，使用写模式 'w' 清空文件内容
        with open('../输出结果/results_detail.csv', 'w', newline='', encoding='utf-8') as file:
            pass  # 不写入任何内容，相当于清空文件

        # 测试1辆车
        for i in range(len(EVs)):
            env = EVs_Env(EVs[i], env_info)
            env.reset()
            # # 加载保存的模型权重
            model_path = f'./model/best_power_path/car_{i+1}.pth'
            DQN_test(env, i,model_path)

        # 读取 CSV 文件
        df = pd.read_csv('../输出结果/results_all.csv', header=None)

        # 获取第二列并计算和（假设第二列的索引为1）
        total_power = df.iloc[:, 1].sum()
        res += (total_power)
    print(res/1)