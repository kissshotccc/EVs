import random

import numpy as np
import pandas as pd

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_data():
    # 50个节点
    data_temp = pd.read_csv('数据集/data.csv', header=None)
    data = []
    for i in range(0, len(data_temp)):
        data.append((data_temp.iloc[i][0], data_temp.iloc[i][1]))

    # 节点间距离
    distance_temp = pd.read_csv('数据集/distance.csv', header=None)
    distance = distance_temp.values

    # 两节点之间是否为充电路段
    roads_temp = pd.read_csv('数据集/roads.csv', header=None)
    roads = roads_temp.values

    # 两个节点之间的行驶速度
    speed_temp = pd.read_csv('数据集/speed.csv', header=None)
    speed = speed_temp.values

    # 两个节点之间是否有路径 主对角线为0
    """
    [[0. 1. 0. ... 1. 1. 0.]    表示在0号节点可到达的位置为[0. 1. 0. ... 1. 1. 0.]，对应的可执行的动作为[1,...,97,98]
     [1. 0. 1. ... 1. 1. 1.]
     [0. 1. 0. ... 1. 1. 0.]
     ...
     [1. 1. 1. ... 0. 1. 1.]
     [1. 1. 1. ... 1. 0. 1.]
     [0. 1. 0. ... 1. 1. 0.]]
    """
    node_road = speed.copy()
    for i in range(len(node_road)):
        for j in range(len(node_road[i])):
            if node_road[i][j] > 0:
                node_road[i][j] = 1

    print(node_road)

    env_info = {
        'data': data,
        'distance': distance,
        'speed': speed,
        'roads': roads,
        'node_road': node_road
    }
    return env_info

def get_model_name():
    model_list = []
    for end in range(0, 100):
        for max_power in range(40, 101, 2):
            for consumption in range(10, 31):
                model_name = 'model_' + str(end) + '_' + str(max_power) + '_' + str(consumption)
                model_list.append(model_name)
    return model_list

def generate_random_car():
    start, end = random.sample(range(0, 100), 2)
    init_power = random.randint(10, 50)
    max_power = random.randint(40, 100)
    dead_line = np.random.normal(2.1, 0.1)
    consumption = random.uniform(10, 30) / 100
    Evs = {
        'start': start,
        'end': end,
        'init_power': init_power,
        'max_power': max_power,
        'dead_line': dead_line,
        'consumption': consumption
    }
    return Evs