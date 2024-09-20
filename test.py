# import gym
# from IPython import display
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
#
# 50个节点
data_50_temp = pd.read_csv('data_set/data_50.csv', header=None)
data_50 = []
for i in range(0, 50):
    data_50.append((data_50_temp.iloc[i][0], data_50_temp.iloc[i][1]))
#
# # 节点间距离
# distance_50_temp = pd.read_csv('data_set/distance_50.csv', header=None)
# distance_50 = distance_50_temp.values
#
# # 两节点之间是否为充电路段
# road_50_temp = pd.read_csv('data_set/roads_50.csv', header=None)
# road_50 = road_50_temp.values
#
# # 两个节点之间的行驶速度
# speed_50_temp = pd.read_csv('data_set/speed_50.csv', header=None)
# speed_50 = speed_50_temp.values
#
# # 每辆电车的信息
# EVs_50_temp = pd.read_csv('data_set/EVs_50.csv', header=None)
# EVs_50 = {}
# for i in range(0, 50):
#     EVs_50[i] = {}
#     EVs_50[i]['start'] = data_50.index((EVs_50_temp.iloc[i][0], EVs_50_temp.iloc[i][1]))
#     EVs_50[i]['end'] = data_50.index((EVs_50_temp.iloc[i][2], EVs_50_temp.iloc[i][3]))
#     EVs_50[i]['init_power'] = EVs_50_temp.iloc[i][4]
#     EVs_50[i]['max_power'] = EVs_50_temp.iloc[i][5]
#     EVs_50[i]['dead_line'] = EVs_50_temp.iloc[i][6]
#     EVs_50[i]['consumption'] = EVs_50_temp.iloc[i][7]
#
# node_road = np.ones((50, 50))
# # 将主对角线元素设为0
# np.fill_diagonal(node_road, 0)
#
# env_info = [data_50, distance_50, speed_50, node_road]
#
#
# print(EVs_50[1])
#

# import matplotlib.pyplot as plt
# import networkx as nx
#
# # 节点的经纬度列表
# # node = [(40.619935, -73.908708), (40.663012, -73.962211), (40.619318, -73.920946), (40.666382, -73.883617), (40.592947, -73.993383)]
#
# # 创建一个图
# G = nx.Graph()
#
# # 添加节点
# for i, n in enumerate(data_50):
#     G.add_node(i, pos=n)
#
# # 为完全图添加边
# for i in range(len(data_50)):
#     for j in range(i + 1, len(data_50)):
#         G.add_edge(i, j)
#
# # 获取节点位置用于绘图
# pos = nx.get_node_attributes(G, 'pos')
#
# # 创建绘图
# plt.figure(figsize=(8, 6))
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10, font_weight='bold')
#
# # 设置坐标轴标签
# plt.title('基于节点坐标的完全无向图')
# plt.xlabel('经度')
# plt.ylabel('纬度')
#
# # 显示图形
# plt.show()


