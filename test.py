import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
data = pd.read_csv('reward.csv')

# 去掉第一列，保留训练次数的数据
# episode_rewards = data.iloc[:, 1:]

# 计算每列的均值
mean_rewards = data.mean(axis=0)
std_rewards = data.std(axis=0)

# 绘制平均值曲线
plt.figure(figsize=(10, 6))
plt.errorbar(range(1, len(mean_rewards) + 1), mean_rewards, yerr=std_rewards, fmt='-', ecolor='green', color='green', capsize=3)

# 设置标题和标签
plt.title('平均奖励 vs 训练次数', fontsize=14)
plt.xlabel('幕（训练次数）', fontsize=12)
plt.ylabel('平均奖励', fontsize=12)

# 设置训练次数间隔显示
plt.xticks(np.arange(0, len(mean_rewards), step=100))

# 显示网格线
plt.grid(True)

# 保存图片
plt.savefig('episode_rewards.jpg')

# 显示图片
plt.show()