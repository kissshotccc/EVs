import csv

a = [
    [1,2,3,4],
    [1,2,2,3],
    [1,1,1,1]
]

# 打开 CSV 文件，使用写模式 'w' 清空文件内容
with open('test.csv', 'w', newline='', encoding='utf-8') as file:
    pass  # 不写入任何内容，相当于清空文件

with open('test.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入数据（追加一行）
    for row in a:
        writer.writerow(row)

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('test.csv', header=None)

# 获取第二列并计算和（假设第二列的索引为1）
second_column_sum = df.iloc[:, 1].sum()

# 打印第二列的和
print("第二列的总和：", second_column_sum)
