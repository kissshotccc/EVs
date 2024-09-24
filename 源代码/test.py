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
