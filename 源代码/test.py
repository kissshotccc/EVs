import csv

a = [
    [1,2,3,4],
    [1,2,3,3]
]

with open('test.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入数据（追加一行）
    for row in a:
        writer.writerow(row)
