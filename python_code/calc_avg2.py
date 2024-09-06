import re
import sys
from collections import defaultdict

# 定义函数来判断一行是否包含中文字符
def contains_chinese(line):
    return any('\u4e00' <= char <= '\u9fff' for char in line)

# 从命令行获取文件名
if len(sys.argv) != 2:
    print("Usage: python script_name.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

# 使用 defaultdict 来存储每个第一列相同的数值的第三列总和和计数
data = defaultdict(lambda: {"sum": 0, "count": 0})

# 打开文件并逐行读取
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 忽略空行和包含中文的行
            if line.strip() == "" or contains_chinese(line):
                continue
            
            # 使用正则表达式提取行中的数字
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            
            # 如果找到数字，且行中有至少3个数字，处理第一列和第三列
            if numbers and len(numbers) >= 3:
                key = int(numbers[0])  # 第一列作为 key
                value = float(numbers[2])  # 第三列作为 value
                data[key]["sum"] += value
                data[key]["count"] += 1

    # 计算每个第一列相同的平均值
    averages = {key: data[key]["sum"] / data[key]["count"] for key in data}

    # 输出结果
    for key, avg in averages.items():
        print(f"第一列为 {key} 的数据组，第三列的平均值为: {avg:.5f}")

except FileNotFoundError:
    print(f"文件 {file_path} 未找到。")
