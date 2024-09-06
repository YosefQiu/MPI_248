import re
import sys

# 定义函数来判断一行是否包含中文字符
def contains_chinese(line):
    return any('\u4e00' <= char <= '\u9fff' for char in line)

# 从命令行获取文件名
if len(sys.argv) != 2:
    print("Usage: python script_name.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

# 初始化存储数字的列表
valid_numbers = []

# 打开文件并逐行读取
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 忽略空行和包含中文的行
            if line.strip() == "" or contains_chinese(line):
                continue
            
            # 使用正则表达式提取行中的数字
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            
            # 如果找到数字，且行中有至少3个数字，取最后一个数字
            if numbers and len(numbers) >= 3:
                valid_numbers.append(float(numbers[-1]))

    # 计算有效数字的平均值
    if valid_numbers:
        average_value = sum(valid_numbers) / len(valid_numbers)
        print("提取的有效数字:", valid_numbers)
        print("有效数字的平均值:", average_value)
    else:
        print("没有找到有效的数字。")

except FileNotFoundError:
    print(f"文件 {file_path} 未找到。")
