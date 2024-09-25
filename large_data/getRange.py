import numpy as np
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Calculate max, min and difference from a .raw file")
parser.add_argument('file_path', type=str, help="Path to the .raw file")

# 解析命令行参数
args = parser.parse_args()

# 读取原始文件中的数据
data = np.fromfile(args.file_path, dtype=np.uint8)  # 假设数据类型为 float32，你可以根据实际情况修改

# 计算最大值、最小值及它们的差值
max_val = np.max(data)
min_val = np.min(data)
difference = max_val - min_val

# 输出结果
print(f"Max value: {max_val}")
print(f"Min value: {min_val}")
print(f"Difference (Max - Min): {difference}")
