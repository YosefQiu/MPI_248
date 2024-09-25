import numpy as np
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Normalize a .raw file to range 0-1 and save as float32")
parser.add_argument('file_path', type=str, help="Path to the .raw file")
parser.add_argument('output_path', type=str, help="Path to save the normalized .raw file")

# 解析命令行参数
args = parser.parse_args()

# 读取原始文件中的数据，假设数据是 uint8 类型的
data = np.fromfile(args.file_path, dtype=np.uint8)

# 找到数据中的最大值和最小值
max_val = np.max(data)
min_val = np.min(data)

# 归一化到 0 到 1 的范围
normalized_data = (data - min_val) / (max_val - min_val)

# 将归一化后的数据保存为 uint8 格式
normalized_data.astype(np.uint8).tofile(args.output_path)

print(f"Data normalized and saved to {args.output_path}")
print(f"Min value after normalization: {np.min(normalized_data)}")
print(f"Max value after normalization: {np.max(normalized_data)}")
