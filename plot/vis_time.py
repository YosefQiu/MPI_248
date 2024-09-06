import matplotlib.pyplot as plt
import numpy as np
import re

# 定义读取和解析文件内容的函数
def parse_file(file_path):
    data = []
    current_group = None  # 初始化为空
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith("process_count:"):
                if current_group is not None:  # 如果 current_group 已经初始化，先保存当前组
                    data.append(current_group)
                current_group = {"process": line.split(":")[1].strip(), "rounds": []}  # 初始化新组
            elif current_group is not None:  # 确保 current_group 已初始化
                if line.startswith("CUDA_DVR:"):
                    current_group["CUDA_DVR"] = float(line.split(":")[1].strip())
                elif line.startswith("alpha_swap:"):
                    current_group["alpha_swap"] = float(line.split(":")[1].strip())
                elif line.startswith("rgb_swap:"):
                    current_group["rgb_swap"] = float(line.split(":")[1].strip())
                elif re.match(r"round\d+:", line):
                    round_data = list(map(float, line.split(":")[1].strip().split()))
                    current_group["rounds"].append(round_data)
        # 将最后的组数据保存到列表中
        if current_group is not None:
            data.append(current_group)
    return data

# 绘制堆叠柱状图的函数
def plot_stacked_bar(data):
    labels = [f"{group['process']} processes" for group in data]
    CUDA_DVR = [group['CUDA_DVR'] for group in data]
    alpha_swap = [group['alpha_swap'] for group in data]

    round_compress = []
    round_decompress = []
    round_communication = []
    gathering = []

    for group in data:
        compress = []
        decompress = []
        communication = []
        
        for r in group['rounds']:
            compress.append(r[0])
            decompress.append(r[1])
            communication.append(r[2])
        
        total_round_time = sum([sum(r) for r in group['rounds']])
        remaining_rgb_swap = group['rgb_swap'] - total_round_time
        
        round_compress.append(compress)
        round_decompress.append(decompress)
        round_communication.append(communication)
        gathering.append(remaining_rgb_swap)

    x = np.arange(len(labels))  
    bar_width = 0.35
    fig, ax = plt.subplots()

    # 底层 CUDA_DVR
    ax.bar(x, CUDA_DVR, label='CUDA_DVR', width=bar_width)

    # 第二层 alpha_swap
    ax.bar(x, alpha_swap, bottom=CUDA_DVR, label='alpha_swap', width=bar_width)

    # 定义相同的颜色
    compress_color = 'lightblue'
    decompress_color = 'lightgreen'
    communication_color = 'lightcoral'

    for i in range(len(data)):
        current_bottom = np.array(CUDA_DVR) + np.array(alpha_swap)

        for j in range(len(round_compress[i])):
            ax.bar(x[i], round_compress[i][j], bottom=current_bottom[i], color=compress_color, label='compress' if i == 0 and j == 0 else "", width=bar_width)
            current_bottom[i] += round_compress[i][j]

            ax.bar(x[i], round_decompress[i][j], bottom=current_bottom[i], color=decompress_color, label='decompress' if i == 0 and j == 0 else "", width=bar_width)
            current_bottom[i] += round_decompress[i][j]

            ax.bar(x[i], round_communication[i][j], bottom=current_bottom[i], color=communication_color, label='communication' if i == 0 and j == 0 else "", width=bar_width)
            current_bottom[i] += round_communication[i][j]

        # 添加 image gathering
        ax.bar(x[i], gathering[i], bottom=current_bottom[i], label='image gathering' if i == 0 else "", width=bar_width)

    ax.set_xlabel('Process Count')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Distributed Volume Rendering with Lossy Compression')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()
    plt.show()

# 主程序
if __name__ == "__main__":
    file_path = input("Enter the path to the TXT file: ")
    data = parse_file(file_path)
    plot_stacked_bar(data)
