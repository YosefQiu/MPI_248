import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置文件夹路径和文件前缀
folder_path = './box/'  # 替换为你的文件夹路径
file_prefix = 'cusz'
file_suffix = '_CT.txt'

# 初始化字典来存储每个阈值对应的压缩和解压缩时间
compression_times = {}
decompression_times = {}

# 遍历文件夹中的20个文件
for i in range(1, 21):
    file_name = f'{file_prefix}{i}{file_suffix}'
    file_path = os.path.join(folder_path, file_name)
    
    # 读取文件中的数据
    df = pd.read_csv(file_path, sep='\s+', header=None)
    
    # 按照阈值进行分类
    for index, row in df.iterrows():
        threshold = row[0]
        compression_time = row[1]
        decompression_time = row[2]
        
        if threshold not in compression_times:
            compression_times[threshold] = []
        if threshold not in decompression_times:
            decompression_times[threshold] = []
        
        compression_times[threshold].append(compression_time)
        decompression_times[threshold].append(decompression_time)

# 使用GridSpec创建一个画布，并为压缩和解压缩时间分别分配空间
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1])

# 左上子图（压缩时间，较大范围）
ax1_top = fig.add_subplot(gs[0, 0])
ax1_top.boxplot([compression_times[threshold] for threshold in sorted(compression_times.keys())],
                labels=[f'{threshold:.0e}' for threshold in sorted(compression_times.keys())])
ax1_top.set_ylim(4.4, 4.6)  # 调整上方图的 y 轴范围
ax1_top.spines['bottom'].set_visible(False)  # 隐藏底部边框
ax1_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 隐藏 x 轴的刻度和标签
ax1_top.set_title('Compression Times')

# 左下子图（压缩时间，较小范围）
ax1_bottom = fig.add_subplot(gs[1, 0])
ax1_bottom.boxplot([compression_times[threshold] for threshold in sorted(compression_times.keys())],
                   labels=[f'{threshold:.0e}' for threshold in sorted(compression_times.keys())])
ax1_bottom.set_ylim(0.55, 1.0)  # 调整下方图的 y 轴范围
ax1_bottom.spines['top'].set_visible(False)  # 隐藏顶部边框
ax1_bottom.set_xlabel('Error Bound')
ax1_bottom.set_ylabel('Time (ms)')

# 右侧子图（解压缩时间）
ax2 = fig.add_subplot(gs[:, 1])
ax2.boxplot([decompression_times[threshold] for threshold in sorted(decompression_times.keys())],
            labels=[f'{threshold:.0e}' for threshold in sorted(decompression_times.keys())])
ax2.set_ylim(0.55, 1.0)  # 调整 y 轴范围以更好地显示数据
ax2.set_title('Decompression Times')
ax2.set_xlabel('Error Bound')
ax2.set_ylabel('Time (ms)')

# 手动调整左右两个图的 y-label 位置，使其对齐
ax1_bottom.yaxis.set_label_coords(-0.1, 0.5)
ax2.yaxis.set_label_coords(-0.1, 0.37)

# 添加波浪线来表示断轴
d = .015  # 波浪线的大小
kwargs = dict(transform=ax1_top.transAxes, color='k', clip_on=False)
ax1_top.plot((-d, +d), (-d, +d), **kwargs)  # 左上角
ax1_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右上角

kwargs.update(transform=ax1_bottom.transAxes)  # 改变波浪线的参考系
ax1_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左下角
ax1_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右下角

plt.tight_layout()
plt.show()
