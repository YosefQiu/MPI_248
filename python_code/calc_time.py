import matplotlib.pyplot as plt
import numpy as np

def read_time_data(file_path):
    """
    从文件中读取时间数据，每行有4个数据：有损压缩的CUDA时间，总时间，不压缩的CUDA时间，不压缩的总时间。
    
    :param file_path: 输入文件的路径
    :return: 数据列表，包含每行的时间数据
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 移除 ms 并转换为 float
            times = [float(t.replace('ms', '')) for t in line.strip().split()]
            data.append(times)
    return data

def plot_time_data(data):
    """
    绘制时间的柱状图，显示总时间及CUDA时间的占比。
    
    :param data: 时间数据的列表
    """
    # 定义标签
    labels = ['With Compression', 'Without Compression']
    
    # 分别提取有损和无损的CUDA时间与总时间
    cuda_times_with_compression = [row[0] for row in data]
    total_times_with_compression = [row[1] for row in data]
    cuda_times_without_compression = [row[2] for row in data]
    total_times_without_compression = [row[3] for row in data]
    
    # 计算柱子的位置
    x = np.arange(len(labels))  # 标签的位置
    width = 0.35  # 柱子的宽度
    
    fig, ax = plt.subplots()
    
    # 绘制柱子
    rects1 = ax.bar(x - width/2, total_times_with_compression, width, label='Total Time with Compression', color='skyblue')
    rects2 = ax.bar(x + width/2, total_times_without_compression, width, label='Total Time without Compression', color='lightcoral')
    
    # 在柱子内部显示CUDA时间
    ax.bar(x - width/2, cuda_times_with_compression, width, label='CUDA Time with Compression', color='blue', alpha=0.7)
    ax.bar(x + width/2, cuda_times_without_compression, width, label='CUDA Time without Compression', color='red', alpha=0.7)
    
    # 添加一些文本标签
    ax.set_ylabel('Time (ms)')
    ax.set_title('Rendering Times with and without Compression')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    file_path = 'time_data.txt'  # 替换为你的文件路径
    data = read_time_data(file_path)
    plot_time_data(data)
