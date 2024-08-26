from PIL import Image
import numpy as np
import sys

def binary_to_png(input_filename, output_filename, width, height):
    # 读取R通道数据
    with open(input_filename, 'rb') as file:
        r_channel = np.fromfile(file, dtype=np.float32, count=width * height)
        g_channel = np.fromfile(file, dtype=np.float32, count=width * height)
        b_channel = np.fromfile(file, dtype=np.float32, count=width * height)

    # 将R, G, B通道数据重组为 (height, width) 格式
    r_channel = r_channel.reshape((height, width))
    g_channel = g_channel.reshape((height, width))
    b_channel = b_channel.reshape((height, width))

    # 将R, G, B通道数据合并成一个 (height, width, 3) 的数组
    data = np.stack((r_channel, g_channel, b_channel), axis=-1)

    # 将数据归一化到0-255范围内，并转换为uint8类型
    data = np.clip(data, 0, 1) * 255
    data = data.astype(np.uint8)

    # 使用PIL保存图像
    image = Image.fromarray(data, 'RGB')
    image.save(output_filename)

def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <combined_file> <image_width> <image_height> <output_image_path>")
        return

    input_filename = sys.argv[1]
    image_width = int(sys.argv[2])
    image_height = int(sys.argv[3])
    output_filename = sys.argv[4]

    binary_to_png(input_filename, output_filename, image_width, image_height)

if __name__ == "__main__":
    main()
