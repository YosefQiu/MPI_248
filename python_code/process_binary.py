import sys
from PIL import Image
import numpy as np

def binary_to_png(input_filename, output_filename, width, height):
    # 读取二进制文件中的数据
    with open(input_filename, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32)

    # 假设数据是RGBA格式，我们需要将其转换为适合PIL处理的8位格式
    # 首先将数据reshape成 (height, width, 4)
    data = data.reshape((height, width, 4))

    # 将数据归一化到0-255范围内，并转换为uint8类型
    data = np.clip(data, 0, 1) * 255
    data = data.astype(np.uint8)

    # 使用PIL保存图像
    image = Image.fromarray(data, 'RGBA')
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
