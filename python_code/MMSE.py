import numpy as np
import sys

def load_binary_file(filename, shape, dtype=np.float32):
    data = np.fromfile(filename, dtype=dtype)
    return data.reshape(shape)

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def calculate_psnr(mse, max_pixel_value=1.0):
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_pixel_value**2 / mse)

def find_max_diff_pixel(image1, image2):
    diff = np.abs(image1 - image2)
    max_diff = np.max(diff)
    max_diff_index = np.unravel_index(np.argmax(diff), diff.shape)
    return max_diff, max_diff_index

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_mse_psnr.py <filename1> <filename2>")
        sys.exit(1)

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    width = 512
    height = 512
    channels = 4  # RGBA

    # 加载二进制文件
    image1 = load_binary_file(filename1, (height, width, channels))
    image2 = load_binary_file(filename2, (height, width, channels))

    # 计算MSE
    mse = calculate_mse(image1, image2)
    print(f'MSE: {mse}')

    # 计算PSNR
    psnr = calculate_psnr(mse, max_pixel_value=1.0)  # 假设最大像素值为1.0
    print(f'PSNR: {psnr} dB')

    # 找到差异最大的像素点
    max_diff, max_diff_index = find_max_diff_pixel(image1, image2)
    print(f'Max difference: {max_diff} at pixel index {max_diff_index}')

    # 将最大差异的像素点索引转换为行和列
    pixel_index = max_diff_index[:2]  # 去掉通道索引，只保留行和列
    row, col = pixel_index
    print(f'Max difference at row {row}, column {col}')