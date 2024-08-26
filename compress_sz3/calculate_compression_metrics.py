import os
import sys
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error

def read_rgb_binary_file(file_path, width, height):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    size = width * height
    r = data[:size].reshape((height, width))
    g = data[size:2*size].reshape((height, width))
    b = data[2*size:3*size].reshape((height, width))
    return r, g, b

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def calculate_psnr(mse, max_pixel_value=255.0):
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_pixel_value ** 2 / mse)

def calculate_mse_psnr(img11, img22, wwidth, hheight):
    r1, g1, b1 = read_rgb_binary_file(img11, wwidth, hheight)
    r2, g2, b2 = read_rgb_binary_file(img22, wwidth, hheight)

    # 计算每个通道的 MSE
    mse_r = calculate_mse(r1, r2)
    mse_g = calculate_mse(g1, g2)
    mse_b = calculate_mse(b1, b2)

    # 平均 MSE
    mse = (mse_r + mse_g + mse_b) / 3

    # 计算 PSNR
    psnr = calculate_psnr(mse)
    return mse, psnr


def calculate_bitrate(compressed_file_path, width, height):
    # 获取压缩文件大小（以字节为单位）
    compressed_file_size_bytes = os.path.getsize(compressed_file_path)
    # 将压缩文件大小转换为比特
    compressed_file_size_bits = compressed_file_size_bytes * 8

    total_pixels = width * height

    # 计算每像素位数（bpp）
    bpp = compressed_file_size_bits / total_pixels
    return bpp


def write_result_to_file(output_file, error_bounded, mse, psnr, bitrate):
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as file:
        file.write(f"{error_bounded}: MSE: {mse:.2f}, PSNR: {psnr:.2f} dB, Bitrate: {bitrate:.2f} bits/pixel\n")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python calculate_compression_metrics.py <original_file> <decompressed_file>  <compressed_file> <output_file> <error_bounded> <image_width> <image_height>")
        sys.exit(1)

    original_file = sys.argv[1]
    decompressed_file = sys.argv[2]
    compressed_file = sys.argv[3]
    output_file = sys.argv[4]
    error_bounded = sys.argv[5]
    wwidth = int(sys.argv[6])
    hheight = int(sys.argv[7])
    try:
        mse, psnr = calculate_mse_psnr(original_file, decompressed_file, wwidth, hheight)
        bitrate = calculate_bitrate(compressed_file, wwidth, hheight)
        write_result_to_file(output_file, error_bounded, mse, psnr, bitrate)
        
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")
