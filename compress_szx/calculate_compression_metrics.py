import os
import sys
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error


def calculate_mse_psnr(img11, img22):
    img1 = np.array(img11)
    img2 = np.array(img22)
    assert img1.shape == img2.shape, "Images must have the same dimensions"

    # Calculate MSE
    mse = mean_squared_error(img1, img2)

    # Calculate PSNR
    psnr = peak_signal_noise_ratio(img1, img2)

    return mse, psnr

def calculate_bitrate(compressed_file_path, image_shape):
    file_size_bytes = os.path.getsize(compressed_file_path)
    file_size_bits = file_size_bytes * 8
    total_pixels = image_shape[0] * image_shape[1]
    bpp = file_size_bits / total_pixels
    return bpp


def write_result_to_file(output_file, error_bounded, mse, psnr, bitrate):
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as file:
        file.write(f"{error_bounded}: MSE: {mse:.2f}, PSNR: {psnr:.2f} dB, Bitrate: {bitrate:.2f} bits/pixel\n")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python calculate_compression_metrics.py <original_file> <compressed_file> <output_file> <error_bounded> <image_format>")
        sys.exit(1)

    original_file = sys.argv[1]
    compressed_file = sys.argv[2]
    output_file = sys.argv[3]
    error_bounded = sys.argv[4]
    image_format = sys.argv[5]

    try:
        original_image = Image.open(original_file).convert('RGB')
        compressed_image = Image.open(compressed_file).convert('RGB')

        
        mse, psnr = calculate_mse_psnr(original_image, compressed_image)
        bitrate = calculate_bitrate(compressed_file, compressed_image.size)
        write_result_to_file(output_file, error_bounded, mse, psnr, bitrate)
        
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")
