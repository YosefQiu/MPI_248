import numpy as np
from PIL import Image
import sys

def read_image_section(file_path, min_x, min_y, max_x, max_y):
    img = Image.open(file_path).convert('RGB')
    img_array = np.array(img)

    # Ensure the specified range is within the image dimensions
    assert 0 <= min_x < img_array.shape[1]
    assert 0 <= min_y < img_array.shape[0]
    assert min_x < max_x <= img_array.shape[1]
    assert min_y < max_y <= img_array.shape[0]

    # Extract the section of the image
    img_section = img_array[min_y:max_y, min_x:max_x]

    # Separate R, G, B channels
    r_channel = img_section[:, :, 0].astype(np.float32).flatten()
    g_channel = img_section[:, :, 1].astype(np.float32).flatten()
    b_channel = img_section[:, :, 2].astype(np.float32).flatten()

    return r_channel, g_channel, b_channel

def save_to_binary_files(r_channel, g_channel, b_channel, base_file_name):
    r_channel.tofile(base_file_name + '_r.bin')
    g_channel.tofile(base_file_name + '_g.bin')
    b_channel.tofile(base_file_name + '_b.bin')
    return len(r_channel), len(g_channel), len(b_channel)

def merge_binary_files(r_channel, g_channel, b_channel, output_file):
    combined = np.concatenate((r_channel, g_channel, b_channel))
    combined.tofile(output_file)
    return combined.size

def main():
    if len(sys.argv) < 6:
        print("Usage: python script.py <mode> <file_path> <min_x> <min_y> <max_x> <max_y>")
        print("Mode 1: Save RGB channels to separate binary files")
        print("Mode 2: Merge RGB binary files to a single binary file")
        return

    mode = int(sys.argv[1])
    file_path = sys.argv[2]
    min_x, min_y = int(sys.argv[3]), int(sys.argv[4])
    max_x, max_y = int(sys.argv[5]), int(sys.argv[6])
    
    if mode == 1:
        # Read the image section
        r_channel, g_channel, b_channel = read_image_section(file_path, min_x, min_y, max_x, max_y)
        # Save the channels to separate binary files
        r_len, g_len, b_len = save_to_binary_files(r_channel, g_channel, b_channel, 'rgb_channels')
        image_width, image_height = max_x - min_x, max_y - min_y
        print("R, G, B channels have been saved to separate binary files.")
        print(f"Image section dimensions: Width = {image_width}, Height = {image_height}")
        print(f"R channel length: {r_len}, G channel length: {g_len}, B channel length: {b_len}")

    elif mode == 2:
        r_channel, g_channel, b_channel = read_image_section(file_path, min_x, min_y, max_x, max_y)
        total_len = merge_binary_files(r_channel, g_channel, b_channel, 'rgb_channels.bin')
        image_width, image_height = max_x - min_x, max_y - min_y
        print("R, G, B channels have been saved to a single binary file.")
        print(f"Image section dimensions: Width = {image_width}, Height = {image_height}")
        print(f"File channel length: {total_len}")
    
    else:
        print("Invalid mode. Use 1 for saving RGB channels, or 2 for merging RGB channels.")

if __name__ == "__main__":
    main()
