import numpy as np
from PIL import Image
import sys

def load_channels_from_binary_files(r_file, g_file, b_file, width, height):
    # Read the binary files
    r_channel = np.fromfile(r_file, dtype=np.float32).reshape((height, width)).astype(np.uint8)
    g_channel = np.fromfile(g_file, dtype=np.float32).reshape((height, width)).astype(np.uint8)
    b_channel = np.fromfile(b_file, dtype=np.float32).reshape((height, width)).astype(np.uint8)
    
    # Stack the channels to form an image array
    img = np.stack((r_channel, g_channel, b_channel), axis=-1)
    return img

def load_image_from_combined_binary_file(file_path, width, height):
    data = np.fromfile(file_path, dtype=np.float32)
    channel_length = width * height
    
    # Ensure the data size matches the expected size
    expected_size = channel_length * 3
    actual_size = data.size
    
    if actual_size != expected_size:
        raise ValueError(f"Size mismatch: expected {expected_size} elements, got {actual_size} elements")
    
    # Split the data into R, G, B channels
    r_channel = data[:channel_length].reshape((height, width)).astype(np.uint8)
    g_channel = data[channel_length:2*channel_length].reshape((height, width)).astype(np.uint8)
    b_channel = data[2*channel_length:].reshape((height, width)).astype(np.uint8)
    
    # Stack the channels to form an image array
    img = np.stack((r_channel, g_channel, b_channel), axis=-1)
    return img

def save_image_from_array(image_array, output_path):
    img = Image.fromarray(image_array, 'RGB')
    img.save(output_path)

def main():
    if len(sys.argv) != 5 and len(sys.argv) != 7:
        print("Usage:")
        print("1. python script.py <combined_file> <image_width> <image_height> <output_image_path>")
        print("2. python script.py <r_file> <g_file> <b_file> <image_width> <image_height> <output_image_path>")
        return

    if len(sys.argv) == 5:
        combined_file = sys.argv[1]
        image_width = int(sys.argv[2])
        image_height = int(sys.argv[3])
        output_image_path = sys.argv[4]

        # Load image from combined binary file
        reconstructed_image = load_image_from_combined_binary_file(combined_file, image_width, image_height)
    else:
        r_file = sys.argv[1]
        g_file = sys.argv[2]
        b_file = sys.argv[3]
        image_width = int(sys.argv[4])
        image_height = int(sys.argv[5])
        output_image_path = sys.argv[6]

        # Load image from separate binary files
        reconstructed_image = load_channels_from_binary_files(r_file, g_file, b_file, image_width, image_height)
    
    # Save the reconstructed image as PNG
    save_image_from_array(reconstructed_image, output_image_path)
    print(f"The binary file(s) have been converted back to a PNG image '{output_image_path}'.")

if __name__ == "__main__":
    main()
