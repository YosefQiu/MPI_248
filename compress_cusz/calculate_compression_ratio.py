import os
import sys

def calculate_compression_ratio(original_file, compressed_file):
    # 获取原始文件大小
    original_size = os.path.getsize(original_file)
    print(f"Original file size: {original_size} bytes")
    
    # 获取压缩后文件大小
    compressed_size = os.path.getsize(compressed_file)
    print(f"Compressed file size: {compressed_size} bytes")
    
    # 计算压缩比
    if compressed_size == 0:
        raise ValueError("Compressed file size is zero, cannot compute compression ratio.")
    
    compression_ratio = original_size / compressed_size
    return compression_ratio

def write_result_to_file(output_file, ratio, error_bounded):
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as file:
        file.write(f"{error_bounded}: Compression ratio: {ratio:.2f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python calculate_compression_ratio.py <original_file> <compressed_file> <output_file> <error_bounded>")
        sys.exit(1)

    original_file = sys.argv[1]
    compressed_file = sys.argv[2]
    output_file = sys.argv[3]
    error_bounded = sys.argv[4]

    try:
        ratio = calculate_compression_ratio(original_file, compressed_file)
        print(f"Compression ratio: {ratio:.2f}")
        write_result_to_file(output_file, ratio, error_bounded)
    except Exception as e:
        print(f"Error: {e}")
