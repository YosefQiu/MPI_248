import sys

def calculate_reduction(original_bytes, compressed_bytes):
    """
    计算通信量减少的百分比。
    
    :param original_bytes: 不使用有损压缩时的总通信量（字节数）
    :param compressed_bytes: 使用有损压缩时的总通信量（字节数）
    :return: 减少的百分比
    """
    reduction = original_bytes - compressed_bytes
    reduction_percentage = (reduction / original_bytes) * 100
    return reduction, reduction_percentage

def process_file(file_path):
    """
    处理输入的TXT文件，计算每行压缩前后的减少百分比，并将结果追加到文件中。
    
    :param file_path: 输入文件的路径
    """
    lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            data = line.strip().split()
            if len(data) == 2:
                compressed_bytes = int(data[0])
                original_bytes = int(data[1])
                
                _, reduction_percentage = calculate_reduction(original_bytes, compressed_bytes)
                
                # 在行末添加减少的百分比
                new_line = f"{line.strip()} {reduction_percentage:.2f}%\n"
                file.write(new_line)
            else:
                file.write(line)  # 如果行格式不正确，保持原样

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    process_file(file_path)
    print(f"Processed file: {file_path}")
