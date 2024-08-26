import sys

def compare_files(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        file1_lines = file1.readlines()
        file2_lines = file2.readlines()

    differences = []
    for i, (line1, line2) in enumerate(zip(file1_lines, file2_lines), start=1):
        if line1 != line2:
            differences.append(i)

    if len(file1_lines) != len(file2_lines):
        longer_file = file1_path if len(file1_lines) > len(file2_lines) else file2_path
        print(f"The files have different lengths. {longer_file} has more lines.")
        for j in range(len(file1_lines), len(file2_lines)):
            differences.append(j + 1)

    return differences

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_files.py <file1> <file2>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    differences = compare_files(file1_path, file2_path)

    if not differences:
        print("The files are identical.")
    else:
        print("The files differ at the following lines:")
        for diff in differences:
            print(f"Line {diff}")