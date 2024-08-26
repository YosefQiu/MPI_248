import os
import struct

# Replace 'your_file.bin' with the path to your .bin file
file_path = 'output_rank_0.bin'

# Define the data type and its size in bytes
# Example for float (4 bytes), double (8 bytes), int (4 bytes)
data_type = 'float'  # Change this based on your file's data type

# Get the size of one element based on the data type
if data_type == 'float':
    element_size = struct.calcsize('f')
elif data_type == 'double':
    element_size = struct.calcsize('d')
elif data_type == 'int':
    element_size = struct.calcsize('i')
else:
    raise ValueError(f"Unknown data type: {data_type}")

# Get the file size in bytes
file_size = os.path.getsize(file_path)

# Calculate the number of elements
num_elements = file_size // element_size

# Print the number of elements
print(f"Number of elements in the file: {num_elements}")
