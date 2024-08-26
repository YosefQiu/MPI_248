import os
import numpy as np
import matplotlib.pyplot as plt

def plot_compression_decompression_times(file_paths):
    x_values = set()
    compression_data = {}
    decompression_data = {}

    # Step 1: Collect all x_values and map times to them
    for file_path in file_paths:
        compression_times = {}
        decompression_times = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.split()
                error_bound = parts[0]
                compression_times[error_bound] = float(parts[1])
                decompression_times[error_bound] = float(parts[2])
                x_values.add(error_bound)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        compression_data[base_name] = compression_times
        decompression_data[base_name] = decompression_times

    # Step 2: Sort x_values and prepare the data for plotting
    x_values = sorted(x_values, key=lambda v: float(v), reverse=True)  # Reverse the order to start from the largest value
    x = np.arange(len(x_values))
    width = 0.35 / len(file_paths)  # Adjusted width to fit all bars
    total_width = width * len(file_paths)  # Total width of one group (compression + decompression)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for each file
    colors = ['#88c1cf', '#fbc4ab', '#ffebb5', '#bfc5a8', '#ffcad4', '#e8e8a6']

    # Step 3: Plot the data
    for i, (base_name, compression_times) in enumerate(compression_data.items()):
        comp_values = [compression_times.get(x_val, np.nan) for x_val in x_values]
        ax.bar(x - total_width/2 + i * width, comp_values, width, 
               label=f'Compression Time ({base_name})', color=colors[i % len(colors)])
        
    for i, (base_name, decompression_times) in enumerate(decompression_data.items()):
        decomp_values = [decompression_times.get(x_val, np.nan) for x_val in x_values]
        ax.bar(x + total_width/2 + i * width, decomp_values, width, 
               label=f'Decompression Time ({base_name})', color=colors[(i + len(file_paths)) % len(colors)])

    # Step 4: Customize plot appearance
    ax.set_title('Compression and Decompression Times (Excluding I/O Time)')
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Time (s)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.legend(loc='upper left', ncol=2, fontsize='small')

    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_compression_decompression_times(['szx_CT.txt', 'sz3_CT.txt', 'cusz_CT.txt'])
plot_compression_decompression_times(['szx_CT.txt', 'cusz_CT.txt'])
