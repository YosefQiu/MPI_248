import os
import matplotlib.pyplot as plt

def plot_compression_ratios(file_paths):
    # Define a list of colors for different lines
    colors = ['#a0d2db', '#fbc4ab', '#bde0fe', '#ffafcc', '#c1e1c5', '#ffc3a0', '#ffebb5', '#fcd5ce', '#e2ece9', '#bfc5a8', '#ffcad4', '#e8e8a6']

    plotted_points = set()

    for idx, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            x_values = []
            compression_ratios = []
            for line in lines:
                parts = line.split(': ')
                threshold = parts[0]
                ratio = float(parts[2].split()[0])
                x_values.append(threshold)
                compression_ratios.append(ratio)
            
            # Get the base file name without extension for the legend
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Plot with specified color
            plt.plot(x_values, compression_ratios, marker='o', label=base_name, color=colors[idx % len(colors)])
            
            # # Display the value at each point
            # for x, y in zip(x_values, compression_ratios):
            #     # Check if this point has been plotted before
            #     if (x, y) not in plotted_points:
            #         plt.text(x, y, f'{y:.2f}', fontsize=8, ha='right', va='bottom')
            #         plotted_points.add((x, y))

    plt.title('Compression Ratio vs Error Bound')
    plt.xlabel('Error Bound')
    plt.ylabel('Compression Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
plot_compression_ratios(['szx_CR.txt', 'sz3_CR.txt', 'cusz_CR.txt'])
