import os
import matplotlib.pyplot as plt

def plot_psnr_vs_bitrate(file_paths):
    # Define a list of colors for different lines
    colors = ['#88c1cf', '#fbc4ab', '#ffebb5', '#bfc5a8', '#ffcad4', '#e8e8a6']
    
    for idx, file_path in enumerate(file_paths):
        psnr_values = []
        bitrate_values = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.split(', ')
                
                # Extract PSNR and Bitrate
                psnr_value = float(parts[1].split(': ')[1].split()[0])
                bitrate_value = float(parts[2].split(': ')[1].split()[0])
                
                psnr_values.append(psnr_value)
                bitrate_values.append(bitrate_value)

        # Get the base file name without extension for the legend
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Plot with specified color
        plt.plot(bitrate_values, psnr_values, marker='o', linestyle='-', label=base_name, color=colors[idx % len(colors)])
        
        # # Display the value at each point
        # for x, y in zip(bitrate_values, psnr_values):
        #     plt.text(x, y, f'{y:.2f}', fontsize=8, ha='right', va='bottom')

    plt.title('PSNR vs Bitrate')
    plt.xlabel('Bitrate (bits/pixel)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
plot_psnr_vs_bitrate(['szx_MPB.txt', 'sz3_MPB.txt', 'cusz_MPB.txt'])
