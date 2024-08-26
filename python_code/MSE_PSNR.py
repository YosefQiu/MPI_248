import numpy as np
from skimage.io import imread
import pandas as pd

def calculate_mse_psnr(image1_path, image2_path):
    # Load the two images
    image1 = imread(image1_path)
    image2 = imread(image2_path)

    # Ensure the images have the same size
    assert image1.shape == image2.shape, "Images must have the same dimensions and number of channels"

    # Calculate the squared differences
    squared_diff = (image1 - image2) ** 2

    # Calculate MSE
    mse = np.mean(squared_diff)

    # Calculate PSNR
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse)) if mse != 0 else float('inf')

    # Find the maximum squared difference
    max_squared_diff = np.max(squared_diff)

    # Convert the squared differences to a DataFrame for better visualization
    squared_diff_df = pd.DataFrame(squared_diff.reshape(-1, squared_diff.shape[-1]), columns=['R', 'G', 'B', 'A'])

    # Find the row with the maximum difference
    max_diff_index = np.unravel_index(np.argmax(squared_diff, axis=None), squared_diff.shape)
    max_diff_row = squared_diff_df.iloc[max_diff_index[0] * squared_diff.shape[1] + max_diff_index[1]]

    return mse, psnr, max_squared_diff, squared_diff_df, max_diff_row

def save_squared_differences_to_csv(squared_diff_df, output_csv_path):
    # Save the squared differences DataFrame to a CSV file
    squared_diff_df.to_csv(output_csv_path, index=False)

# Example usage
image1_path = './res.png'
image2_path = './output_rank_0.png'
output_csv_path = 'squared_differences.csv'

mse, psnr, max_squared_diff, squared_diff_df, max_diff_row = calculate_mse_psnr(image1_path, image2_path)
print(f"MSE: {mse}")
print(f"PSNR: {psnr} dB")
print(f"Maximum Squared Difference: {max_squared_diff}")
print(f"Row with Maximum Difference:\n{max_diff_row}")

# Save the squared differences to a CSV file
save_squared_differences_to_csv(squared_diff_df, output_csv_path)
print(f"Squared differences saved to {output_csv_path}")