from PIL import Image
import numpy as np
import math

def calculate_mse(image1, image2):
    # Convert images to numpy arrays
    img1 = np.array(image1)
    img2 = np.array(image2)
    
    # Compute the MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)
    return mse

def calculate_psnr(mse, max_pixel_value=255.0):
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

# Load the two images
image1 = Image.open('ground_truth.png')
image2 = Image.open('output_2.png')

# Calculate MSE and PSNR
mse = calculate_mse(image1, image2)
psnr = calculate_psnr(mse)

# Output the results
print(f'MSE: {mse}')
print(f'PSNR: {psnr} dB')
