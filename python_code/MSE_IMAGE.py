from PIL import Image
import numpy as np
import math

# 计算均方误差（MSE）
def calculate_mse(image1, image2):
    # 将图像转换为numpy数组
    img1 = np.array(image1)
    img2 = np.array(image2)
    
    # 计算MSE（均方误差）
    mse = np.mean((img1 - img2) ** 2)
    return mse

# 计算PSNR（峰值信噪比）
def calculate_psnr(mse, max_pixel_value=255.0):
    if mse == 0:
        return float('inf')  # 如果MSE为0，返回无穷大
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

# 查找最大差异像素
def find_max_diff_pixel(image1, image2):
    img1 = np.array(image1)
    img2 = np.array(image2)
    
    # 计算差异矩阵
    diff = np.abs(img1 - img2)
    
    # 找到最大差异
    max_diff = np.max(diff)
    max_diff_index = np.unravel_index(np.argmax(diff), diff.shape)
    
    return max_diff, max_diff_index

# 加载两个图像
image1 = Image.open('/global/homes/q/qiuyf/MPI_248/build/ground_truth.png')
image2 = Image.open('/global/homes/q/qiuyf/MPI_248/build/output_16.png')

# 计算MSE和PSNR
mse = calculate_mse(image1, image2)
psnr = calculate_psnr(mse)

# 查找最大误差像素
max_diff, max_diff_index = find_max_diff_pixel(image1, image2)

# 输出结果
print(f'MSE: {mse}')
print(f'PSNR: {psnr} dB')
print(f'Maximum difference: {max_diff} at pixel {max_diff_index}')
