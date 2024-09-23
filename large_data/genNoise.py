import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise3
import sys
import vtk
from vtk.util import numpy_support

# 设置分辨率

# 将数据从浮点数转换为 unsigned char 类型
min_val = np.min(perlin_noise_data)
max_val = np.max(perlin_noise_data)

# 如果 max_val 和 min_val 相同，会导致归一化出现问题，应该加以检查
if max_val > min_val:
    perlin_noise_data_uint8 = np.uint8((perlin_noise_data - min_val) / (max_val - min_val) * 255)
else:
    # 处理特例：如果所有值都相同
    perlin_noise_data_uint8 = np.zeros_like(perlin_noise_data, dtype=np.uint8)

# 保存为 .raw 文件
raw_filename = file_path + f"/perlin_noise_data_{resolution}.raw"
perlin_noise_data_uint8.tofile(raw_filename)

# 读取生成的梯度噪声数据
noise_data = np.load(npy_filename)

# 创建 VTK 图像数据对象
imageData = vtk.vtkImageData()
imageData.SetDimensions(noise_data.shape)
imageData.SetSpacing(1.0, 1.0, 1.0)

# 将 numpy 数组转换为 VTK 数组
vtk_data_array = numpy_support.numpy_to_vtk(num_array=noise_data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

# 设置 VTK 数组
imageData.GetPointData().SetScalars(vtk_data_array)

# 保存为 .vti 文件
vti_filename = file_path + f"/perlin_noise_data_{resolution}.vti"
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(vti_filename)
writer.SetInputData(imageData)
writer.Write()

