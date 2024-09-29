#!/bin/bash
#SBATCH --job-name=perlin_noise      # 作业名称
#SBATCH --account=m4259              # 项目账户
#SBATCH --qos=regular                  # 使用 debug 队列
#SBATCH --time=01:00:00              # 运行时间上限 5 分钟
#SBATCH --nodes=1                    
#SBATCH --constraint=cpu         # 使用 Haswell 节点



# 定义分辨率和输出路径
resolution=4096
output_path=/pscratch/sd/q/qiuyf

# 运行生成Perlin噪声的Python脚本
python genNoise.py $resolution $output_path

# 生成 8192 分辨率的噪声
resolution2=8192
python genNoise.py $resolution2 $output_path
