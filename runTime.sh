#!/bin/bash

# 固定参数
input_file="/pscratch/sd/q/qiuyf/perlin_noise_data_1024.raw"
dim_x=1024
dim_y=1024
dim_z=1024
width=8192
height=8192
cam_dy=1.0 
cam_vz=0.6


# 计算旋转角度增量
pi=$(echo "scale=10; 4*a(1)" | bc -l)  # 计算 pi 值
step=$(echo "$pi * 2 / 3" | bc -l)    # 每次增加的角度：2π / 36

# 运行 3 次
for i in {0..2}
do
    cam_dx=$(echo "$i * $step" | bc -l)  # 计算当前的 cam_dx
    iteration_str="Run_$i"               # 构造运行次数的字符串

    echo "Running iteration $i with cam_dx=$cam_dx, cam_dy=$cam_dy, and identifier $iteration_str"
    
    # 执行 srun 命令
    srun --export=ALL --ntasks=16 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=closest ./CUDA_MPI \
        $input_file $dim_x $dim_y $dim_z $width $height $cam_dx $cam_dy $cam_vz 0 0 $iteration_str
    
done
