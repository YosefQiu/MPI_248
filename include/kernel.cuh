#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include "helper_math.h"
typedef unsigned char VolumeType;

struct Ray {
	float3 o;  // origin
	float3 d;  // direction
};

class CudaKernel
{
public:
	CudaKernel();
	~CudaKernel();
public:
	void initCuda(void* h_volume, cudaExtent volumeSize);
	void freeCudaBuffers();
	void render_kernel(dim3 gridSize, dim3 blockSize, float* d_output, uint imageW, uint imageH, 
		float3 camPos, float3 camLookAt, float3 cam_U, float3 cam_V, float ccam_dz,
		float3 boxMin, float3 boxMax, float3 big_boxMin, float3 big_boxMax, float3 data_compensation,
		float density, float brightness, float transferOffset, float transferScale,
		uint* d_minMaxXY);
public:
	cudaArray* d_volumeArray;
	cudaArray* d_transferFuncArray;
	cudaTextureObject_t texObject;    // For 3D texture
	cudaTextureObject_t transferTex;  // For 1D transfer function texture
	
};

