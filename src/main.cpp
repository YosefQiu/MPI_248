#include "ggl.h"

#include "mpi.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "FileManager.h"
#include "kernel.cuh"

#include "stb_image.h"
#include "stb_image_write.h"
#include "Utils.h"
#include "Processor.h"

const char* default_volumeFilename = "./res/Bucky.raw";
const char* volumeFilename;
cudaExtent volumeTotalSize;

unsigned int image_width;
unsigned int image_height;

dim3 blockSize(16, 16);
dim3 gridSize;

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

int err;
Processor* p = nullptr;

float dx, dy, dz;

#if __linux__
struct timeval startTime;
struct timeval endTime;
struct timeval costStart;
double totalCost = 0;
#endif

int main(int argc, char* argv[])
{
	err = MPI_Init(&argc, &argv);
	if (err != MPI_SUCCESS)
	{
		std::cerr << "[ERROR]::MPI_Init:: MPI Init Error." << std::endl;
		MPI_Abort(MPI_COMM_WORLD, err);
	}
	int xdim, ydim, zdim;
	if (argc == 1)
	{
		volumeFilename = default_volumeFilename;
		xdim = 32; ydim = 32; zdim = 32;
		volumeTotalSize = make_cudaExtent(xdim, ydim, zdim);
		image_width = 512;
		image_height = 512;
		dx = 1.0f; dy = 1.0f; dz = 0.25f;
	}
	else
	{
		volumeFilename = argv[1];
		xdim = std::atoi(argv[2]); ydim = std::atoi(argv[3]); zdim = std::atoi(argv[4]);
		volumeTotalSize = make_cudaExtent(xdim, ydim, zdim);
		if (argc >= 7)
		{
			image_width = std::atoi(argv[5]);
			image_height = std::atoi(argv[6]);
		}
		if (argc >= 9)
		{
			dx = std::atof(argv[7]);
			dy = std::atof(argv[8]);
			dz = std::atof(argv[9]);
		}
		else
		{
			dx = 1.0f; dy = 1.0f; dz = 0.25f;
		}
	}
		
	gridSize = dim3((image_width + blockSize.x - 1) / blockSize.x, (image_height + blockSize.y - 1) / blockSize.y);
	p = new Processor;
	auto rank = p->Processor_ID; auto size = p->Processor_Size;
	char hostname[256];
    gethostname(hostname, sizeof(hostname));  // 获取主机名
	// 先获取节点上可用的 GPU 数量
    
	findCudaDevice(argc, (const char**)argv);
	
	CudaKernel cudakernel;
	int gpu_count;
    cudaGetDeviceCount(&gpu_count);

    // 映射每个进程到特定的 GPU（在本地节点上）
    int gpu_id = p->Processor_ID % gpu_count;
    cudaSetDevice(gpu_id);

    int device_id;
    cudaGetDevice(&device_id);
	std::cout << "Process " << p->Processor_ID 
		<< " is using GPU " << device_id << " / " << gpu_count 
		<< " on " << hostname << std::endl;

	p->initScreen(image_width, image_height);
	p->initRayCaster(p->camera_plane_x, p->camera_plane_y);
	
	if (p->Processor_ID == 0)
		p->init_master(volumeFilename, p->a, p->b, volumeTotalSize);
	else
		p->init_node(p->a, p->b, p->Processor_ID);

	p->initKDTree();
	p->initImage(image_width, image_height);
	p->initData(volumeFilename);
	p->initOpti(); // cam_dx = cam_dy = 0.0
	p->setCameraProperty(dx, dy);

	float3 big_boxMin, big_boxMax;
	big_boxMin.x = -1.0f * p->whole_data_len.x / 2.0; big_boxMin.y = -1.0f * p->whole_data_len.y / 2.0; big_boxMin.z = -1.0f * p->whole_data_len.z / 2.0;
	big_boxMax = big_boxMin + p->whole_data_len + make_float3(-1, -1, -1);
	float3 boxMin = p->bMin; float3 boxMax = p->bMax;
	boxMax.x = p->bMax.x; boxMax.y = p->bMax.y; boxMax.z = p->bMax.z;
	float3 compensation = p->data_compensation;


	printf("Process %d, [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", rank, boxMin.x, boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z, compensation.x, compensation.y, compensation.z);
	//printf("Process %d, [%f, %f, %f],  [%f, %f, %f]\n", rank, big_boxMin.x, big_boxMin.y, big_boxMin.z, big_boxMax.x, big_boxMax.y, big_boxMax.z);

	// size_t local_volume_size = 32.0 * 32.0 * 32.0 * sizeof(VolumeType);
	// size_t volume_size = volumeTotalSize.width * volumeTotalSize.height * volumeTotalSize.depth * sizeof(VolumeType);
	// void* h_volume = FileManager::loadPartialRawFile2(volumeFilename, local_volume_size, 0, 32, 0, 32, 0, 32, volume_size);


	size_t local_volume_size = (p->data_b.x - p->data_a.x + 1) * (p->data_b.y - p->data_a.y + 1) * (p->data_b.z - p->data_a.z + 1) * sizeof(VolumeType);

	unsigned char* d_volume;
	cudaMalloc(&d_volume, local_volume_size);
	cudaMemcpy(d_volume, p->data, local_volume_size, cudaMemcpyHostToDevice);
	// free(h_volume);

	uint h_minMaxXY[4] = { image_width, image_height, 0, 0 };
	uint* d_minMaxXY;
	cudaMalloc(&d_minMaxXY, 4 * sizeof(uint));
	cudaMemcpy(d_minMaxXY, h_minMaxXY, 4 * sizeof(uint), cudaMemcpyHostToDevice);

	//int3 d_volumeSize;
	cudaExtent initCuda_size = make_cudaExtent((p->data_b.x - p->data_a.x + 1), (p->data_b.y - p->data_a.y + 1), (p->data_b.z - p->data_a.z + 1));
	std::cout << "[init cuda size]:: PID [ " << p->Processor_ID << " ] [ " << p->data_b.x - p->data_a.x + 1 << ", " << p->data_b.y - p->data_a.y + 1 << " , " << p->data_b.z - p->data_a.z + 1 << "]" << std::endl;
	//d_volumeSize = make_int3(int(p->b.x - p->a.x + 1), int(p->b.y - p->a.y + 1), int(p->b.z - p->a.z + 1));
	if (size == 8)
	{
		initCuda_size = make_cudaExtent(513, 513, 512);
	}
	//cudaExtent initCuda_size = make_cudaExtent(32, 32, 32);
	cudakernel.initCuda(d_volume, initCuda_size);
	//cudaFree(d_volume);

	float* d_output_rgb;	
	float* d_output_alpha;
	cudaMalloc((void**)&d_output_rgb, image_width * image_height * 3 * sizeof(float));
	cudaMalloc((void**)&d_output_alpha, image_width * image_height * 1 * sizeof(float));

	cudaMemset(d_output_rgb, 0, image_width * image_height * 3 * sizeof(float));
	cudaMemset(d_output_alpha, 0, image_width * image_height * 1 * sizeof(float));

	// 记录渲染开始时间（MPI 和 CUDA）
    double start_time = MPI_Wtime();
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventRecord(cuda_start, 0);
	
	cudakernel.render_kernel(gridSize, blockSize, d_output_rgb, d_output_alpha, 
		image_width, image_height,
		p->camera->from, p->camera->to, p->camera->u, p->camera->v, dz,
		boxMin, boxMax, big_boxMin, big_boxMax, compensation,
		density, brightness, transferOffset, transferScale,
		d_minMaxXY);
	
	cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);
    float cuda_time = 0.0f;
    cudaEventElapsedTime(&cuda_time, cuda_start, cuda_stop);
	// 暂停计时，准备进行数据拷贝和处理
    double pause_time = MPI_Wtime();
    
	// 汇总所有进程的 CUDA 时间到主进程
    float total_cuda_time_all_processes = 0.0f;
    MPI_Reduce(&cuda_time, &total_cuda_time_all_processes, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (p->Processor_ID == 0)
    {
        std::cout << "Total CUDA execution time (across all processes): " << total_cuda_time_all_processes << " ms" << std::endl;
    }


	cudaMemcpy(h_minMaxXY, d_minMaxXY, 4 * sizeof(uint), cudaMemcpyDeviceToHost);

	// h_minMaxXY now contains the min and max x, y coordinates of the pixels that intersect the AABB
	printf("Min x: %d, Min y: %d\n", h_minMaxXY[0], h_minMaxXY[1]);
	printf("Max x: %d, Max y: %d\n", h_minMaxXY[2], h_minMaxXY[3]);
	std::cout << "[range]:: PID [ " << p->Processor_ID << " ] [ MinX, MinY ] , [ MaxX, MaxY ] [ " << h_minMaxXY[0] << " , " << h_minMaxXY[1] << " ] [ " << h_minMaxXY[2] << " , " << h_minMaxXY[3] << " ]" << std::endl;
	// Clean up
	cudaFree(d_minMaxXY);

	// Check for kernel errors
	getLastCudaError("Kernel failed");

	// Allocate host memory for the image
	//float* h_output_rgb = new float[image_width * image_height * 4];

	// Copy data back to host
	//cudaMemcpy(h_output, d_output, image_width * image_height * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout << "[CPU]:: PID [ " << p->Processor_ID << " ] copy back to host" << std::endl;
	//MPI_Barrier(MPI_COMM_WORLD);

	float* h_rgb = new float[image_width * image_height * 3];
	float* h_alpha = new float[image_width * image_height * 1];
	cudaMemcpy(h_rgb, d_output_rgb, image_width * image_height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_alpha, d_output_alpha, image_width * image_height * 1 * sizeof(float), cudaMemcpyDeviceToHost);
	//Utils::splitRGBA(h_output, image_width, image_height, h_rgb, h_alpha);

	// 将RGB和ALPHA数据合并
	unsigned char* h_img_uc2 = new unsigned char[image_width * image_height * 4];
	for (int i = 0; i < image_width * image_height; ++i) {
		h_img_uc2[i * 4 + 0] = static_cast<unsigned char>(h_rgb[i * 3 + 0] * 255.0f);  // R
		h_img_uc2[i * 4 + 1] = static_cast<unsigned char>(h_rgb[i * 3 + 1] * 255.0f);  // G
		h_img_uc2[i * 4 + 2] = static_cast<unsigned char>(h_rgb[i * 3 + 2] * 255.0f);  // B
		h_img_uc2[i * 4 + 3] = static_cast<unsigned char>(h_alpha[i] * 255.0f);        // A
	}

	char outputFilename[128];
	char outputFilnamebin[128];
	if (p->Processor_Size == 1)
	{
		sprintf(outputFilename, "ground_truth.png");
		// 生成GT的binary
		float* h_output = nullptr;
		Utils::combineRGBA(h_rgb, h_alpha, image_width, image_height, h_output);
		Utils::saveArrayAsBinary("ground_truth.bin", h_output, image_width, image_height);
		delete[] h_output;
		h_output = nullptr;
	}
	else
	{
		//sprintf(outputFilnamebin, "output_rank_%d.bin", rank);
		//saveArrayAsBinaryRGB(outputFilnamebin, h_output, h_minMaxXY[0], h_minMaxXY[1], h_minMaxXY[2], h_minMaxXY[3], image_width, image_height);

		//int newWidth = h_minMaxXY[2] - h_minMaxXY[0] + 1;
		//int newHeight = h_minMaxXY[3] - h_minMaxXY[1] + 1;

		////std::cout << "[TEST]:: " << p->Processor_ID << " " << newWidth << " " << newHeight << std::endl;


		//float* ffoutputData = new float[newWidth * newHeight * 3];
		//// 读取二进制文件并重新组合数据
		//loadBinaryFile(outputFilnamebin, newWidth, newHeight, ffoutputData);

		//// 保存为PNG图像
		//sprintf(outputFilename, "output_rank_%d_rgb.png", rank);
		//saveImageAsPNG(outputFilename, ffoutputData, newWidth, newHeight);

		sprintf(outputFilename, "output_rank_%d.png", rank);
	}




	// Save image as PNG
	if (stbi_write_png(outputFilename, image_width, image_height, 4, h_img_uc2, image_width * 4))
		std::cout << "Finished " << outputFilename << " [ " << image_width << " X " << image_height << " ]" << std::endl;
	else
		std::cout << "[ERROR]:: NO Finished " << outputFilename << " [ " << image_width << " X " << image_height << " ]" << std::endl;

	
	// 恢复计时，继续统计剩余时间
    double resume_time = MPI_Wtime();
    start_time += (resume_time - pause_time);  // 扣除暂停期间的时间

	p->binarySwap_Alpha(h_alpha);
	float global_error_bounded = 1E-2;
	int range_w = static_cast<int>(h_minMaxXY[2] - h_minMaxXY[0] + 1);
	int range_h = static_cast<int>(h_minMaxXY[3] - h_minMaxXY[1] + 1);
	float* error_array = new float[range_w * range_h];
	if (p->Processor_Size == 2 || p->Processor_Size == 4)
	{
		for (auto hight_idx = h_minMaxXY[1]; hight_idx <= h_minMaxXY[3]; ++hight_idx)
		{
			for (auto width_idx = h_minMaxXY[0]; width_idx <= h_minMaxXY[2]; ++width_idx)
			{
				auto p_alpha = p->obr_alpha[hight_idx * image_width + width_idx];
				auto tmp_error = global_error_bounded / (1 + p_alpha);
				error_array[(hight_idx - h_minMaxXY[1]) * range_w + (width_idx - h_minMaxXY[0])] = tmp_error;
			}
		}
	
		// 找到error_array中的最大值
		float* max_error_ptr = std::min_element(error_array, error_array + range_w * range_h);
		float max_error = *max_error_ptr;
		std::cout << "[ERROR_BOUNDED]:: PID [ " << p->Processor_ID << " ] max_error " << max_error << std::endl;
	}

	p->binarySwap_RGB(h_rgb, false);

	// 记录最终结束时间
    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    // 汇总所有进程的总时间到主进程
    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	// 在主进程上输出总时间
    if (p->Processor_ID == 0)
    {
        std::cout << "Total execution time from render_kernel (excluding copy and split) (across all processes): " 
		<< (max_time * 1000) << " ms." << std::endl;
    }


	//p->binarySwap(h_output);

	// std::cout << "[Processor::binarySwap_RGB]:: PID " << p->Processor_ID 
    //           << " Total Sent Bytes: " << p->totalSentBytes 
    //           << " Total Received Bytes: " << p->totalReceivedBytes << std::endl;

	size_t totalSentBytesAllProcesses = 0;
    size_t totalReceivedBytesAllProcesses = 0;

    MPI_Reduce(&p->totalSentBytes, &totalSentBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&p->totalReceivedBytes, &totalReceivedBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // 在主进程上输出总通信量
    if (p->Processor_ID == 0)
    {
        std::cout << "[Processor::binarySwap_RGB]:: Total Sent Bytes: " << totalSentBytesAllProcesses 
                  << " Total Received Bytes: " << totalReceivedBytesAllProcesses << std::endl;
    }


	delete[] error_array;
	error_array = nullptr;
	if (p->Processor_ID == 0 && p->kdTree->depth != 0)
	{
		float* obr_rgba = nullptr;
		Utils::combineRGBA(p->obr_rgb, p->obr_alpha, image_width, image_height, obr_rgba);
		// Convert float image to unsigned char image
		unsigned char* h_img_uc = new unsigned char[image_width * image_height * 4];
		for (int i = 0; i < image_width * image_height * 4; ++i) {
			h_img_uc[i] = static_cast<unsigned char>(obr_rgba[i] * 255.0f);
		}
		sprintf(outputFilename, "output_%d.png", size);
		// Save image as PNG
		stbi_write_png(outputFilename, image_width, image_height, 4, h_img_uc, image_width * 4);

		// 生成output 的二进制
		sprintf(outputFilename, "output_%d.bin", size);
		Utils::saveArrayAsBinary(outputFilename, obr_rgba, image_width, image_height);
		std::cout << "finished FIIIIIIIIIIII\n";
	}





	delete[] h_alpha;
	delete[] h_rgb;

	cudaFree(d_output_rgb);
	cudaFree(d_output_alpha);

	

	if (p->Processor_Size == 1)
		system("pause");
	if (p) { delete p; p = NULL; }


	
	err = MPI_Finalize();
	if (err != MPI_SUCCESS)
	{
		std::cerr << "[ERROR]::MPI_Finalize:: MPI Finalize Error." << std::endl;
		MPI_Abort(MPI_COMM_WORLD, err);
	}
	return 0;
}

// #include <mpi.h>
// #include <cuda_runtime.h>
// #include <iostream>

// int main(int argc, char *argv[]) {
//     MPI_Init(&argc, &argv);

//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     int gpu_count;
//     cudaGetDeviceCount(&gpu_count);
//     std::cout << "Rank " << rank << " sees " << gpu_count << " GPU(s)" << std::endl;
// 	int gpu_id = rank % 4; // 假设每个节点有 4 个 GPU
// 	cudaSetDevice(gpu_id);

// 	int device_id;
// 	cudaGetDevice(&device_id);
// 	std::cout << "Process " << rank << " is bound to GPU " << device_id << std::endl;
//     MPI_Finalize();
//     return 0;
// }
