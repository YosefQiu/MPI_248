#include "ggl.h"

#include "mpi.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "FileManager.h"
#include "kernel.cuh"

#include "stb_image.h"
#include "stb_image_write.h"
#include "Utils.h"
#include "Processor.cuh"

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
bool usecomress = false;
bool useeffarea = false;


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
		if (argc >= 11)
		{
			dx = std::atof(argv[7]);
			dy = std::atof(argv[8]);
			dz = std::atof(argv[9]);
			usecomress = std::atoi(argv[10]);
			useeffarea = std::atoi(argv[11]);
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
	// int gpu_count;
    // cudaGetDeviceCount(&gpu_count);
    // // 映射每个进程到特定的 GPU（在本地节点上）
    // int gpu_id = p->Processor_ID % gpu_count;
    // cudaSetDevice(gpu_id);

    // int device_id;
    // cudaGetDevice(&device_id);
	// std::cout << "Process " << p->Processor_ID 
	// 	<< " is using GPU " << device_id << " / " << gpu_count 
	// 	<< " on " << hostname << std::endl;

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

	// 打印包围盒
	float3 big_boxMin, big_boxMax;
	big_boxMin.x = -1.0f * p->whole_data_len.x / 2.0; big_boxMin.y = -1.0f * p->whole_data_len.y / 2.0; big_boxMin.z = -1.0f * p->whole_data_len.z / 2.0;
	big_boxMax = big_boxMin + p->whole_data_len + make_float3(-1, -1, -1);
	float3 boxMin = p->bMin; float3 boxMax = p->bMax;
	boxMax.x = p->bMax.x; boxMax.y = p->bMax.y; boxMax.z = p->bMax.z;
	float3 compensation = p->data_compensation;
	printf("Process %d, [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", rank, boxMin.x, boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z, compensation.x, compensation.y, compensation.z);
	
	// 分配设备内存-数据
	size_t local_volume_size = (p->data_b.x - p->data_a.x + 1) * (p->data_b.y - p->data_a.y + 1) * (p->data_b.z - p->data_a.z + 1) * sizeof(VolumeType);
	unsigned char* d_volume;
	cudaMalloc(&d_volume, local_volume_size);
	cudaMemcpy(d_volume, p->data, local_volume_size, cudaMemcpyHostToDevice);

	// 分配设备内存-最小最大坐标
	uint h_minMaxXY[4] = { image_width, image_height, 0, 0 };
	uint* d_minMaxXY;
	cudaMalloc(&d_minMaxXY, 4 * sizeof(uint));
	cudaMemcpy(d_minMaxXY, h_minMaxXY, 4 * sizeof(uint), cudaMemcpyHostToDevice);

	// 初始化CUDA 用来做数据采样相关
	cudaExtent initCuda_size = make_cudaExtent((p->data_b.x - p->data_a.x + 1), (p->data_b.y - p->data_a.y + 1), (p->data_b.z - p->data_a.z + 1));
	std::cout << "[init cuda size]:: PID [ " << p->Processor_ID << " ] [ " 
		<< p->data_b.x - p->data_a.x + 1 << ", " 
		<< p->data_b.y - p->data_a.y + 1 << " , " 
		<< p->data_b.z - p->data_a.z + 1 << "]" << std::endl;
	cudakernel.initCuda(d_volume, initCuda_size);

	// 分配设备内存-输出
	float* d_output_rgb;	
	float* d_output_alpha;
	cudaMalloc((void**)&d_output_rgb, image_width * image_height * 3 * sizeof(float));
	cudaMalloc((void**)&d_output_alpha, image_width * image_height * 1 * sizeof(float));
	cudaMemset(d_output_rgb, 0, image_width * image_height * 3 * sizeof(float));
	cudaMemset(d_output_alpha, 0, image_width * image_height * 1 * sizeof(float));

	cudaEvent_t cuda_start, cuda_stop;
    float elapsedTime;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    // Start recording time
    cudaEventRecord(cuda_start, 0);
	cudakernel.render_kernel(gridSize, blockSize, d_output_rgb, d_output_alpha, 
		image_width, image_height,
		p->camera->from, p->camera->to, p->camera->u, p->camera->v, dz,
		boxMin, boxMax, big_boxMin, big_boxMax, compensation,
		density, brightness, transferOffset, transferScale,
		d_minMaxXY);
    cudaDeviceSynchronize();
	cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);
    cudaEventElapsedTime(&elapsedTime, cuda_start, cuda_stop);

    // Cleanup CUDA events
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);
	Utils::recordCudaRenderTime("./cuda_render_time.txt", p->Processor_Size, p->Processor_ID, elapsedTime);

	// 打印有效数据范围
	cudaMemcpy(h_minMaxXY, d_minMaxXY, 4 * sizeof(uint), cudaMemcpyDeviceToHost);
	// std::cout << "[range]:: PID [ " << p->Processor_ID << " ] [ MinX, MinY ] , [ MaxX, MaxY ] [ " 
	// 	<< h_minMaxXY[0] << " , " << h_minMaxXY[1] << " ] [ " 
	// 	<< h_minMaxXY[2] << " , " << h_minMaxXY[3] << " ]" << std::endl;
	cudaFree(d_minMaxXY);

	// Check for kernel errors
	getLastCudaError("Kernel failed");


	
	// 拷贝RGB/ALPHA数据到主机
	float* h_rgb = new float[image_width * image_height * 3];
	float* h_alpha = new float[image_width * image_height * 1];
	cudaMemcpy(h_rgb, d_output_rgb, image_width * image_height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_alpha, d_output_alpha, image_width * image_height * 1 * sizeof(float), cudaMemcpyDeviceToHost);

	// 一个检查：检查CUDA内核的输出结果（PNG 和 BIN）
	unsigned char* h_img_uc2 = new unsigned char[image_width * image_height * 4];
	float* tmp_rgb = new float[image_width * image_height * 3];
	Utils::convertRRRGGGBBBtoRGB(h_rgb, image_width * image_height * 3, tmp_rgb);
	for (int i = 0; i < image_width * image_height; ++i) {
		h_img_uc2[i * 4 + 0] = static_cast<unsigned char>(tmp_rgb[i * 3 + 0] * 255.0f);  // R
		h_img_uc2[i * 4 + 1] = static_cast<unsigned char>(tmp_rgb[i * 3 + 1] * 255.0f);  // G
		h_img_uc2[i * 4 + 2] = static_cast<unsigned char>(tmp_rgb[i * 3 + 2] * 255.0f);  // B
		h_img_uc2[i * 4 + 3] = static_cast<unsigned char>(h_alpha[i] * 255.0f);        // A
	}
	char outputFilename[128];
	char outputFilnamebin[128];
	if (p->Processor_Size == 1)
	{
		sprintf(outputFilename, "ground_truth.png");
		// 生成GT的binary
		float* h_output = nullptr;
		Utils::combineRGBA(tmp_rgb, h_alpha, image_width, image_height, h_output);
		Utils::saveArrayAsBinary("ground_truth.bin", h_output, image_width, image_height);
		delete[] h_output;
		h_output = nullptr;
	}
	else
	{
		//	保存每个进程的输出
		sprintf(outputFilename, "output_rank_%d.png", rank);
	}
	if (stbi_write_png(outputFilename, image_width, image_height, 4, h_img_uc2, image_width * 4))
		std::cout << "Finished " << outputFilename << " [ " << image_width << " X " << image_height << " ]" << std::endl;
	else
		std::cout << "[ERROR]:: NO Finished " << outputFilename << " [ " << image_width << " X " << image_height << " ]" << std::endl;
	delete[] h_img_uc2; h_img_uc2 = nullptr;
	// delete[] tmp_rgb; tmp_rgb = nullptr;


	// 同步所有进程，确保每个进程都在同一个时刻开始计时
	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime(); // 开始计时
	//p->binarySwap_Alpha_GPU(d_output_alpha);
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

	// // TODO test d_alpha_valus_ u to cpu p->alpha_value_u
	// int value_size = p->kdTree->depth * p->obr_x * p->obr_y;
	// float* tmp_u = new float[value_size];
	// cudaMemcpy(tmp_u, p->d_alpha_values_u, value_size * sizeof(float), cudaMemcpyDeviceToHost);
	// int idx = 0;
	// for (int u = 0; u < p->kdTree->depth; ++u) 
	// {
	// 	for (int y = 0; y < p->obr_y; ++y) 
	// 	{
	// 		 for (int x = 0; x < p->obr_x; ++x) 
	// 		{
	// 			p->alpha_values_u[u][y][x] = tmp_u[idx++];  // 从 tmp_u 中提取值并赋给 alpha_values_u[u][y][x]
	// 		}
	// 	}
	// }

	MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程都完成操作
	double end_time = MPI_Wtime(); // 结束计时
	double elapsed_time = end_time - start_time;
	elapsed_time *= 1000.0;
	Utils::recordCudaRenderTime("./alpha_change_time.txt", p->Processor_Size, p->Processor_ID, elapsed_time);

	
	MPI_Barrier(MPI_COMM_WORLD); // 同步所有进程
	double start_time_swap = MPI_Wtime(); // 记录 binarySwap_RGB 开始时间
	p->binarySwap_RGB(h_rgb, (int)h_minMaxXY[0], (int)h_minMaxXY[1], (int)h_minMaxXY[2], (int)h_minMaxXY[3], usecomress, useeffarea);
	//p->binarySwap_RGB_GPU(d_output_rgb, (int)h_minMaxXY[0], (int)h_minMaxXY[1], (int)h_minMaxXY[2], (int)h_minMaxXY[3], usecomress);
	MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程都完成操作
	double end_time_swap = MPI_Wtime(); // 记录 binarySwap_RGB 结束时间
	double elapsed_time_swap = (end_time_swap - start_time_swap) * 1000.0; // 转换为毫秒
	Utils::recordCudaRenderTime("./binarySwap_RGB_time.txt", p->Processor_Size, p->Processor_ID, elapsed_time_swap);
	
	// 计算通信量
	size_t totalSentBytesAllProcesses = 0;
    size_t totalReceivedBytesAllProcesses = 0;
    MPI_Reduce(&p->totalSentBytes, &totalSentBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&p->totalReceivedBytes, &totalReceivedBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	size_t alpha_totalSentBytesAllProcesses = 0;
    size_t alpha_totalReceivedBytesAllProcesses = 0;
    MPI_Reduce(&p->alpha_totalSentBytes, &alpha_totalSentBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&p->alpha_totalReceivedBytes, &alpha_totalReceivedBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (p->Processor_ID == 0)
    {
        std::cout << "[Processor::binarySwap_RGB]:: Total RGB Bytes: " << totalSentBytesAllProcesses 
                  << " Total ALPHA Bytes: " << alpha_totalReceivedBytesAllProcesses << std::endl;			
    }
  
	//delete[] error_array;
	//error_array = nullptr;


	// 保存最终结果
	if (p->Processor_ID == 0 && p->kdTree->depth != 0)
	{
		//float* result_alpha = new float[image_width * image_height];
		//cudaMemcpy(result_alpha, p->d_obr_alpha, image_width * image_height * sizeof(float), cudaMemcpyDeviceToHost);

		float* obr_rgba = nullptr;
		float* reslut_rgb = new float[image_width * image_height * 3];
		Utils::convertRRRGGGBBBtoRGB(p->obr_rgb, image_width * image_height * 3, reslut_rgb);
		Utils::combineRGBA(reslut_rgb, p->obr_alpha, image_width, image_height, obr_rgba);
		unsigned char* h_img_uc = new unsigned char[image_width * image_height * 4];
		for (int i = 0; i < image_width * image_height * 4; ++i) {
			h_img_uc[i] = static_cast<unsigned char>(obr_rgba[i] * 255.0f);
		}
		sprintf(outputFilename, "output_%d.png", size);
		stbi_write_png(outputFilename, image_width, image_height, 4, h_img_uc, image_width * 4);

		// 生成output 的二进制
		sprintf(outputFilename, "output_%d.bin", size);
		Utils::saveArrayAsBinary(outputFilename, obr_rgba, image_width, image_height);
		std::cout << "finished FIIIIIIIIIIII\n";
		delete[] h_img_uc; h_img_uc = nullptr;
		delete[] obr_rgba; obr_rgba = nullptr;

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

