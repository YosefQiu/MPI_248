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

#define SAVE_CUDA_RESULT 1



dim3 blockSize(16, 16);
dim3 gridSize;

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

int err;
Processor* p = nullptr;

std::string volumeFilename;
cudaExtent volumeTotalSize;
unsigned int image_width;
unsigned int image_height;
int xdim, ydim, zdim;
float dx, dy, dz;
bool usecomress = false;
bool useeffarea = false;
bool display = false;
std::string iteration_str;

std::string total_time_filename = "./totalTime";
std::string time_filename = "./time";

int main(int argc, char* argv[])
{
	err = MPI_Init(&argc, &argv);
	if (err != MPI_SUCCESS)
	{
		std::cerr << "[ERROR]::MPI_Init:: MPI Init Error." << std::endl;
		MPI_Abort(MPI_COMM_WORLD, err);
	}
	
	if (argc < 12)
	{
		std::cerr << "[ERROR]::main:: Usage: " << argv[0] << " <volumeFilename> <xdim> <ydim> <zdim> <image_width> <image_height> <cam_dx> <cam_dy> <cam_dz> <usecompress> <useeffarea> <iteration_str>" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	else
	{
		Utils::parseArguments(argc, argv, volumeFilename, xdim, ydim, zdim, image_width, image_height, dx, dy, dz, usecomress, useeffarea, iteration_str);
		volumeTotalSize = make_cudaExtent(xdim, ydim, zdim);
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
    int gpu_id = p->Processor_ID % gpu_count;
    cudaSetDevice(gpu_id);

	if(usecomress)
	{
		total_time_filename = "./totalTime_compress" + std::to_string(size) + ".txt";
		time_filename = "./time_compress" + std::to_string(size) + ".txt";
	}
	else
	{
		total_time_filename = "./totalTime_NOcompress" + std::to_string(size) + ".txt";
		time_filename = "./time_NOcompress" + std::to_string(size) + ".txt";
	}
	p->save_time_file_path = time_filename;
#pragma region GPUInfoPrint
    // int device_id;
    // cudaGetDevice(&device_id);
	// std::cout << "Process " << p->Processor_ID 
	// 	<< " is using GPU " << device_id << " / " << gpu_count 
	// 	<< " on " << hostname << std::endl;
	// std::cout << "Process " << p->Processor_ID << " on " << hostname << std::endl;
#pragma endregion

	p->initScreen(image_width, image_height);
	p->initRayCaster(p->camera_plane_x, p->camera_plane_y);
	
	if (p->Processor_ID == 0)
		p->init_master(volumeFilename, p->a, p->b, volumeTotalSize);
	else
		p->init_node(p->a, p->b, p->Processor_ID);

	p->initKDTree();
	p->initImage(image_width, image_height);
	p->initData(volumeFilename.c_str());
	std::cout << "doned initdata\n";
	p->initOpti(); // cam_dx = cam_dy = 0.0
	p->setCameraProperty(dx, dy, dz);

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

	// 初始化CUDA 用来做数据采样相关 //TODO?
	cudaExtent initCuda_size = make_cudaExtent((p->data_b.x - p->data_a.x + 1), (p->data_b.y - p->data_a.y + 1), (p->data_b.z - p->data_a.z));
	std::cout << "[init cuda size]:: PID [ " << p->Processor_ID << " ] [ " 
		<< initCuda_size.width << ", " 
		<< initCuda_size.height << " , " 
		<< initCuda_size.depth << "]" << std::endl;
	cudakernel.initCuda(d_volume, initCuda_size);

	// 分配设备内存-输出
	float* d_output_rgb;	
	float* d_output_alpha;
	cudaMalloc((void**)&d_output_rgb, image_width * image_height * 3 * sizeof(float));
	cudaMalloc((void**)&d_output_alpha, image_width * image_height * 1 * sizeof(float));
	cudaMemset(d_output_rgb, 0, image_width * image_height * 3 * sizeof(float));
	cudaMemset(d_output_alpha, 0, image_width * image_height * 1 * sizeof(float));

	cudaEvent_t cuda_start, cuda_stop;
    float cuda_elapsedTime;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

	double end_to_end_start, end_to_end_end, end_to_end_time;

	MPI_Barrier(MPI_COMM_WORLD);
	end_to_end_start = MPI_Wtime();
	
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
    cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);

    // 记录CUDA 运行时间
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);
	if (p->Processor_ID == 0)
	{
		Utils::recordCudaRenderTime(time_filename.c_str(), "cuda_rende_time:", iteration_str + ":", cuda_elapsedTime);
	}
	

	// 打印有效数据范围
	cudaMemcpy(h_minMaxXY, d_minMaxXY, 4 * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaFree(d_minMaxXY);
#if SAVE_CUDA_RESULT == true
	std::cout << "[range]:: PID [ " << p->Processor_ID << " ] [ MinX, MinY ] , [ MaxX, MaxY ] [ " 
		<< h_minMaxXY[0] << " , " << h_minMaxXY[1] << " ] [ " 
		<< h_minMaxXY[2] << " , " << h_minMaxXY[3] << " ]" << std::endl;
#endif
	
	getLastCudaError("Kernel failed");

	// 拷贝RGB/ALPHA数据到主机
	float* h_rgb = new float[image_width * image_height * 3];
	float* h_alpha = new float[image_width * image_height * 1];
	cudaMemcpy(h_rgb, d_output_rgb, image_width * image_height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_alpha, d_output_alpha, image_width * image_height * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#if SAVE_CUDA_RESULT == true
	// 一个检查：检查CUDA内核的输出结果（PNG 和 BIN）
	unsigned char* h_img_uc2 = new unsigned char[image_width * image_height * 4];
	float* tmp_rgb = new float[image_width * image_height * 3];
	Utils::convertRRRGGGBBBtoRGB(h_rgb, image_width * image_height * 3, tmp_rgb);
	for (int i = 0; i < image_width * image_height; ++i) 
	{
		h_img_uc2[i * 4 + 0] = static_cast<unsigned char>(tmp_rgb[i * 3 + 0] * 255.0f);  // R
		h_img_uc2[i * 4 + 1] = static_cast<unsigned char>(tmp_rgb[i * 3 + 1] * 255.0f);  // G
		h_img_uc2[i * 4 + 2] = static_cast<unsigned char>(tmp_rgb[i * 3 + 2] * 255.0f);  // B
		h_img_uc2[i * 4 + 3] = static_cast<unsigned char>(h_alpha[i] * 255.0f);        	 // A
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
	delete[] tmp_rgb; tmp_rgb = nullptr;
#endif

	if(usecomress == true)
	{
		double alpha_start_time, alpha_end_time, alpha_elapsed_time;
		double rgb_start_time, rgb_end_time, rgb_elapsed_time;
		cudaDeviceSynchronize();

		MPI_Barrier(MPI_COMM_WORLD);
		alpha_start_time = MPI_Wtime();
		p->binarySwap_Alpha_GPU(d_output_alpha);
		MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程都完成操作
		alpha_end_time = MPI_Wtime();
		alpha_elapsed_time = alpha_end_time - alpha_start_time;
		alpha_elapsed_time *= 1000.0f;
		if (p->Processor_ID == 0)
			Utils::recordCudaRenderTime(time_filename.c_str(), "binarySwap_Alpha Time:", iteration_str + ":", alpha_elapsed_time);
		
		//TODO 扣掉这部分
		p->AlphaGathering_CPU();
		//p->binarySwap_Alpha(h_alpha);
		float global_error_bounded = 1E-2;
		int range_w = static_cast<int>(h_minMaxXY[2] - h_minMaxXY[0] + 1);
		int range_h = static_cast<int>(h_minMaxXY[3] - h_minMaxXY[1] + 1);
		float* error_array = new float[range_w * range_h];
		if (p->Processor_Size == 2 || p->Processor_Size == 4 || p->Processor_Size == 8 || p->Processor_Size == 16)
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
			// std::cout << "[ERROR_BOUNDED]:: PID [ " << p->Processor_ID << " ] max_error " << max_error << std::endl;
		}



		MPI_Barrier(MPI_COMM_WORLD); // 同步所有进程
		rgb_start_time = MPI_Wtime();
		//p->binarySwap_RGB(h_rgb, (int)h_minMaxXY[0], (int)h_minMaxXY[1], (int)h_minMaxXY[2], (int)h_minMaxXY[3], usecomress, useeffarea);
		p->binarySwap_RGB_GPU(d_output_rgb, (int)h_minMaxXY[0], (int)h_minMaxXY[1], (int)h_minMaxXY[2], (int)h_minMaxXY[3], usecomress, useeffarea);
		MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程都完成操作
		rgb_end_time = MPI_Wtime();
		rgb_elapsed_time = rgb_end_time - rgb_start_time;
		rgb_elapsed_time *= 1000.0f;
		if (p->Processor_ID == 0)
			Utils::recordCudaRenderTime(time_filename.c_str(), "binarySwap_RGB Time:", iteration_str + ":", rgb_elapsed_time);
		
		delete[] error_array;
		error_array = nullptr;
	}
	else if(usecomress == false)
	{
		cudaDeviceSynchronize();
		double start_time, end_time, elapsed_time;
		MPI_Barrier(MPI_COMM_WORLD);
		start_time = MPI_Wtime(); 
		// p->binarySwap(h_rgb, h_alpha);
		p->binarySwap_GPU(d_output_rgb, d_output_alpha);
		MPI_Barrier(MPI_COMM_WORLD); 
		end_time = MPI_Wtime(); 
		elapsed_time = end_time - start_time;
		elapsed_time *= 1000.0f;
		if (p->Processor_ID == 0)
		{
			Utils::recordCudaRenderTime(time_filename.c_str(),  "binarySwap Time:", iteration_str + ":", elapsed_time);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	end_to_end_end = MPI_Wtime();
	end_to_end_time = end_to_end_end - end_to_end_start;
	end_to_end_time *= 1000.0f;
		
	if(p->Processor_ID == 0)
	{
		Utils::recordTotalTime(total_time_filename.c_str(), end_to_end_time, iteration_str);
	}
	
#pragma region CalcBytes
	// 计算通信量
	size_t totalSentBytesAllProcesses = 0;
	size_t alphaSentBytesAllProcesses = 0;
	size_t rgbSentBytesAllProcesses = 0;
	if(usecomress == true)
	{
		MPI_Reduce(&p->alpha_totalSentBytes, &alphaSentBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&p->rgb_totalSentBytes, &rgbSentBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		if (p->Processor_ID == 0)
		{
			Utils::recordByte(time_filename.c_str(), "alphaSentBytes:", iteration_str + ":", alphaSentBytesAllProcesses);	
			Utils::recordByte(time_filename.c_str(), "rgbSentBytes:", iteration_str + ":", rgbSentBytesAllProcesses);	
		}
	}
	else
	{
		MPI_Reduce(&p->rgba_totalSentBytes, &totalSentBytesAllProcesses, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		if (p->Processor_ID == 0)
			Utils::recordByte(time_filename.c_str(), "totalSentBytes:", iteration_str + ":", totalSentBytesAllProcesses);
		
	}
#pragma endregion CalcBytes

   
  
	if(p->Processor_ID == 0)
	{
		Utils::AddEndline(time_filename.c_str());
	}
	

#if SAVE_CUDA_RESULT == true
	// 保存最终结果
	if (p->Processor_ID == 0 && p->kdTree->depth != 0 && usecomress == true)
	{
		// 结果在GPU上
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
	else if((p->Processor_ID == 0 && p->kdTree->depth != 0 && usecomress == false))
	{
		// 结果在CPU上
		unsigned char* h_img_uc = new unsigned char[image_width * image_height * 4];
		for (int i = 0; i < image_width * image_height * 4; ++i) {
			h_img_uc[i] = static_cast<unsigned char>(p->obr[i] * 255.0f);
		}
		sprintf(outputFilename, "output_%d.png", size);
		stbi_write_png(outputFilename, image_width, image_height, 4, h_img_uc, image_width * 4);

		// 生成output 的二进制
		sprintf(outputFilename, "output_%d.bin", size);
		Utils::saveArrayAsBinary(outputFilename, p->obr, image_width, image_height);
		std::cout << "finished FIIIIIIIIIIII\n";
		delete[] h_img_uc; h_img_uc = nullptr;
	}
#endif
	
#pragma region CleanUp
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
#pragma endregion
	return 0;
}

