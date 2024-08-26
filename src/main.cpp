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

	findCudaDevice(argc, (const char**)argv);
	CudaKernel cudakernel;

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

	float* d_output;
	cudaMalloc((void**)&d_output, image_width * image_height * 4 * sizeof(float));

	cudaMemset(d_output, 0, image_width * image_height * 4 * sizeof(float));

	
	cudakernel.render_kernel(gridSize, blockSize, d_output, image_width, image_height,
		p->camera->from, p->camera->to, p->camera->u, p->camera->v, dz,
		boxMin, boxMax, big_boxMin, big_boxMax, compensation,
		density, brightness, transferOffset, transferScale,
		d_minMaxXY);

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
	float* h_output = new float[image_width * image_height * 4];

	// Copy data back to host
	cudaMemcpy(h_output, d_output, image_width * image_height * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "[CPU]:: PID [ " << p->Processor_ID << " ] copy back to host" << std::endl;
	//MPI_Barrier(MPI_COMM_WORLD);

	float* h_rgb = nullptr;
	float* h_alpha = nullptr;
	Utils::splitRGBA(h_output, image_width, image_height, h_rgb, h_alpha);

	// Convert float image to unsigned char image
	unsigned char* h_img_uc2 = new unsigned char[image_width * image_height * 4];
	for (int i = 0; i < image_width * image_height * 4; ++i) {
		h_img_uc2[i] = static_cast<unsigned char>(h_output[i] * 255.0f);
	}

	char outputFilename[128];
	char outputFilnamebin[128];
	if (p->Processor_Size == 1)
	{
		sprintf(outputFilename, "ground_truth.png");
		// 生成GT的binary
		Utils::saveArrayAsBinary("ground_truth.bin", h_output, image_width, image_height);
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

	p->binarySwap_Alpha(h_alpha);
	p->binarySwap_RGB(h_rgb);
	//p->binarySwap(h_output);

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





	delete[] h_output;

	cudaFree(d_output);

	

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

