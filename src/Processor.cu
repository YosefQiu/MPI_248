﻿#include "Processor.cuh"
#include "FileManager.h"
#include "Utils.h"

#define GATHERING_IMAGE 1
#define USE_GATHER 1

__global__ void fillSendBuffer(float* d_img_alpha, float* d_alpha_sbuffer, Point2Di sa, Point2Di sb, int obr_x) 
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + sa.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + sa.y;
    if (x >= sa.x && x <= sb.x && y >= sa.y && y <= sb.y) 
	{
        int img_index = y * obr_x + x; // 从原图中获取索引
        int buffer_index = (y - sa.y) * (sb.x - sa.x + 1) + (x - sa.x); // 在缓冲区中的索引
        d_alpha_sbuffer[buffer_index] = d_img_alpha[img_index]; // 将值存储到缓冲区中
    }
}

__global__ void loadColorBufferRRGGBBKernel(float* d_obr_rgb, float* d_rgb_sbuffer, Point2Di sa, Point2Di sb, int obr_x, int obr_y)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + sa.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + sa.y;

	if (x >= sa.x && x <= sb.x && y >= sa.y && y <= sb.y)
	{
		int pixelIndex = y * obr_x + x;
		// R, G, B 通道在 d_rgb_sbuffer 中的起始位置
        int totalPixels = (sb.x - sa.x + 1) * (sb.y - sa.y + 1);
        int rOffset_sbuffer = 0;                      // R 通道从 0 开始
        int gOffset_sbuffer = totalPixels;            // G 通道从 totalPixels 开始
        int bOffset_sbuffer = totalPixels * 2;        // B 通道从 2 * totalPixels 开始

		// R, G, B 通道在 d_obr_rgb 中的起始偏移量
        int totalImagePixels = obr_x * obr_y;
        int rOffset_obr = 0;                          // R 通道从 0 开始
        int gOffset_obr = totalImagePixels;           // G 通道从 totalImagePixels 开始
        int bOffset_obr = totalImagePixels * 2;       // B 通道从 2 * totalImagePixels 开始

		// 计算 d_rgb_sbuffer 中的目标位置索引
        int bufferIndex = (y - sa.y) * (sb.x - sa.x + 1) + (x - sa.x);

        // 提取 R, G, B 分量
        float r = d_obr_rgb[rOffset_obr + pixelIndex];
        float g = d_obr_rgb[gOffset_obr + pixelIndex];
        float b = d_obr_rgb[bOffset_obr + pixelIndex];

		// 将 R, G, B 分别存储到 d_rgb_sbuffer 中的分段区域
        d_rgb_sbuffer[rOffset_sbuffer + bufferIndex] = r;
        d_rgb_sbuffer[gOffset_sbuffer + bufferIndex] = g;
        d_rgb_sbuffer[bOffset_sbuffer + bufferIndex] = b;

	}
}

__global__ void updateAlpha(float* d_img_alpha, float* d_alpha_rbuffer, float* d_alpha_values_u, 
						Point2Di ra, Point2Di rb, int obr_x, int obr_y, bool over, int u)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + ra.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ra.y;
    if (x >= ra.x && x <= rb.x && y >= ra.y && y <= rb.y) 
	{
        int img_index = y * obr_x + x;  // 在 d_img_alpha 中的索引
        int buffer_index = (y - ra.y) * (rb.x - ra.x + 1) + (x - ra.x);  // 在接收缓冲区中的索引
        int alpha_values_index = u * (obr_y * obr_x) + y * obr_x + x; // 在 d_alpha_values_u 中的索引

        // 从 d_img_alpha 中读取 Alpha 值
        float a_float = d_img_alpha[img_index];

        if (!over) {
            // 前到后合成
            d_alpha_values_u[alpha_values_index] = d_alpha_rbuffer[buffer_index];
            a_float = d_alpha_rbuffer[buffer_index] + a_float * (1.0f - d_alpha_rbuffer[buffer_index]);
        } else {
            // 后到前合成
            d_alpha_values_u[alpha_values_index] = a_float;
            a_float = a_float + d_alpha_rbuffer[buffer_index] * (1.0f - a_float);
        }

        // 将合成后的 Alpha 值写回 d_img_alpha
        d_img_alpha[img_index] = a_float;
    }
}

__global__ void compositingColorRRGGBBKernel(float* d_obr_rgb, float* d_rgb_rbuffer, float* d_alpha_values_u,
                                             Point2Di ra, Point2Di rb, int obr_x, int obr_y, bool over, int u)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + ra.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ra.y;

    if (x <= rb.x && y <= rb.y)
	{
		// 当前像素索引
        int pixelIndex = y * obr_x + x;

        // R, G, B 通道在 d_obr_rgb 中的偏移量
        int totalImagePixels = obr_x * obr_y;
        int rOffset_obr = 0;
        int gOffset_obr = totalImagePixels;
        int bOffset_obr = totalImagePixels * 2;

        // R, G, B 通道在 d_rgb_rbuffer 中的偏移量
        int totalPixels = (rb.x - ra.x + 1) * (rb.y - ra.y + 1);
        int rOffset_sbuffer = 0;
        int gOffset_sbuffer = totalPixels;
        int bOffset_sbuffer = totalPixels * 2;

		// 计算 d_rgb_rbuffer 中的索引
        int bufferIndex = (y - ra.y) * (rb.x - ra.x + 1) + (x - ra.x);

        // 从 d_obr_rgb 中提取 R, G, B 分量
        float r_float = d_obr_rgb[rOffset_obr + pixelIndex];
        float g_float = d_obr_rgb[gOffset_obr + pixelIndex];
        float b_float = d_obr_rgb[bOffset_obr + pixelIndex];

        // 从 d_rgb_rbuffer 中提取 R, G, B 分量
        float r_float_sbuffer = d_rgb_rbuffer[rOffset_sbuffer + bufferIndex];
        float g_float_sbuffer = d_rgb_rbuffer[gOffset_sbuffer + bufferIndex];
        float b_float_sbuffer = d_rgb_rbuffer[bOffset_sbuffer + bufferIndex];

        // 从 d_alpha_values_u 中提取 Alpha 值
        int alphaIndex = u * (obr_y * obr_x) + pixelIndex;
        float a_float = d_alpha_values_u[alphaIndex];

		// 进行颜色混合
        if (!over) {
            r_float = r_float_sbuffer + r_float * (1.0f - a_float); // Red
            g_float = g_float_sbuffer + g_float * (1.0f - a_float); // Green
            b_float = b_float_sbuffer + b_float * (1.0f - a_float); // Blue
        } else {
            r_float = r_float + r_float_sbuffer * (1.0f - a_float); // Red
            g_float = g_float + g_float_sbuffer * (1.0f - a_float); // Green
            b_float = b_float + b_float_sbuffer * (1.0f - a_float); // Blue
        }

        // 将修改后的 R, G, B 分量写回 d_obr_rgb
        d_obr_rgb[rOffset_obr + pixelIndex] = r_float;
        d_obr_rgb[gOffset_obr + pixelIndex] = g_float;
        d_obr_rgb[bOffset_obr + pixelIndex] = b_float;
	}
}

__global__ void processReceivedAlpha(float* d_img_alpha, float* d_fbuffer, Point2Di fa, Point2Di fb, int obr_x)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + fa.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + fa.y;
	
	if (x >= fa.x && x <= fb.x && y >= fa.y && y <= fb.y) 
	{
		int img_index = y * obr_x + x; // 在 d_img_alpha 中的索引
		int buffer_index = (y - fa.y) * (fb.x - fa.x + 1) + (x - fa.x); // 在接收缓冲区中的索引
		
		// 从接收缓冲区中读取 Alpha 值
		float a_float = d_fbuffer[buffer_index];

		// 将 Alpha 值写回 d_img_alpha
		d_img_alpha[img_index] = a_float;
		
	}
}

__global__ void copyAlphaToBuffer(float* d_img_alpha, float* d_fbuffer, Point2Di fa, Point2Di fb, int obr_x)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + fa.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + fa.y;

    if (x >= fa.x && x <= fb.x && y >= fa.y && y <= fb.y)
    {
        // 计算 img_alpha 中的索引
        int img_index = y * obr_x + x;
        
        // 计算 d_fbuffer 中的索引，从第 4 个位置开始存储
        int buffer_index = (y - fa.y) * (fb.x - fa.x + 1) + (x - fa.x) ;

        // 将 d_img_alpha 中的值拷贝到 d_fbuffer 中
        d_fbuffer[buffer_index] = d_img_alpha[img_index];
    }
}

__global__ void convertRRRGGGBBBtoRGBKernel(const float* src_buffer, float* dst_buffer, size_t num_pixels) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pixels) {
        // 每个线程处理一个像素
        size_t r_index = idx;
        size_t g_index = idx + num_pixels;
        size_t b_index = idx + 2 * num_pixels;

        // 将 RRR GGG BBB 转换为 RGB
        dst_buffer[3 * idx + 0] = src_buffer[r_index];   // R分量
        dst_buffer[3 * idx + 1] = src_buffer[g_index];   // G分量
        dst_buffer[3 * idx + 2] = src_buffer[b_index];   // B分量
    }
}

__global__ void mergeRGBAToBuffer(float* d_imgColor, float* d_imgAlpha, float* d_img, int imgWidth, int imgHeight)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算像素索引
    if (x < imgWidth && y < imgHeight) 
	{
        int idx = y * imgWidth + x; // 当前像素的索引

        // RGB 数据：imgColor 有 3 个浮点数（RGB）每个像素
        d_img[idx * 4 + 0] = d_imgColor[idx * 3 + 0];  // Red
        d_img[idx * 4 + 1] = d_imgColor[idx * 3 + 1];  // Green
        d_img[idx * 4 + 2] = d_imgColor[idx * 3 + 2];  // Blue
        d_img[idx * 4 + 3] = d_imgAlpha[idx];        	// Alpha
    }
}

__global__ void fillSendRGBABuffer(float* d_img, float* d_rgba_sbuffer, Point2Di sa, Point2Di sb, int obr_x)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + sa.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + sa.y;
	if (x >= sa.x && x <= sb.x && y >= sa.y && y <= sb.y) 
	{
		int index = (y - sa.y) * (sb.x - sa.x + 1) + (x - sa.x); 
        int pixelIndex = (y * obr_x + x) * 4; 

        float r = d_img[pixelIndex + 0];
        float g = d_img[pixelIndex + 1];
        float b = d_img[pixelIndex + 2];
        float a = d_img[pixelIndex + 3];

        d_rgba_sbuffer[index * 4 + 0] = r;
        d_rgba_sbuffer[index * 4 + 1] = g;
        d_rgba_sbuffer[index * 4 + 2] = b;
        d_rgba_sbuffer[index * 4 + 3] = a;
	}
	
}

__global__ void compositingRGBAKernel(float* d_img, float* d_rbuffer, Point2Di ra, Point2Di rb, int obr_x, int obr_y, bool over, int u)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + ra.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ra.y;

	if (x <= rb.x && y <= rb.y)
	{
		int pixelIndex = (y * obr_x + x) * 4;
		int buffer_index = ((y - ra.y) * (rb.x - ra.x + 1) + (x - ra.x)) * 4;  // 在接收缓冲区中的索引
		float r_float = d_img[pixelIndex + 0];
        float g_float = d_img[pixelIndex + 1];
        float b_float = d_img[pixelIndex + 2];
        float a_float = d_img[pixelIndex + 3];

		if (!over)
		{
			r_float = d_rbuffer[buffer_index + 0] + r_float * (1.0f - d_rbuffer[buffer_index + 3]);
            g_float = d_rbuffer[buffer_index + 1] + g_float * (1.0f - d_rbuffer[buffer_index + 3]);
            b_float = d_rbuffer[buffer_index + 2] + b_float * (1.0f - d_rbuffer[buffer_index + 3]);
            a_float = d_rbuffer[buffer_index + 3] + a_float * (1.0f - d_rbuffer[buffer_index + 3]);
		}
		else
		{
			r_float = r_float + d_rbuffer[buffer_index + 0] * (1.0f - a_float);
            g_float = g_float + d_rbuffer[buffer_index + 1] * (1.0f - a_float);
            b_float = b_float + d_rbuffer[buffer_index + 2] * (1.0f - a_float);
            a_float = a_float + d_rbuffer[buffer_index + 3] * (1.0f - a_float);
		}
		d_img[pixelIndex + 0] = r_float;
        d_img[pixelIndex + 1] = g_float;
        d_img[pixelIndex + 2] = b_float;
        d_img[pixelIndex + 3] = a_float;
	}
}

__global__ void gather_fillRGBBuffer(float* d_img, float* d_fbuffer, int obr_x, int obr_y, 
			int rOffset_obr, int gOffset_obr, int bOffset_obr, 
			int rOffset_fbuffer, int gOffset_fbuffer, int bOffset_fbuffer,
			Point2Di fa, Point2Di fb)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + fa.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + fa.y;

	// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     d_fbuffer[0] = fa.x;
    //     d_fbuffer[1] = fa.y;
    //     d_fbuffer[2] = fb.x;
    //     d_fbuffer[3] = fb.y;
    // }

	if(x >= fa.x && x <= fb.x && y >= fa.y && y <= fb.y)
	{
		int pixelIndex = y * obr_x + x;
		int localIndex = (y - fa.y) * (fb.x - fa.x + 1) + (x - fa.x);

		float r = d_img[rOffset_obr + pixelIndex];
        float g = d_img[gOffset_obr + pixelIndex];
        float b = d_img[bOffset_obr + pixelIndex];

		// 按照 rrrgggbbb 的顺序存储到 d_fbuffer 中
        d_fbuffer[rOffset_fbuffer + localIndex] = r;
        d_fbuffer[gOffset_fbuffer + localIndex] = g;
        d_fbuffer[bOffset_fbuffer + localIndex] = b;
	}
}

__global__ void gather_merge(float* d_sub_fbuffer, float* d_obr_rgb, 
			int rOffset_obr, int gOffset_obr, int bOffset_obr, 
			Point2Di fa, Point2Di fb, int obr_x)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + fa.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + fa.y;
	if (x <= fb.x && y <= fb.y) 
	{
        int local_idx = (y - fa.y) * (fb.x - fa.x + 1) + (x - fa.x);  // 当前像素在 sub_fbuffer 中的索引
        int pixelIndex = y * obr_x + x;  // 当前像素在 obr_rgb 中的索引

		// 合并 R, G, B 数据
        d_obr_rgb[rOffset_obr + pixelIndex] = d_sub_fbuffer[4 + local_idx];
        d_obr_rgb[gOffset_obr + pixelIndex] = d_sub_fbuffer[4 + local_idx + (fb.x - fa.x + 1) * (fb.y - fa.y + 1)];
        d_obr_rgb[bOffset_obr + pixelIndex] = d_sub_fbuffer[4 + local_idx + 2 * (fb.x - fa.x + 1) * (fb.y - fa.y + 1)];
	}
}


__global__ void gather_fileRGBABuffer(float* d_img, float* d_fbuffer, int obr_x, int obr_y, Point2Di fa, Point2Di fb)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + fa.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + fa.y;

	if (x >= fa.x && x <= fb.x && y >= fa.y && y <= fb.y) 
	{
		int pixelIndex = (y * obr_x + x) * 4; // 每个像素有4个值 (RGBA)
        int bufferIndex = ((y - fa.y) * (fb.x - fa.x + 1) + (x - fa.x)) * 4 + 4; // 为每个像素存储4个值

        d_fbuffer[bufferIndex + 0] = d_img[pixelIndex + 0]; // R
        d_fbuffer[bufferIndex + 1] = d_img[pixelIndex + 1]; // G
        d_fbuffer[bufferIndex + 2] = d_img[pixelIndex + 2]; // B
        d_fbuffer[bufferIndex + 3] = d_img[pixelIndex + 3]; // A
	}
}

__global__ void gather_mergeRGBA(float* d_sub_buffer, float* d_obr_rgba, Point2Di fa, Point2Di fb, int obr_x)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + fa.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + fa.y;

    if (x <= fb.x && y <= fb.y) 
    {
        int pixelIndex = (y * obr_x + x) * 4;  // 当前像素在 d_obr_rgba 中的索引，4 表示 RGBA 4 个分量
        int bufferIndex = ((y - fa.y) * (fb.x - fa.x + 1) + (x - fa.x)) * 4 + 4;  // 当前像素在 d_sub_buffer 中的索引

        // 将 d_sub_buffer 中的 RGBA 分量合并到 d_obr_rgba 中
        d_obr_rgba[pixelIndex + 0] = d_sub_buffer[bufferIndex + 0]; // R
        d_obr_rgba[pixelIndex + 1] = d_sub_buffer[bufferIndex + 1]; // G
        d_obr_rgba[pixelIndex + 2] = d_sub_buffer[bufferIndex + 2]; // B
        d_obr_rgba[pixelIndex + 3] = d_sub_buffer[bufferIndex + 3]; // A
    }
}

void Processor::init_node(float3& a, float3& b, int id)
{
	int r[3];
	MPI_Bcast(r, 3, MPI_INT, 0, MPI_COMM_WORLD);

	a.x = 0; a.y = 0; a.z = 0;
	b.x = r[0]; b.y = r[1]; b.z = r[2];
	whole_data_len = make_float3(b.x + 1, b.y + 1, b.z + 1);
}
void Processor::init_master(const std::string& s, float3& a, float3& b, cudaExtent volumeTotalSize)
{
	if (read_data(s, a, b, volumeTotalSize))
	{
		int r[3]; r[0] = b.x; r[1] = b.y; r[2] = b.z;
		whole_data_len = make_float3(b.x + 1, b.y + 1, b.z + 1);
		MPI_Bcast(r, 3, MPI_INT, 0, MPI_COMM_WORLD);
		std::cout << "[init_master]::MASTER: Bcast data region  [ " << r[0] << "," << r[1] << "," << r[2] << " ]" << std::endl;
	}
	else
		MPI_Abort(MPI_COMM_WORLD, 99);
}
void Processor::initScreen(unsigned int w, unsigned int h)
{
	this->camera_plane_x = w;
	this->camera_plane_y = h;
	scr_x = camera_plane_x; scr_y = camera_plane_y;
}

void Processor::initRayCaster(unsigned int w, unsigned int h)
{
	initCamera(0, 0, 0);
	setRatioUV();
}

void Processor::initCamera(int rozm_x, int rozm_y, int rozm_z)
{
	cam_old_dx = cam_old_dy = 0.0f;
	cam_dx = 0.0f;
	cam_dy = 0.0f;

	auto squar = rozm_x * rozm_x + rozm_y * rozm_y + rozm_z * rozm_z;
	auto length = squar >= 0.0f ? powf(squar, 0.5f) : 0.0f;

	cam_old_vz = cam_vz = length * 0.5f;

	setCamera();
}

void Processor::updateCamera(int rozm_x, int rozm_y, int rozm_z)
{
	auto squar = rozm_x * rozm_x + rozm_y * rozm_y + rozm_z * rozm_z;
	auto length = squar >= 0.0f ? powf(squar, 0.5f) : 0.0f;

	//cam_old_vz = cam_vz = length;
	cam_old_vz = cam_vz = 1000.0f;
	setCamera();
}

void Processor::setRatioUV()
{
	float ratio = (float)scr_x / (float)scr_y;
	camera->setRatioUV(ratio);
}

int Processor::data_size()
{
	if (!this->kdTree->root) return -1;
	Node* t = this->kdTree->root;
	while (t->back != nullptr && t->front != nullptr)
	{
		if (inInterv(Processor_ID, t->front->i1, t->front->i2))
			t = t->front;
		else
			t = t->back;
	}
	a = t->a; b = t->b;
	data_a = t->data_a; data_b = t->data_b;
	box_a = t->bbox_a; box_b = t->bbox_b;
	return (data_b.x - data_a.x + 1) * (data_b.y - data_a.y + 1) * (data_b.z - data_a.z + 1);
}

float mocnina(float x)
{
	return x * x;
}


float odmocnina(float x)
{
	return x >= 0.0f ? powf(x, 0.5f) : 0.0f;
}


void Processor::binarySwap(float* imgColor, float* imageAlpha)
{
	float* img = nullptr;
	float* reslut_rgb = new float[obr_x * obr_y * 3];
	Utils::convertRRRGGGBBBtoRGB(imgColor, obr_x * obr_y * 3, reslut_rgb);
	Utils::combineRGBA(reslut_rgb, imageAlpha, obr_x, obr_y, img);

	this->plan = new Plan[kdTree->depth];
	float3 view_dir = camera->to - camera->from;

	createPlan(Processor_ID, kdTree->depth, kdTree->root, view_dir, plan);
	obr = img;
	MPI_Barrier(MPI_COMM_WORLD);
	/* PART I - BINARY SWAP */
	int u = 0;					// 当前二叉交换的层次
	Point2Di sa, sb;			// 要发送的子图像的起始和结束位置
	Point2Di ra, rb;			// 要接收的子图像的起始和结束位置
	
	double each_round_time_start, each_round_time_end, each_round_time;
	while (kdTree->depth != 0 && u < kdTree->depth)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		each_round_time_start = MPI_Wtime();
		this->setDimensions(u, obr_a, obr_b, sa, sb, ra, rb);   // 设置发送和接收的子图像的尺寸
		this->loadBuffer(sa, sb);                               // 填充发送缓冲区		

		//std::cout << "PID " << ID << " round " << u << " begin [ " << sa.x << " , " << sa.y << " ] , [ " << sb.x << " , " << sb.y << " ] -> [ " << ra.x << " , " << ra.y << " ] , [ " << rb.x << " , " << rb.y << " ] " << std::endl;

		int sendcount = (std::abs(sa.x - sb.x) + 1) * (std::abs(sa.y - sb.y) + 1) * 4;
		int recvcount = (std::abs(ra.x - rb.x) + 1) * (std::abs(ra.y - rb.y) + 1) * 4;

		MPI_Sendrecv(&sbuffer[0], sendcount, MPI_FLOAT, this->plan[u].pid, /*TAG1*/ this->Processor_ID,
			&rbuffer[0], recvcount, MPI_FLOAT, this->plan[u].pid, /*TAG2*/ this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);

		obr_a = ra; obr_b = rb; //更新图像起始和结束位置
		this->compositng(u);
		u++;
		std::cout << "[binarySwap]:: PID " << Processor_ID << " Finished round [ " << u << " ] " << std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
		each_round_time_end = MPI_Wtime();
		each_round_time = (each_round_time_end - each_round_time_start) * 1000.0f;
		if (Processor_ID == 0)
		{
			Utils::recordBSETime(save_time_file_path.c_str(),  u, each_round_time);
		}
		// 计算通信量  
		rgba_totalSentBytes += sendcount * sizeof(float);
	}
#if GATHERING_IMAGE == 0
	/* PART II - Final Image Gathering */
	// 计算缓冲区大小
	int sendWidth = std::abs(obr_b.x - obr_a.x) + 1;
	int sendHeight = std::abs(obr_b.y - obr_a.y) + 1;
	int bufsize = sendWidth * sendHeight * 4;
	bufsize += 4;
	float* fbuffer = new float[bufsize];// 接收数据的缓冲区
	//std::cout << "PID " << Processor_ID << " obr_a [ " << obr_a.x << " " << obr_a.y << " ] obr_b [ " << obr_b.x << " " << obr_b.y << " ] buffer size " << bufsize <<
	//	" width " << sendWidth << " height " << sendHeight << std::endl;

	if (Processor_ID == 0)
	{
		for (u = 1; u < this->Processor_Size; u++)
		{
			
			MPI_Recv(&fbuffer[0], bufsize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Processor_status);
			Point2Di fa, fb;
			fa.x = (int)fbuffer[0]; fa.y = (int)fbuffer[1]; fb.x = (int)fbuffer[2]; fb.y = (int)fbuffer[3];
			//std::cout << "[partII::Recv]::PID " << Processor_ID << " fa " << fa.x << " " << fa.y << " fb " << fb.x << " " << fb.y << std::endl;
			
			int index = 4;
			for (int j = fa.y; j <= fb.y; j++)
			{
				for (int i = fa.x; i <= fb.x; i++)
				{
					float r = fbuffer[index++];
					float g = fbuffer[index++];
					float b = fbuffer[index++];
					float a = fbuffer[index++];

					int pixelIndex = (j * obr_x + i) * 4;

					// 将转换后的 float 分量存储到 obr 数组中
					obr[pixelIndex + 0] = r;
					obr[pixelIndex + 1] = g;
					obr[pixelIndex + 2] = b;
					obr[pixelIndex + 3] = a;
				}
			}

			rgba_totalSentBytes += bufsize * sizeof(float);
		}

		obr_a.x = 0; obr_a.y = 0;
		obr_b.x = obr_x ; obr_b.y = obr_y;
	}
	else
	{
		fbuffer[0] = obr_a.x; fbuffer[1] = obr_a.y;
		fbuffer[2] = obr_b.x; fbuffer[3] = obr_b.y;

		int index = 4; // 假设 fbuffer 的前4个字节是一些元数据
		for (int j = obr_a.y; j <= obr_b.y; j++)
		{
			for (int i = obr_a.x; i <= obr_b.x; i++)
			{
				// 计算当前像素在 obr 数组中的起始位置
				int pixelIndex = (j * obr_x + i) * 4;

				// 从 obr 数组中读取 RGBA 分量
				float r_float = obr[pixelIndex + 0];
				float g_float = obr[pixelIndex + 1];
				float b_float = obr[pixelIndex + 2];
				float a_float = obr[pixelIndex + 3];


				// 将转换后的分量存储到 fbuffer 中
				fbuffer[index++] = r_float;
				fbuffer[index++] = g_float;
				fbuffer[index++] = b_float;
				fbuffer[index++] = a_float;
			}
		}
		
		MPI_Send(&fbuffer[0], bufsize, MPI_FLOAT, 0, this->Processor_ID, MPI_COMM_WORLD);
	}
	if (fbuffer) { delete[] fbuffer; fbuffer = nullptr; }
	std::cout << "PID [ " << Processor_ID << " ] finished IMAGE COMPOSITING " << std::endl;
#endif
	this->reset();
}

void Processor::binarySwap_GPU(float* imgColor, float* imageAlpha)
{
	this->plan = new Plan[kdTree->depth];
	float3 view_dir = camera->to - camera->from;
	createPlan(Processor_ID, kdTree->depth, kdTree->root, view_dir, plan);
	
	size_t buffer_len = obr_x * obr_y * 3;
	size_t num_rgb_pixels = buffer_len / 3;
	int blockSize = 256;
    int gridSize = (num_rgb_pixels + blockSize - 1) / blockSize;
	float* d_imgColor = nullptr;
	cudaMalloc(&d_imgColor, buffer_len * sizeof(float));
	convertRRRGGGBBBtoRGBKernel<<<gridSize, blockSize>>>(imgColor, d_imgColor, num_rgb_pixels);
	cudaDeviceSynchronize();
	getLastCudaError("convertRRRGGGBBBtoRGB kernel failed");

	// 初始化 GPU 图像数据 将 d_imgColor 和 imageAlpha 合并成一个
	cudaMalloc(&d_obr_rgba, obr_x * obr_y * 4 * sizeof(float));
	cudaDeviceSynchronize();
	int numPixels = obr_x * obr_y;
	dim3 blockDim(16, 16);
	dim3 gridDim((obr_x + blockDim.x - 1) / blockDim.x, 
				 (obr_y + blockDim.y - 1) / blockDim.y); 
	mergeRGBAToBuffer<<<gridDim, blockDim>>>(d_imgColor, imageAlpha, d_obr_rgba, obr_x, obr_y);
	cudaDeviceSynchronize();
	getLastCudaError("mergeRGBAToBuffer kernel failed");

	MPI_Barrier(MPI_COMM_WORLD);
	// part I binary swap
	int u = 0;					// 当前二叉交换的层次
	Point2Di sa, sb;			// 要发送的子图像的起始和结束位置
	Point2Di ra, rb;			// 要接收的子图像的起始和结束位置
	Point2Di oa, ob;
	
	double each_round_time_start, each_round_time_end, each_round_time;
	double gather_d_time_start, gather_d_time_end, gather_d_time;
	gather_d_time = 0.0;

	while (kdTree->depth != 0 && u < kdTree->depth)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		each_round_time_start = MPI_Wtime();
		this->setDimensions(u, arae_a, area_b, sa, sb, ra, rb);
		this->setDimensions(u, obr_a, obr_b, sa, sb, ra, rb);
		

		oa.x = ra.x; oa.y = ra.y;
		ob.x = rb.x; ob.y = rb.y;

		int sendcount = (std::abs(sa.x - sb.x) + 1) * (std::abs(sa.y - sb.y) + 1) * 4;
		int recvcount = (std::abs(oa.x - ob.x) + 1) * (std::abs(oa.y - ob.y) + 1) * 4;

	
		// 填充缓冲区 // CUDA 内核
		int tmp_width = sb.x - sa.x + 1;
		int tmp_height = sb.y - sa.y + 1;
		blockDim = dim3(16, 16);
		gridDim = dim3((tmp_width + blockDim.x - 1) / blockDim.x, (tmp_height + blockDim.y - 1) / blockDim.y);
		fillSendRGBABuffer<<<gridDim, blockDim>>>(d_obr_rgba, d_rgba_sbuffer, sa, sb, obr_x);
		cudaDeviceSynchronize();
		getLastCudaError("fillSendRGBABuffer kernel failed");

		// std::cout << "[binarySwap_RGB]:: PID " << Processor_ID << " Finished loadColorBufferRRGGBBKernel " << std::endl;

		// 发送和接收
		MPI_Sendrecv(d_rgba_sbuffer, sendcount, MPI_FLOAT, this->plan[u].pid, /*TAG1*/ this->Processor_ID,
					   d_rgba_rbuffer, recvcount, MPI_FLOAT, this->plan[u].pid, /*TAG2*/ this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);
		
		tmpRecvCound = recvcount;
		// totalSentBytes += sendcount * sizeof(float);


		// obr_rgb_a = ra; obr_rgb_b = rb;//更新图像起始和结束位置
		obr_a = ra; obr_b = rb;//更新图像起始和结束位置
		arae_a = ra; area_b = rb;

		// this->compositngColorRRGGBB(u);
		blockDim = dim3(16, 16);
		gridDim = dim3((obr_b.x - obr_a.x + 1 + blockDim.x - 1) / blockDim.x, 
					(obr_b.y - obr_a.y + 1 + blockDim.y - 1) / blockDim.y);
		
		compositingRGBAKernel<<<gridDim, blockDim>>>(d_obr_rgba, d_rgba_rbuffer, obr_a, obr_b, obr_x, obr_y, plan[u].over, u);
		cudaDeviceSynchronize();
		getLastCudaError("compositingRGBAKernel kernel failed");
		
		rgba_totalSentBytes += sendcount * sizeof(float);
		u++;
		MPI_Barrier(MPI_COMM_WORLD);
		each_round_time_end = MPI_Wtime();
		each_round_time = (each_round_time_end - each_round_time_start) * 1000.0f;
		
		// std::cout << "[binarySwap_RGB]:: PID " << Processor_ID << " Finished round [ " << u << " ] " << std::endl;
		if (Processor_ID == 0)
		{
			Utils::recordBSETime(save_time_file_path.c_str(),  u, each_round_time);
		}
		
	}
	std::cout << "[binarySwap_RGB_GPU]:: PID " << Processor_ID << " Finished BINARY SWAP " << std::endl;

	double gather_time_s, gather_time_e, gather_time;

	MPI_Barrier(MPI_COMM_WORLD);
	gather_time_s = MPI_Wtime();

	// 拷贝回CPU
	// obr = new float[obr_x * obr_y * 4];
	// cudaMemcpy(obr, d_obr_rgba, obr_x * obr_y * 4 * sizeof(float), cudaMemcpyDeviceToHost);

#if GATHERING_IMAGE == 0
	/* PART II - Final Image Gathering */
	// 计算缓冲区大小
	int sendWidth = std::abs(obr_b.x - obr_a.x) + 1;
	int sendHeight = std::abs(obr_b.y - obr_a.y) + 1;
	int bufsize = sendWidth * sendHeight * 4;
	bufsize += 4;
	float* fbuffer = new float[bufsize];// 接收数据的缓冲区
	
	if (Processor_ID == 0)
	{
		for (u = 1; u < this->Processor_Size; u++)
		{
			
			MPI_Recv(&fbuffer[0], bufsize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Processor_status);
			Point2Di fa, fb;
			fa.x = (int)fbuffer[0]; fa.y = (int)fbuffer[1]; fb.x = (int)fbuffer[2]; fb.y = (int)fbuffer[3];
			//std::cout << "[partII::Recv]::PID " << Processor_ID << " fa " << fa.x << " " << fa.y << " fb " << fb.x << " " << fb.y << std::endl;
			
			int index = 4;
			for (int j = fa.y; j <= fb.y; j++)
			{
				for (int i = fa.x; i <= fb.x; i++)
				{
					float r = fbuffer[index++];
					float g = fbuffer[index++];
					float b = fbuffer[index++];
					float a = fbuffer[index++];

					int pixelIndex = (j * obr_x + i) * 4;

					// 将转换后的 float 分量存储到 obr 数组中
					obr[pixelIndex + 0] = r;
					obr[pixelIndex + 1] = g;
					obr[pixelIndex + 2] = b;
					obr[pixelIndex + 3] = a;
				}
			}

			rgba_totalSentBytes += bufsize * sizeof(float);
		}

		obr_a.x = 0; obr_a.y = 0;
		obr_b.x = obr_x ; obr_b.y = obr_y;
	}
	else
	{
		fbuffer[0] = obr_a.x; fbuffer[1] = obr_a.y;
		fbuffer[2] = obr_b.x; fbuffer[3] = obr_b.y;

		int index = 4; // 假设 fbuffer 的前4个字节是一些元数据
		for (int j = obr_a.y; j <= obr_b.y; j++)
		{
			for (int i = obr_a.x; i <= obr_b.x; i++)
			{
				// 计算当前像素在 obr 数组中的起始位置
				int pixelIndex = (j * obr_x + i) * 4;

				// 从 obr 数组中读取 RGBA 分量
				float r_float = obr[pixelIndex + 0];
				float g_float = obr[pixelIndex + 1];
				float b_float = obr[pixelIndex + 2];
				float a_float = obr[pixelIndex + 3];


				// 将转换后的分量存储到 fbuffer 中
				fbuffer[index++] = r_float;
				fbuffer[index++] = g_float;
				fbuffer[index++] = b_float;
				fbuffer[index++] = a_float;
			}
		}
		
		MPI_Send(&fbuffer[0], bufsize, MPI_FLOAT, 0, this->Processor_ID, MPI_COMM_WORLD);
	}
	if (fbuffer) { delete[] fbuffer; fbuffer = nullptr; }
	std::cout << "PID [ " << Processor_ID << " ] finished IMAGE COMPOSITING " << std::endl;
#endif

#if USE_GATHER == 1
	double bf_gather_s, bf_gather_e, bf_gather;

	MPI_Barrier(MPI_COMM_WORLD);
	bf_gather_s = MPI_Wtime();
	/* PART II - Final Image Gathering */
	// 计算每个进程的缓冲区大小
	int sendWidth = std::abs(obr_b.x - obr_a.x) + 1;
	int sendHeight = std::abs(obr_b.y - obr_a.y) + 1;
	int bufsize = sendWidth * sendHeight * 4 + 4; // +4 用于存储元数据
	float* fbuffer = new float[bufsize];
	float* d_fbuffer = nullptr;
	cudaMalloc(&d_fbuffer, bufsize * sizeof(float));
	// 进程 0 分配接收缓冲区
	float* d_recvbuf = nullptr;
	int* recvcounts = nullptr;
	int* displs = nullptr;
	if (Processor_ID == 0) 
	{
		// 为每个进程分配接收缓冲区
		recvcounts = new int[Processor_Size];
		displs = new int[Processor_Size];
	
		// 设置每个进程的 recvcounts 和 displs
		for (int i = 0; i < Processor_Size; i++) 
		{
			recvcounts[i] = bufsize;
			displs[i] = i * bufsize; // 每个进程的数据按顺序排列
			// std::cout << "[GATHER]:: PID " << Processor_ID << " recvcounts " << recvcounts[i] << " displs " << displs[i] << std::endl;
		}
	
		// 分配总接收缓冲区
		cudaMalloc(&d_recvbuf, Processor_Size * bufsize * sizeof(float));
	}

	// 填充 fbuffer 前4个字节存储元数据
	Point2Di fa, fb;
	cudaMemcpy(&d_fbuffer[0], &obr_a.x, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_fbuffer[1], &obr_a.y, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_fbuffer[2], &obr_b.x, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_fbuffer[3], &obr_b.y, sizeof(float), cudaMemcpyHostToDevice);
	fa.x = obr_a.x; fa.y = obr_a.y; 
	fb.x = obr_b.x; fb.y = obr_b.y;
	blockDim = dim3(16, 16);
	gridDim = dim3((obr_b.x - obr_a.x + 1 + blockDim.x - 1) / blockDim.x, 
				(obr_b.y - obr_a.y + 1 + blockDim.y - 1) / blockDim.y);
	gather_fileRGBABuffer<<<gridDim, blockDim>>>(d_obr_rgba, d_fbuffer, obr_x, obr_y, fa, fb);
	cudaDeviceSynchronize();
	getLastCudaError("gather_fileRGBABuffer kernel failed");
	// fbuffer[0] = obr_a.x;
	// fbuffer[1] = obr_a.y;
	// fbuffer[2] = obr_b.x;
	// fbuffer[3] = obr_b.y;

	// int index = 4; // 前4个字节存储元数据
	// for (int j = obr_a.y; j <= obr_b.y; j++) 
	// {
	// 	for (int i = obr_a.x; i <= obr_b.x; i++) 
	// 	{
	// 		int pixelIndex = (j * obr_x + i) * 4;

	// 		// 从 obr 数组中读取 RGBA 分量
	// 		float r_float = obr[pixelIndex + 0];
	// 		float g_float = obr[pixelIndex + 1];
	// 		float b_float = obr[pixelIndex + 2];
	// 		float a_float = obr[pixelIndex + 3];

	// 		// 存储到 fbuffer 中
	// 		fbuffer[index++] = r_float;
	// 		fbuffer[index++] = g_float;
	// 		fbuffer[index++] = b_float;
	// 		fbuffer[index++] = a_float;
	// 	}
	// }

	// // 进程 0 分配接收缓冲区
	// float* recvbuf = nullptr;
	// int* recvcounts = nullptr;
	// int* displs = nullptr;

	// if(Processor_ID == 0)
	// {
	// 	recvcounts = new int[Processor_Size];
	// 	displs = new int[Processor_Size];
	// 	// 设置所有 recvcounts 为 fixed_bufsize
	// 	for (int i = 0; i < Processor_Size; i++) 
	// 	{
	// 		recvcounts[i] = bufsize;
	// 	}
	
	// 	displs[0] = 0;
	// 	int i = 0;
	// 	std::cout << "recvcounts[" << i << "] = " << recvcounts[0] << " displs[" << i << "] = " << displs[i] << std::endl;
	// 	for (int i = 1; i < Processor_Size; i++) 
	// 	{
	// 		displs[i] = displs[i - 1] + recvcounts[i - 1];
	// 		std::cout << "recvcounts[" << i << "] = " << recvcounts[i] << " displs[" << i << "] = " << displs[i] << std::endl;
	// 	}
	// 	int totalSize = displs[Processor_Size - 1] + recvcounts[Processor_Size - 1];
	// 	recvbuf = new float[totalSize];
	// }
	MPI_Barrier(MPI_COMM_WORLD);
	bf_gather_e = MPI_Wtime();
	bf_gather = (bf_gather_e - bf_gather_s) * 1000.0f;
	if (Processor_ID == 0)
	{
		std::cout << "Before Gather Time: " << bf_gather << std::endl;
	}
	// 使用 MPI_Gatherv 收集每个进程的数据
	MPI_Gatherv(d_fbuffer, bufsize, MPI_FLOAT, d_recvbuf, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// 进程 0 合并图像
	if (Processor_ID == 0) 
	{
		for (int u = 0; u < Processor_Size; u++) 
		{
			int offset = displs[u];
			float* d_fbufferRecv = nullptr;
			cudaMalloc(&d_fbufferRecv, bufsize * sizeof(float));
			cudaMemcpy(d_fbufferRecv, d_recvbuf + offset, bufsize * sizeof(float), cudaMemcpyDeviceToDevice);
			Point2Di fa, fb;
			cudaMemcpy(&fa.x, &d_recvbuf[offset + 0], sizeof(float), cudaMemcpyDeviceToHost);
    		cudaMemcpy(&fa.y, &d_recvbuf[offset + 1], sizeof(float), cudaMemcpyDeviceToHost);
    		cudaMemcpy(&fb.x, &d_recvbuf[offset + 2], sizeof(float), cudaMemcpyDeviceToHost);
    		cudaMemcpy(&fb.y, &d_recvbuf[offset + 3], sizeof(float), cudaMemcpyDeviceToHost);
			// std::cout << "fa " << fa.x << " " << fa.y << " fb " << fb.x << " " << fb.y << std::endl;
			blockDim = dim3(16, 16);
			gridDim = dim3((fb.x - fa.x + 1 + blockDim.x - 1) / blockDim.x, 
						(fb.y - fa.y + 1 + blockDim.y - 1) / blockDim.y);
			gather_mergeRGBA<<<gridDim, blockDim>>>(d_fbufferRecv, d_obr_rgba, fa, fb, obr_x);
			cudaDeviceSynchronize();
			getLastCudaError("gather_mergeRGBA kernel failed");
			// float* fbufferRecv = recvbuf + displs[u];
			// Point2Di fa, fb;
			// fa.x = static_cast<int>(fbufferRecv[0]);
			// fa.y = static_cast<int>(fbufferRecv[1]);
			// fb.x = static_cast<int>(fbufferRecv[2]);
			// fb.y = static_cast<int>(fbufferRecv[3]);

			// int index = 4;
			// for (int j = fa.y; j <= fb.y; j++) {
			// 	for (int i = fa.x; i <= fb.x; i++) {
			// 		int pixelIndex = (j * obr_x + i) * 4;

			// 		// 从 fbufferRecv 中读取 RGBA 并存入 obr 数组
			// 		obr[pixelIndex + 0] = fbufferRecv[index++];
			// 		obr[pixelIndex + 1] = fbufferRecv[index++];
			// 		obr[pixelIndex + 2] = fbufferRecv[index++];
			// 		obr[pixelIndex + 3] = fbufferRecv[index++];
			// 	}
			// }
		}
	}

	// 清理内存
	// if (fbuffer) { delete[] fbuffer; }
	if (Processor_ID == 0) {
		delete[] recvcounts;
		delete[] displs;
		
	}
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	gather_time_e = MPI_Wtime();
	gather_time = (gather_time_e - gather_time_s) * 1000.0f;
	obr = new float[obr_x * obr_y * 4];
	cudaMemcpy(obr, d_obr_rgba, obr_x * obr_y * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	if (Processor_ID == 0)
	{
		Utils::recordCudaRenderTime(save_time_file_path.c_str(),  "gather Time:", std::to_string(Processor_Size - 1), gather_time);
	}

	this->reset();
}

void Processor::binarySwap_Alpha_GPU(float* d_img_alpha)
{
	this->plan = new Plan[kdTree->depth];
	float3 view_dir = camera->to - camera->from;
	createPlan(Processor_ID, kdTree->depth, kdTree->root, view_dir, plan);
	d_obr_alpha = d_img_alpha;


	

	// part I binary swap
	int u = 0;					// 当前二叉交换的层次
	Point2Di sa, sb;			// 要发送的子图像的起始和结束位置
	Point2Di ra, rb;			// 要接收的子图像的起始和结束位置


	MPI_Barrier(MPI_COMM_WORLD);
	// double start_time_alpha = MPI_Wtime(); // 开始计时
	
	while (kdTree->depth != 0 && u < kdTree->depth)
	{
		//std::cout << "[binarySwap_Alpha_GPU]:: PID " << Processor_ID << " Begin round [ " << u << " ] " << std::endl;
		this->setDimensions(u, obr_alpha_a, obr_alpha_b, sa, sb, ra, rb); // 更新sa,sb,ra,rb
		
		// 计算发送和接收的大小
		int sendcount = (std::abs(sa.x - sb.x) + 1) * (std::abs(sa.y - sb.y) + 1) * 1;
		int recvcount = (std::abs(ra.x - rb.x) + 1) * (std::abs(ra.y - rb.y) + 1) * 1;

		
		
		// 填充缓冲区 // CUDA 内核
		dim3 blockDim(16, 16);
		dim3 gridDim((sb.x - sa.x + 1 + blockDim.x - 1) / blockDim.x, (sb.y - sa.y + 1 + blockDim.y - 1) / blockDim.y);
		
		fillSendBuffer<<<gridDim, blockDim>>>(d_obr_alpha, d_alpha_sbuffer, sa, sb, obr_x);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		cudaDeviceSynchronize();

		
		// 发送和接收（直接使用GPU指针）
		MPI_Sendrecv(d_alpha_sbuffer, sendcount, MPI_FLOAT, this->plan[u].pid, /*TAG1*/ this->Processor_ID,
					 d_alpha_rbuffer, recvcount, MPI_FLOAT, this->plan[u].pid, /*TAG2*/ this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);
	
	
		// 更新alpha的值
        dim3 gridDimUpdate((rb.x - ra.x + 1 + blockDim.x - 1) / blockDim.x, (rb.y - ra.y + 1 + blockDim.y - 1) / blockDim.y);
        updateAlpha<<<gridDimUpdate, blockDim>>>(d_img_alpha, d_alpha_rbuffer, d_alpha_values_u, 
				ra, rb, obr_x, obr_y, plan[u].over, u);
        err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		cudaDeviceSynchronize();

		obr_alpha_a = ra; obr_alpha_b = rb;//更新图像起始和结束位置

		alpha_totalSentBytes += sendcount * sizeof(float);

		
		u++;
		//std::cout << "[binarySwap_Alpha_GPU]:: PID " << Processor_ID << " Finished round [ " << u << " ] " << std::endl;
	}
	// MPI_Barrier(MPI_COMM_WORLD);
	// double end_time_alpha = MPI_Wtime(); // 结束计时
	// double elapsed_time_alpha = end_time_alpha - start_time_alpha;
	// elapsed_time_alpha *= 1000.0;
	// Utils::recordCudaRenderTime("./alpha_change_time.txt",Processor_Size, Processor_ID, elapsed_time_alpha);
}

void Processor::AlphaGathering_CPU()
{
	obr_alpha = new float[obr_x * obr_y];
	cudaMemcpy(obr_alpha, d_obr_alpha, obr_x * obr_y * sizeof(float), cudaMemcpyDeviceToHost);
	// part II final image gathering
	// 计算缓冲区大小
	int sendWidth = std::abs(obr_alpha_b.x - obr_alpha_a.x) + 1;
	int sendHeight = std::abs(obr_alpha_b.y - obr_alpha_a.y) + 1;
	int bufsize = sendWidth * sendHeight * 1;
	bufsize += 4;
	float* fbuffer = new float[bufsize];// 接收数据的缓冲区
	float* recvbuf = nullptr;

	// 准备要发送的数据
	fbuffer[0] = obr_alpha_a.x;
	fbuffer[1] = obr_alpha_a.y;
	fbuffer[2] = obr_alpha_b.x;
	fbuffer[3] = obr_alpha_b.y;

	int index = 4; // 前四个字节是元数据
	for (int j = obr_alpha_a.y; j <= obr_alpha_b.y; j++) {
		for (int i = obr_alpha_a.x; i <= obr_alpha_b.x; i++) {
			int pixelIndex = j * obr_x + i;
			fbuffer[index++] = obr_alpha[pixelIndex]; // 填充数据
		}
	}

	// 在进程 0 上分配接收缓冲区
	if (Processor_ID == 0) {
		recvbuf = new float[Processor_Size * bufsize]; // 进程数 * 每个进程的数据大小
	}

	// 使用 MPI_Gather 从所有进程收集数据到进程 0
	MPI_Gather(fbuffer, bufsize, MPI_FLOAT, recvbuf, bufsize, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// 处理进程 0 收集的数据
	if (Processor_ID == 0) {
		for (int p = 1; p < Processor_Size; p++) {
			int offset = p * bufsize;
			Point2Di fa, fb;
			fa.x = (int)recvbuf[offset + 0];
			fa.y = (int)recvbuf[offset + 1];
			fb.x = (int)recvbuf[offset + 2];
			fb.y = (int)recvbuf[offset + 3];
			int index = 4;
			for (int j = fa.y; j <= fb.y; j++) {
				for (int i = fa.x; i <= fb.x; i++) {
					int pixelIndex = j * obr_x + i;
					obr_alpha[pixelIndex] = recvbuf[offset + index++];
				}
			}
		}
	}

	// 释放内存
	if (fbuffer) { delete[] fbuffer; }
	if (recvbuf && Processor_ID == 0) { delete[] recvbuf; }
	this->reset();

}

#pragma region hide
void Processor::binarySwap_Alpha(float* img_alpha)
{
	this->plan = new Plan[kdTree->depth];
	float3 view_dir = camera->to - camera->from;
	createPlan(Processor_ID, kdTree->depth, kdTree->root, view_dir, plan);
	obr_alpha = img_alpha;

	alpha_values_u = new float** [kdTree->depth];
	for (int u = 0; u < kdTree->depth; ++u) {
		alpha_values_u[u] = new float* [obr_y];
		for (int y = 0; y < obr_y; ++y) {
			alpha_values_u[u][y] = new float[obr_x];
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);
	// part I binary swap
	int u = 0;					// 当前二叉交换的层次
	Point2Di sa, sb;			// 要发送的子图像的起始和结束位置
	Point2Di ra, rb;			// 要接收的子图像的起始和结束位置
	while (kdTree->depth != 0 && u < kdTree->depth)
	{
		this->setDimensions(u, obr_alpha_a, obr_alpha_b, sa, sb, ra, rb);
		// 填充缓冲区
		int index = 0;
		for (int j = sa.y; j <= sb.y; j++)
		{
			for (int i = sa.x; i <= sb.x; i++)
			{
				// Assuming row-major order and 4 floats per pixel (RGBA)
				int pixelIndex = (j * obr_x + i) * 1;

				// Extracting color components from the float array
				float a = obr_alpha[pixelIndex + 0];
				alpha_sbuffer[index++] = a;
			}
		}
		

		// 计算发送和接收的大小
		int sendcount = (std::abs(sa.x - sb.x) + 1) * (std::abs(sa.y - sb.y) + 1) * 1;
		int recvcount = (std::abs(ra.x - rb.x) + 1) * (std::abs(ra.y - rb.y) + 1) * 1;
		// 发送和接收
		MPI_Sendrecv(&alpha_sbuffer[0], sendcount, MPI_FLOAT, this->plan[u].pid, /*TAG1*/ this->Processor_ID,
			&alpha_rbuffer[0], recvcount, MPI_FLOAT, this->plan[u].pid, /*TAG2*/ this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);
	
		obr_alpha_a = ra; obr_alpha_b = rb;//更新图像起始和结束位置
		this->visibleFunc(u); // 更新alpha的值
		u++;
		// std::cout << "[binarySwap_Alpha]:: PID " << Processor_ID << " Finished round [ " << u << " ] " << std::endl;
	}

	// part II final image gathering
	// 计算缓冲区大小
	int sendWidth = std::abs(obr_alpha_b.x - obr_alpha_a.x) + 1;
	int sendHeight = std::abs(obr_alpha_b.y - obr_alpha_a.y) + 1;
	int bufsize = sendWidth * sendHeight * 1;
	bufsize += 4;
	float* fbuffer = new float[bufsize];// 接收数据的缓冲区

	if (Processor_ID == 0)
	{
		for (u = 1; u < this->Processor_Size; u++)
		{
			MPI_Recv(&fbuffer[0], bufsize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Processor_status);
			Point2Di fa, fb;
			fa.x = (int)fbuffer[0]; fa.y = (int)fbuffer[1]; fb.x = (int)fbuffer[2]; fb.y = (int)fbuffer[3];
			int index = 4;
			for (int j = fa.y; j <= fb.y; j++)
			{
				for (int i = fa.x; i <= fb.x; i++)
				{
					float a = fbuffer[index++];

					int pixelIndex = j * obr_x + i;
					obr_alpha[pixelIndex] = a;
				}
			}
		}

		obr_alpha_a.x = 0; obr_alpha_a.y = 0;
		obr_alpha_b.x = obr_x; obr_alpha_b.y = obr_y;
	}
	else
	{
		fbuffer[0] = obr_alpha_a.x; fbuffer[1] = obr_alpha_a.y;
		fbuffer[2] = obr_alpha_b.x; fbuffer[3] = obr_alpha_b.y;

		int index = 4; // 假设 fbuffer 的前4个字节是一些元数据
		for (int j = obr_alpha_a.y; j <= obr_alpha_b.y; j++)
		{
			for (int i = obr_alpha_a.x; i <= obr_alpha_b.x; i++)
			{
				// 计算当前像素在 obr_alpha 数组中的起始位置
				int pixelIndex = j * obr_x + i;

				// 从 obr_alpha 数组中读取Alpha分量
				float a_float = obr_alpha[pixelIndex];

				// 将Alpha分量存储到 fbuffer 中
				fbuffer[index++] = a_float;
			}
		}
		MPI_Send(&fbuffer[0], bufsize, MPI_FLOAT, 0, this->Processor_ID, MPI_COMM_WORLD);
	}
	if (fbuffer) { delete[] fbuffer; fbuffer = nullptr; }
	// std::cout << "PID [ " << Processor_ID << " ] finished IMAGE ALPHA COMPOSITING " << std::endl;
	this->reset();

}
#pragma endregion hide

#pragma region hide
void Processor::binarySwap_RGB(float* img_color, int MinX, int MinY, int MaxX, int MaxY, bool bUseCompression, bool bUseArea)
{
	
	this->plan = new Plan[kdTree->depth];
	float3 view_dir = camera->to - camera->from;
	createPlan(Processor_ID, kdTree->depth, kdTree->root, view_dir, plan);
	obr_rgb = img_color;

	Point2Di ea; ea.x = MinX; ea.y = MinY;
	Point2Di eb; eb.x = MaxX; eb.y = MaxY;
	Point2Di overlap_a; 
	Point2Di overlap_b;

	// 统计时间
	double compress_time = 0.0;
	double decompress_time = 0.0;
	double each_round_time = 0.0;

	MPI_Barrier(MPI_COMM_WORLD);
	// part I binary swap
	int u = 0;					// 当前二叉交换的层次
	Point2Di sa, sb;			// 要发送的子图像的起始和结束位置
	Point2Di ra, rb;			// 要接收的子图像的起始和结束位置
	Point2Di oa, ob;
	float total_error = 0.005f;
	float process_error = total_error * 0.6f;
	float gather_error = process_error * 0.4f;
	float remaining_error = process_error;  // 剩余的误差限度
	// Way 1
	float errorBound = process_error / kdTree->depth;
	errorBound = errorBound * 255.0f;
	// way 2 
	float run_one = process_error / pow(2, 1);
	float run_two = process_error / pow(2, 2);
	while (kdTree->depth != 0 && u < kdTree->depth)
	{
		// MPI_Barrier(MPI_COMM_WORLD);
		// double start_round_time = MPI_Wtime();
		this->setDimensions(u, arae_a, area_b, sa, sb, ra, rb);
		this->setDimensions(u, obr_rgb_a, obr_rgb_b, sa, sb, ra, rb);
		// this->setDimensions(u, arae_a, arae_a, sa, sb, ra, rb);
		// 填充缓冲区

		if(bUseArea == true)
		{
			bool flag = computeOverlap(sa, sb, ea, eb, overlap_a, overlap_b);
			if(flag == false)
			{
				// 如果没有交集，跳过这个轮次
				std::cout << "[binarySwap_RGB]:: PID " << Processor_ID << " No overlap in round [ " << u << " ]" << std::endl;
				u++;
				//obr_rgb_a = oa; obr_rgb_b = ob;//更新图像起始和结束位置
				arae_a = ra; area_b = rb;
				continue;
			}
			std::cout << "[======] PID " << Processor_ID << " u " << u << " sa sb [ " 
			<< sa.x << " " << sa.y << " , " << sb.x << " " << sb.y << " ] ea eb [ "
			<< ea.x << " " << ea.y << " , " << eb.x << " " << eb.y << " ] oa ob [ "
			<< overlap_a.x << " " << overlap_a.y << " , " << overlap_b.x << " " << overlap_b.y << " ] "
			<< std::endl;
			
			if(flag == false)
			{
				sa = overlap_a; sb = overlap_b;
				int overlapInfo[4] = {overlap_a.x, overlap_a.y, overlap_b.x, overlap_b.y};
				int recvOverlapInfo[4];
				MPI_Sendrecv(overlapInfo, 4, MPI_INT, this->plan[u].pid, this->Processor_ID,
							recvOverlapInfo, 4, MPI_INT, this->plan[u].pid, this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);
				
							
				oa.x = recvOverlapInfo[0]; oa.y = recvOverlapInfo[1];
				ob.x = recvOverlapInfo[2]; ob.y = recvOverlapInfo[3];
				//std::cout << "finished ========================= recv 1\n";
			}
			else
			{
				oa.x = ra.x; oa.y = ra.y;
				ob.x = rb.x; ob.y = rb.y;
			}
		}
		else
		{
			oa.x = ra.x; oa.y = ra.y;
			ob.x = rb.x; ob.y = rb.y;
		}
		

		// this->loadColorBuffer(sa, sb);
		this->loadColorBufferRRGGBB(sa, sb); // sbuffer rrggbb 格式
		if(bUseCompression)
		{
			// if (u == 0) {
			// 	errorBound = run_two;
			// }
			// else if (u == 1) {
			// 	errorBound = run_one;
			// }
			size_t outSize; 
			size_t nbEle = (std::abs(sa.x - sb.x) + 1) * (std::abs(sa.y - sb.y) + 1) * 3;
			
			
			// double start_compress = MPI_Wtime();
			unsigned char* bytes =  SZx_fast_compress_args(SZx_NO_BLOCK_FAST_CMPR, SZx_FLOAT, rgb_sbuffer, &outSize, ABS, errorBound, 0.001, 0, 0, 0, 0, 0, 0, nbEle);
			// double end_compress = MPI_Wtime();
			// compress_time += (end_compress - start_compress);
			// way 1
			std::cout << "[binarySwap_RGB]:: PID " << Processor_ID << " CALC round [ " << u 
				<< " ] COMPRESS nbEle [ " << nbEle << "] compression size [ " << outSize 
				<< " ] CR " << 1.0f*nbEle*sizeof(float)/outSize << " ] " 
				<< " errorBound " << errorBound << std::endl;
	
			// 测试。理论可以得到类似的结果 
			// 解压缩需要 bytes, byteLength, nbEle 要用MPI发送
			// 先发送 byteLength 和 nbEle
			size_t sendInfo[2] = { outSize, nbEle };
			size_t recvInfo[2];
			MPI_Sendrecv(sendInfo, 2, MPI_UNSIGNED_LONG, this->plan[u].pid, this->Processor_ID,
					recvInfo, 2, MPI_UNSIGNED_LONG, this->plan[u].pid, this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);


			size_t recv_byteLength = recvInfo[0];
			size_t recv_nbEle = recvInfo[1];
			// 发送压缩的数据
			unsigned char* receivedCompressedBytes = new unsigned char[recv_byteLength];
			MPI_Sendrecv(bytes, outSize, MPI_UNSIGNED_CHAR, this->plan[u].pid, /*TAG1*/ this->Processor_ID,
					receivedCompressedBytes, recv_byteLength, MPI_UNSIGNED_CHAR, this->plan[u].pid, /*TAG2*/ this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);

			// double start_decompress = MPI_Wtime();
			float *decompressedData = (float*)SZx_fast_decompress(SZx_NO_BLOCK_FAST_CMPR, SZx_FLOAT, receivedCompressedBytes, recv_byteLength, 0, 0, 0, 0, recv_nbEle);
			// double end_decompress = MPI_Wtime();
			// decompress_time += (end_decompress - start_decompress);
			float* tmp_buffer = new float[recv_nbEle];
			//Utils::convertRRRGGGBBBtoRGB(decompressedData, recv_nbEle, tmp_buffer);
			tmpRecvCound = recv_nbEle;
			
			// obr_rgb_a = ra; obr_rgb_b = rb;//更新图像起始和结束位置
			obr_rgb_a = oa; obr_rgb_b = ob;//更新图像起始和结束位置
			arae_a = ra; area_b = rb;
			
			this->compositngColorRRGGBB(u, decompressedData);
			//计算实际误差


			delete[] receivedCompressedBytes;
			delete[] bytes;
			delete[] decompressedData;

			// totalSentBytes += outSize;
		}
		else if(bUseCompression == false) 
		{
			// 计算发送和接收的大小
			int sendcount = (std::abs(sa.x - sb.x) + 1) * (std::abs(sa.y - sb.y) + 1) * 3;
			int recvcount = (std::abs(oa.x - ob.x) + 1) * (std::abs(oa.y - ob.y) + 1) * 3;
			// 发送和接收
			MPI_Sendrecv(&rgb_sbuffer[0], sendcount, MPI_FLOAT, this->plan[u].pid, /*TAG1*/ this->Processor_ID,
				  		 &rgb_rbuffer[0], recvcount, MPI_FLOAT, this->plan[u].pid, /*TAG2*/ this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);
			
			tmpRecvCound = recvcount;
			// totalSentBytes += sendcount * sizeof(float);

			obr_rgb_a = oa; obr_rgb_b = ob;//更新图像起始和结束位置
			arae_a = ra; area_b = rb;
			this->compositngColorRRGGBB(u);
		}
		
		
		
		u++;
		// double end_round_time = MPI_Wtime();
		// each_round_time += (end_round_time - start_round_time);
		// std::cout << "[binarySwap_RGB]:: PID " << Processor_ID << " Finished round [ " << u << " ] " << std::endl;
		// Utils::recordCudaRenderTime("./compress_time.txt", u, Processor_ID, compress_time * 1000.0f);
		// Utils::recordCudaRenderTime("./decompress_time.txt", u, Processor_ID, decompress_time * 1000.0f);
		// Utils::recordCudaRenderTime("./each_round_time.txt", u, Processor_ID, each_round_time * 1000.0f);
		
	}



	// part II final image gathering
	// 计算缓冲区大小
	obr_rgb_a.x = arae_a.x; obr_rgb_a.y = arae_a.y;
	obr_rgb_b.x = area_b.x; obr_rgb_b.y = area_b.y;
	int sendWidth = std::abs(obr_rgb_b.x - obr_rgb_a.x) + 1;
	int sendHeight = std::abs(obr_rgb_b.y - obr_rgb_a.y) + 1;
	// int sendWidth = std::abs(area_b.x - arae_a.x) + 1;
	// int sendHeight = std::abs(area_b.y - arae_a.y) + 1;
	int bufsize = sendWidth * sendHeight * 3;
	bufsize += 4;
	float* fbuffer = new float[bufsize];// 接收数据的缓冲区



	// R, G, B 通道在 obr_rgb 中的起始偏移量
	int totalPixels = obr_x * obr_y;
	int rOffset_obr = 0;                      // R 通道从 0 开始
	int gOffset_obr = totalPixels;            // G 通道从 totalPixels 开始
	int bOffset_obr = totalPixels * 2;        // B 通道从 2 * totalPixels 开始 

	// 所有进程填充自己的 fbuffer 数据，包括进程 0
	fbuffer[0] = obr_rgb_a.x; 
	fbuffer[1] = obr_rgb_a.y; 
	fbuffer[2] = obr_rgb_b.x; 
	fbuffer[3] = obr_rgb_b.y;

	int index = 4; // 假设 fbuffer 的前4个字节是一些元数据
	for (int j = obr_rgb_a.y; j <= obr_rgb_b.y; j++) {
		for (int i = obr_rgb_a.x; i <= obr_rgb_b.x; i++) {
			int pixelIndex = (j * obr_x + i) * 1;

			// 从 color 数组中读取 RGB 分量
			float r_float = obr_rgb[rOffset_obr + pixelIndex];
			float g_float = obr_rgb[gOffset_obr + pixelIndex];
			float b_float = obr_rgb[bOffset_obr + pixelIndex];

			// 将RGB分量存储到 fbuffer 中
			fbuffer[index++] = r_float;
			fbuffer[index++] = g_float;
			fbuffer[index++] = b_float;
		}
	}

	// 进程 0 分配接收缓冲区
	float* recvbuf = nullptr;
	if (Processor_ID == 0) {
		
		recvbuf = new float[Processor_Size * bufsize]; // 接收所有进程数据的缓冲区
	}
	if (bUseCompression == false)
	{
		// 所有进程调用 MPI_Gather，将 fbuffer 发送给进程 0
		MPI_Gather(fbuffer, bufsize, MPI_FLOAT, recvbuf, bufsize, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// 进程 0 收集数据后处理
		if (Processor_ID == 0)
		{
			for (int p = 0; p < Processor_Size; p++) 
			{
				int offset = p * bufsize;
				Point2Di fa, fb;
				fa.x = static_cast<int>(recvbuf[offset + 0]);
				fa.y = static_cast<int>(recvbuf[offset + 1]);
				fb.x = static_cast<int>(recvbuf[offset + 2]);
				fb.y = static_cast<int>(recvbuf[offset + 3]);

				int index = 4;
				for (int j = fa.y; j <= fb.y; j++) {
					for (int i = fa.x; i <= fb.x; i++) {
						int pixelIndex = (j * obr_x + i) * 1;

						float r = recvbuf[offset + index++];
						float g = recvbuf[offset + index++];
						float b = recvbuf[offset + index++];

						obr_rgb[rOffset_obr + pixelIndex] = r;
						obr_rgb[gOffset_obr + pixelIndex] = g;
						obr_rgb[bOffset_obr + pixelIndex] = b;
					}
				}
			}
			
		}
		// 清理资源
		if (fbuffer) { delete[] fbuffer; }
		if (recvbuf) { delete[] recvbuf; }
	}
	else if(bUseCompression == true)
	{
		// 压缩数据
		size_t outSize;
		size_t nbEle = bufsize - 4;
		unsigned char* compressedData = SZx_fast_compress_args(SZx_WITH_BLOCK_FAST_CMPR, SZx_FLOAT, fbuffer + 4, &outSize, REL, gather_error, 0.001, 0, 0, 0, 0, 0, 0, nbEle);

		// 定义 MetaData 结构体
		struct MetaData
		{
			int fa_x, fa_y, fb_x, fb_y;  // 数据范围
			size_t outSize;              // 压缩后的大小
			size_t nbEle;                // 原始数据元素个数
		};

		// 创建并发送元数据
		MetaData meta = {static_cast<int>(obr_rgb_a.x), static_cast<int>(obr_rgb_a.y),
						static_cast<int>(obr_rgb_b.x), static_cast<int>(obr_rgb_b.y), outSize, nbEle};

		MetaData* recvMeta = nullptr;
		if (Processor_ID == 0)
		{
			recvMeta = new MetaData[Processor_Size];  // 接收所有进程的元数据
		}

		// 使用 MPI_Gather 收集元数据
		MPI_Gather(&meta, sizeof(MetaData), MPI_BYTE, recvMeta, sizeof(MetaData), MPI_BYTE, 0, MPI_COMM_WORLD);

		unsigned char* recvbuf_compressed = nullptr;
		int* recvOutSizes = nullptr;
		int* recvnbEleSize = nullptr;
		int* displs = nullptr;
		// 进程 0 处理元数据，准备进行 MPI_Gatherv
		if (Processor_ID == 0)
		{
			// 处理元数据：提取每个进程的 outSize 和 nbEle，并计算总的接收大小
			recvOutSizes = new int[Processor_Size];  // 每个进程的压缩数据大小 outSize
			recvnbEleSize = new int[Processor_Size]; // 每个进程原始数据的元素个数 nbEle
			displs = new int[Processor_Size];        // 偏移量

			size_t totalCompressedSize = 0;
			displs[0] = 0;  // 初始化第一个偏移量

			// 处理元数据并计算总压缩大小和偏移量
			for (int i = 0; i < Processor_Size; i++)
			{
				recvOutSizes[i] = static_cast<int>(recvMeta[i].outSize);
				recvnbEleSize[i] = static_cast<int>(recvMeta[i].nbEle);
				totalCompressedSize += recvOutSizes[i];  // 计算所有压缩数据的总大小

				if (i > 0)
				{
					displs[i] = displs[i - 1] + recvOutSizes[i - 1];
				}
				std::cout << "[DOTEST===] " << i << " outSize " << recvOutSizes[i] << " nbELeSize " << recvnbEleSize[i] << std::endl;
			}

			// 为进程 0 分配接收压缩数据的缓冲区
			recvbuf_compressed = new unsigned char[totalCompressedSize];
		}

		// 使用 MPI_Gatherv 收集压缩数据
		// std::cout << "[==========================] before" << std::endl;
		MPI_Gatherv(compressedData, static_cast<int>(outSize), MPI_UNSIGNED_CHAR, recvbuf_compressed, recvOutSizes, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
		// std::cout << "[==========================] after" << std::endl;
		if(Processor_ID == 0)
		{
			// 进程 0 解压和合成数据
			for (int p = 1; p < Processor_Size; p++)
			{
				// 获取压缩数据和元素个数
				unsigned char* recvCompressedData = recvbuf_compressed + displs[p];
				size_t recv_outSize = static_cast<size_t>(recvOutSizes[p]);
				size_t recv_nbEle = static_cast<size_t>(recvnbEleSize[p]);

				// 打印调试信息，确保 recvCompressedData 和 recvOutSizes 正确
				std::cout << "[Process " << p << "] Compressed data size: " << recvOutSizes[p] 
						<< ", Element count: " << recv_nbEle << std::endl;

				// 检查recvCompressedData 和 recvOutSizes[p] 是否有效
				if (recvCompressedData == nullptr || recvOutSizes[p] == 0) {
					std::cerr << "Error: Invalid compressed data or size for process " << p << std::endl;
					continue; // 跳过错误的进程
				}

				// 解压每个进程的数据
				std::cout << "[2222222]:: " << p << "before decompress" << std::endl;
				float* decompressedData = (float*)SZx_fast_decompress(SZx_WITH_BLOCK_FAST_CMPR, SZx_FLOAT, recvCompressedData, recv_outSize, 0, 0, 0, 0, recv_nbEle);
				std::cout << "[2222222]:: " << p << "finished decompress" << std::endl;
				// 使用解压后的数据处理 RGB 数据
				MetaData& meta = recvMeta[p];
				Point2Di fa, fb;
				fa.x = meta.fa_x;
				fa.y = meta.fa_y;
				fb.x = meta.fb_x;
				fb.y = meta.fb_y;

				int index = 0;  // 解压后的数据从第0项开始
				for (int j = fa.y; j <= fb.y; j++)
				{
					for (int i = fa.x; i <= fb.x; i++)
					{
						int pixelIndex = (j * obr_x + i) * 1;
						float r = decompressedData[index++];
						float g = decompressedData[index++];
						float b = decompressedData[index++];

						// 将解压后的 R, G, B 数据存储到 obr_rgb 中
						obr_rgb[rOffset_obr + pixelIndex] = r;
						obr_rgb[gOffset_obr + pixelIndex] = g;
						obr_rgb[bOffset_obr + pixelIndex] = b;
					}
				}

				// 清理解压后的数据
				delete[] decompressedData;
			}

			// 清理内存
			delete[] recvOutSizes;
			delete[] recvnbEleSize;
			delete[] displs;
			delete[] recvbuf_compressed;
			delete[] recvMeta;
		}

		// 清理压缩数据
		delete[] compressedData;
	
	}
		
	
	
/*
	// R, G, B 通道在 obr_rgb 中的起始偏移量
	int totalPixels = obr_x * obr_y;
	int rOffset_obr = 0;                      // R 通道从 0 开始
	int gOffset_obr = totalPixels;            // G 通道从 totalPixels 开始
	int bOffset_obr = totalPixels * 2;        // B 通道从 2 * totalPixels 开始 
	
	if (Processor_ID == 0)
	{
		for (u = 1; u < this->Processor_Size; u++)
		{
			MPI_Recv(&fbuffer[0], bufsize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Processor_status);
			Point2Di fa, fb;
			fa.x = (int)fbuffer[0]; fa.y = (int)fbuffer[1]; fb.x = (int)fbuffer[2]; fb.y = (int)fbuffer[3];
			int index = 4;
			for (int j = fa.y; j <= fb.y; j++)
			{
				for (int i = fa.x; i <= fb.x; i++)
				{
					float r = fbuffer[index++];
					float g = fbuffer[index++];
					float b = fbuffer[index++];

					int pixelIndex = (j * obr_x + i) * 1;
					obr_rgb[rOffset_obr + pixelIndex] = r;
					obr_rgb[gOffset_obr + pixelIndex] = g;
					obr_rgb[bOffset_obr + pixelIndex] = b;
				}
			}
		}

		obr_rgb_a.x = 0; obr_rgb_a.y = 0;
		obr_rgb_b.x = obr_x; obr_rgb_b.y = obr_y;
	}
	else
	{
		fbuffer[0] = obr_rgb_a.x; fbuffer[1] = obr_rgb_a.y;
		fbuffer[2] = obr_rgb_b.x; fbuffer[3] = obr_rgb_b.y;

		// R, G, B 通道在 obr_rgb 中的起始偏移量
		int totalPixels = obr_x * obr_y;
		int rOffset = 0;                      // R 通道从 0 开始
		int gOffset = totalPixels;            // G 通道从 totalPixels 开始
		int bOffset = totalPixels * 2;        // B 通道从 2 * totalPixels 开始

		int index = 4; // 假设 fbuffer 的前4个字节是一些元数据
		for (int j = obr_rgb_a.y; j <= obr_rgb_b.y; j++)
		{
			for (int i = obr_rgb_a.x; i <= obr_rgb_b.x; i++)
			{
				// 计算当前像素在 obr_rgb 数组中的起始位置
				int pixelIndex = (j * obr_x + i) * 1;

				// 从 color 数组中读取 RGB 分量
				float r_float = obr_rgb[rOffset  + pixelIndex];
				float g_float = obr_rgb[gOffset  + pixelIndex];
				float b_float = obr_rgb[bOffset  + pixelIndex];

				// 将RGB分量存储到 fbuffer 中
				fbuffer[index++] = r_float;
				fbuffer[index++] = g_float;
				fbuffer[index++] = b_float;
			}
		}
		MPI_Send(&fbuffer[0], bufsize, MPI_FLOAT, 0, this->Processor_ID, MPI_COMM_WORLD);
	}
	if (fbuffer) { delete[] fbuffer; fbuffer = nullptr; }
*/

	// std::cout << "PID [ " << Processor_ID << " ] finished IMAGE COLOR COMPOSITING " << std::endl;
	this->reset();


	for (int u = 0; u < kdTree->depth; ++u) {
		for (int y = 0; y < obr_y; ++y) {
			delete[] alpha_values_u[u][y];
		}
		delete[] alpha_values_u[u];
	}
	delete[] alpha_values_u;
	alpha_values_u = nullptr;
}
#pragma endregion hide

void Processor::binarySwap_RGB_GPU(float* img, int MinX, int MinY, int MaxX, int MaxY, bool bUseCompression, bool bUseArea)
{
	d_obr_rgb = img;

	// 为了计算有效面积
	Point2Di ea; ea.x = MinX; ea.y = MinY;
	Point2Di eb; eb.x = MaxX; eb.y = MaxY;
	Point2Di overlap_a; 
	Point2Di overlap_b;


	MPI_Barrier(MPI_COMM_WORLD);
	// part I binary swap
	int u = 0;					// 当前二叉交换的层次
	Point2Di sa, sb;			// 要发送的子图像的起始和结束位置
	Point2Di ra, rb;			// 要接收的子图像的起始和结束位置
	Point2Di oa, ob;

	// 误差限度
	float total_error = 0.005f;
	float process_error = total_error * 0.6f;
	float gather_error = process_error * 0.4f;
	float remaining_error = process_error;  // 剩余的误差限度
	// Way 1
	float errorBound = process_error / kdTree->depth;
	// way 2 
	float run_one = process_error / pow(2, 1);
	float run_two = process_error / pow(2, 2);

	double each_round_time_start, each_round_time_end, each_round_time;
	double compress_time_start, compress_time_end, compress_time;
	double decompress_time_start, decompress_time_end, decompress_time;
	double gather_d_time_start, gather_d_time_end, gather_d_time;
	gather_d_time = 0.0;
	while (kdTree->depth != 0 && u < kdTree->depth)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		each_round_time_start = MPI_Wtime();
		this->setDimensions(u, arae_a, area_b, sa, sb, ra, rb);
		this->setDimensions(u, obr_rgb_a, obr_rgb_b, sa, sb, ra, rb);
		

		if(bUseArea == true)
		{
			bool flag = computeOverlap(sa, sb, ea, eb, overlap_a, overlap_b);
			if(flag == false)
			{
				// 如果没有交集，跳过这个轮次
				// std::cout << "[binarySwap_RGB]:: PID " << Processor_ID << " No overlap in round [ " << u << " ]" << std::endl;
				u++;
				//obr_rgb_a = oa; obr_rgb_b = ob;//更新图像起始和结束位置
				arae_a = ra; area_b = rb;
				continue;
			}
			if(flag == true)
			{
				sa = overlap_a; sb = overlap_b;
				int overlapInfo[4] = {overlap_a.x, overlap_a.y, overlap_b.x, overlap_b.y};
				int recvOverlapInfo[4];
				MPI_Sendrecv(overlapInfo, 4, MPI_INT, this->plan[u].pid, this->Processor_ID,
							recvOverlapInfo, 4, MPI_INT, this->plan[u].pid, this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);
				
							
				oa.x = recvOverlapInfo[0]; oa.y = recvOverlapInfo[1];
				ob.x = recvOverlapInfo[2]; ob.y = recvOverlapInfo[3];
				// std::cout << "finished ========================= recv 1\n";
			}
			else
			{
				oa.x = ra.x; oa.y = ra.y;
				ob.x = rb.x; ob.y = rb.y;
			}
		}
		else
		{
			oa.x = ra.x; oa.y = ra.y;
			ob.x = rb.x; ob.y = rb.y;
		}

		int sendcount = (std::abs(sa.x - sb.x) + 1) * (std::abs(sa.y - sb.y) + 1) * 3;
		int recvcount = (std::abs(oa.x - ob.x) + 1) * (std::abs(oa.y - ob.y) + 1) * 3;


		// 填充缓冲区 // CUDA 内核
		dim3 blockDim(16, 16);
		dim3 gridDim((sb.x - sa.x + 1 + blockDim.x - 1) / blockDim.x, (sb.y - sa.y + 1 + blockDim.y - 1) / blockDim.y);
		loadColorBufferRRGGBBKernel<<<gridDim, blockDim>>>(d_obr_rgb, d_rgb_sbuffer, sa, sb, obr_x, obr_y);
		cudaDeviceSynchronize();

		// std::cout << "[binarySwap_RGB]:: PID " << Processor_ID << " Finished loadColorBufferRRGGBBKernel " << std::endl;

		if(bUseCompression)
		{
			// if (u == 0) {
			// 	errorBound = run_two;
			// }
			// else if (u == 1) {
			// 	errorBound = run_one;
			// }
			
			MPI_Barrier(MPI_COMM_WORLD);
			compress_time_start = MPI_Wtime();
			nbEle = (std::abs(sa.x - sb.x) + 1) * (std::abs(sa.y - sb.y) + 1) * 3;
			SZp_compress_deviceptr_f32(d_rgb_sbuffer, d_cmpBytes, nbEle, &outSize, errorBound, stream);
			getLastCudaError("SZp compress failed");
			MPI_Barrier(MPI_COMM_WORLD);
			compress_time_end = MPI_Wtime();
			compress_time = (compress_time_end - compress_time_start) * 1000.0f;
			
			// // 解压缩需要 bytes, byteLength, nbEle 要用MPI发送
			// // 先发送 byteLength 和 nbEle
			// 先发送 byteLength 和 nbEle
			size_t sendInfo[2] = { outSize, nbEle };
			size_t recvInfo[2];
			MPI_Sendrecv(sendInfo, 2, MPI_UNSIGNED_LONG, this->plan[u].pid, this->Processor_ID,
					     recvInfo, 2, MPI_UNSIGNED_LONG, this->plan[u].pid, this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);

			// size_t recv_byteLength = recvInfo[0];
			// size_t recv_nbEle = recvInfo[1];
			
			// 发送压缩的数据
			cudaMalloc(&d_rgb_rbuffer, recvInfo[1] * sizeof(float));

			MPI_Sendrecv(d_cmpBytes, outSize, MPI_UNSIGNED_CHAR, this->plan[u].pid, this->Processor_ID,
						 receivedCompressedBytes, recvInfo[0], MPI_UNSIGNED_CHAR, this->plan[u].pid, this->plan[u].pid, MPI_COMM_WORLD, &Processor_status);
			
			
			
			MPI_Barrier(MPI_COMM_WORLD);
			decompress_time_start = MPI_Wtime();
			SZp_decompress_deviceptr_f32(d_rgb_rbuffer, receivedCompressedBytes, recvInfo[1], recvInfo[0], errorBound, stream);
			getLastCudaError("SZp decompress failed");
			// cudaMemcpy(d_rgb_rbuffer, d_decData, recv_nbEle * sizeof(float), cudaMemcpyDeviceToDevice);
			MPI_Barrier(MPI_COMM_WORLD);
			decompress_time_end = MPI_Wtime();
			decompress_time = (decompress_time_end - decompress_time_start) * 1000.0f;
			
			

			tmpRecvCound = recvInfo[1];

			rgb_totalSentBytes += outSize;


			// obr_rgb_a = ra; obr_rgb_b = rb;//更新图像起始和结束位置
			obr_rgb_a = oa; obr_rgb_b = ob;//更新图像起始和结束位置
			arae_a = ra; area_b = rb;
			
			// this->compositngColorRRGGBB(u);
			dim3 blockDim(16, 16);
			dim3 gridDim((obr_rgb_b.x - obr_rgb_a.x + 1 + blockDim.x - 1) / blockDim.x, 
						(obr_rgb_b.y - obr_rgb_a.y + 1 + blockDim.y - 1) / blockDim.y);

			compositingColorRRGGBBKernel<<<gridDim, blockDim>>>(d_obr_rgb, d_rgb_rbuffer, d_alpha_values_u, 
																obr_rgb_a, obr_rgb_b, obr_x, obr_y, plan[u].over, u);
			cudaDeviceSynchronize();

		}
		

		u++;
		MPI_Barrier(MPI_COMM_WORLD);
		each_round_time_end = MPI_Wtime();
		each_round_time = (each_round_time_end - each_round_time_start) * 1000.0f;
		
		// std::cout << "[binarySwap_RGB]:: PID " << Processor_ID << " Finished round [ " << u << " ] " << std::endl;
		if (Processor_ID == 0)
		{
			Utils::recordBSETime(save_time_file_path.c_str(),  u, each_round_time, compress_time, decompress_time);
		}
		
	}

	double gather_time_s, gather_time_e, gather_time_total;
	double before_gather_s, before_gather_e, before_gather_total;
	double after_gather_s, after_gather_e, after_gather_total;
	// obr_rgb = new float[obr_x * obr_y * 3];
	// cudaMemcpy(obr_rgb, d_obr_rgb, obr_x * obr_y * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	gather_time_s = MPI_Wtime();
// #if GPU_WAY == 1
// 	// 方法二
// 	/* PART II - Final Image Gathering */
// 	// 计算缓冲区大小
// 	obr_rgb_a.x = arae_a.x; obr_rgb_a.y = arae_a.y;
// 	obr_rgb_b.x = area_b.x; obr_rgb_b.y = area_b.y;
// 	int sendWidth = std::abs(obr_rgb_b.x - obr_rgb_a.x) + 1;
// 	int sendHeight = std::abs(obr_rgb_b.y - obr_rgb_a.y) + 1;
// 	int bufsize = sendWidth * sendHeight * 3;
// 	bufsize += 4;
// 	float* fbuffer = new float[bufsize];// 接收数据的缓冲区
	
// 	// R, G, B 通道在 obr_rgb 中的起始偏移量
// 	int totalPixels = obr_x * obr_y;
// 	int rOffset_obr = 0;                      // R 通道从 0 开始
// 	int gOffset_obr = totalPixels;            // G 通道从 totalPixels 开始
// 	int bOffset_obr = totalPixels * 2;        // B 通道从 2 * totalPixels 开始 

// 	if (Processor_ID == 0)
// 	{
// 		for (u = 1; u < this->Processor_Size; u++)
// 		{
			
// 			MPI_Recv(&fbuffer[0], bufsize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Processor_status);
// 			Point2Di fa, fb;
// 			fa.x = (int)fbuffer[0]; fa.y = (int)fbuffer[1]; fb.x = (int)fbuffer[2]; fb.y = (int)fbuffer[3];
			
// 			int index = 4;

// 			for (int j = fa.y; j <= fb.y; j++)
// 			{
// 				for (int i = fa.x; i <= fb.x; i++)
// 				{
// 					int pixelIndex = (j * obr_x + i) * 1;
// 					float r = fbuffer[index++];
// 					float g = fbuffer[index++];
// 					float b = fbuffer[index++];

// 					// 将解压后的 R, G, B 数据存储到 obr_rgb 中
// 					obr_rgb[rOffset_obr + pixelIndex] = r;
// 					obr_rgb[gOffset_obr + pixelIndex] = g;
// 					obr_rgb[bOffset_obr + pixelIndex] = b;
// 				}
// 			}

// 			rgba_totalSentBytes += bufsize * sizeof(float);
// 		}

// 		obr_a.x = 0; obr_a.y = 0;
// 		obr_b.x = obr_x ; obr_b.y = obr_y;
// 	}
// 	else
// 	{
		
// 		// 所有进程填充自己的 fbuffer 数据，包括进程 0
// 		fbuffer[0] = obr_rgb_a.x; 
// 		fbuffer[1] = obr_rgb_a.y; 
// 		fbuffer[2] = obr_rgb_b.x; 
// 		fbuffer[3] = obr_rgb_b.y;

// 		int index = 4; // 假设 fbuffer 的前4个字节是一些元数据
// 		for (int j = obr_rgb_a.y; j <= obr_rgb_b.y; j++) 
// 		{
// 			for (int i = obr_rgb_a.x; i <= obr_rgb_b.x; i++) 
// 			{
// 				int pixelIndex = (j * obr_x + i) * 1;

// 				// 从 color 数组中读取 RGB 分量
// 				float r_float = obr_rgb[rOffset_obr + pixelIndex];
// 				float g_float = obr_rgb[gOffset_obr + pixelIndex];
// 				float b_float = obr_rgb[bOffset_obr + pixelIndex];

// 				// 将RGB分量存储到 fbuffer 中
// 				fbuffer[index++] = r_float;
// 				fbuffer[index++] = g_float;
// 				fbuffer[index++] = b_float;
// 			}
// 		}
		
// 		MPI_Send(&fbuffer[0], bufsize, MPI_FLOAT, 0, this->Processor_ID, MPI_COMM_WORLD);
// 	}
// #endif 

#if USE_GATHER == 1
	/* PART II - Final Image Gathering */
	// 计算缓冲区大小

	MPI_Barrier(MPI_COMM_WORLD);
	before_gather_s = MPI_Wtime();
	obr_rgb_a.x = arae_a.x; obr_rgb_a.y = arae_a.y;
	obr_rgb_b.x = area_b.x; obr_rgb_b.y = area_b.y;
	uint64_t sendWidth = std::abs(obr_rgb_b.x - obr_rgb_a.x) + 1;
	uint64_t sendHeight = std::abs(obr_rgb_b.y - obr_rgb_a.y) + 1;
	uint64_t bufsize = sendWidth * sendHeight * 3;
	bufsize += 4;
	float* d_fbuffer = nullptr;
	cudaMalloc(&d_fbuffer, bufsize * sizeof(float));
	// 进程 0 分配接收缓冲区
	float* d_recvbuf = nullptr;
	int* recvcounts = nullptr;
	int* displs = nullptr;
	if (Processor_ID == 0) 
	{
		// 为每个进程分配接收缓冲区
		recvcounts = new int[Processor_Size];
		displs = new int[Processor_Size];
	
		// 设置每个进程的 recvcounts 和 displs
		for (int i = 0; i < Processor_Size; i++) 
		{
			recvcounts[i] = bufsize;
			displs[i] = i * bufsize; // 每个进程的数据按顺序排列
			// std::cout << "[GATHER]:: PID " << Processor_ID << " recvcounts " << recvcounts[i] << " displs " << displs[i] << std::endl;
		}
	
		// 分配总接收缓冲区
		cudaMalloc(&d_recvbuf, Processor_Size * bufsize * sizeof(float));
	}
	
	// R, G, B 通道在 obr_rgb 中的起始偏移量
	int totalPixels = obr_x * obr_y;
	int rOffset_obr = 0;                      // R 通道从 0 开始
	int gOffset_obr = totalPixels;            // G 通道从 totalPixels 开始
	int bOffset_obr = totalPixels * 2;        // B 通道从 2 * totalPixels 开始 
	
	int total_fbuffer_Pixels = (obr_rgb_b.y - obr_rgb_a.y + 1) * (obr_rgb_b.x - obr_rgb_a.x + 1);
	int rOffset_fbuffer = 4;                      			// R 通道从 4 开始
	int gOffset_fbuffer = 4 + total_fbuffer_Pixels;   		// G 通道从 total_fbuffer_Pixels 开始
	int bOffset_fbuffer = 4 + total_fbuffer_Pixels * 2; 	// B 通道从 2 * total_fbuffer_Pixels 开始

	// 所有进程填充自己的 fbuffer 数据，包括进程 0
	dim3 blockDim(16, 16);
	dim3 gridDim((obr_rgb_b.x - obr_rgb_a.x + 1 + blockDim.x - 1) / blockDim.x, 
				(obr_rgb_b.y - obr_rgb_a.y + 1 + blockDim.y - 1) / blockDim.y);
				
	Point2Di fa, fb;
	cudaMemcpy(&d_fbuffer[0], &obr_rgb_a.x, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_fbuffer[1], &obr_rgb_a.y, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_fbuffer[2], &obr_rgb_b.x, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_fbuffer[3], &obr_rgb_b.y, sizeof(float), cudaMemcpyHostToDevice);
	fa.x = obr_rgb_a.x; fa.y = obr_rgb_a.y; 
	fb.x = obr_rgb_b.x; fb.y = obr_rgb_b.y;
	gather_fillRGBBuffer<<<gridDim, blockDim>>>(d_obr_rgb, d_fbuffer, obr_x, obr_y, 
				rOffset_obr, gOffset_obr, bOffset_obr,
				rOffset_fbuffer, gOffset_fbuffer, bOffset_fbuffer, fa, fb);
	cudaDeviceSynchronize();
	getLastCudaError("gather_fillRGBBuffer failed");
	
	
	MPI_Barrier(MPI_COMM_WORLD);
	before_gather_e = MPI_Wtime();
	before_gather_total = 0.0f;
	before_gather_total = (before_gather_e - before_gather_s) * 1000.0f;
	if (Processor_ID == 0)
	{
		std::cout << "before gather Time: " << before_gather_total << std::endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	after_gather_s = MPI_Wtime();

	// 使用 MPI_Gatherv 收集每个进程的数据到进程 0
	MPI_Gatherv(d_fbuffer, bufsize, MPI_FLOAT, d_recvbuf, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	float* d_sub_buffer = nullptr;
	if (Processor_ID == 0)
	{
		for (u = 1; u < this->Processor_Size; u++)
		{
			int offset = displs[u];
			float* d_fbufferRecv = nullptr;
			cudaMalloc(&d_fbufferRecv, bufsize * sizeof(float));
			cudaMemcpy(d_fbufferRecv, d_recvbuf + offset, bufsize * sizeof(float), cudaMemcpyDeviceToDevice);
			Point2Di fa, fb;
			cudaMemcpy(&fa.x, &d_recvbuf[offset + 0], sizeof(float), cudaMemcpyDeviceToHost);
    		cudaMemcpy(&fa.y, &d_recvbuf[offset + 1], sizeof(float), cudaMemcpyDeviceToHost);
    		cudaMemcpy(&fb.x, &d_recvbuf[offset + 2], sizeof(float), cudaMemcpyDeviceToHost);
    		cudaMemcpy(&fb.y, &d_recvbuf[offset + 3], sizeof(float), cudaMemcpyDeviceToHost);
			// std::cout << "fa " << fa.x << " " << fa.y << " fb " << fb.x << " " << fb.y << std::endl;
			blockDim = dim3(16, 16);
			gridDim = dim3((fb.x - fa.x + 1 + blockDim.x - 1) / blockDim.x, 
						(fb.y - fa.y + 1 + blockDim.y - 1) / blockDim.y);
			gather_merge<<<gridDim, blockDim>>>(d_fbufferRecv, d_obr_rgb, rOffset_obr, gOffset_obr, bOffset_obr, fa, fb, obr_x);
			cudaDeviceSynchronize();
			getLastCudaError("gather_merge failed");

			rgba_totalSentBytes += bufsize * sizeof(float);
		}

	}
#endif
	MPI_Barrier(MPI_COMM_WORLD);
	after_gather_e = MPI_Wtime();
	after_gather_total = 0.0f;
	after_gather_total = (after_gather_e - after_gather_s) * 1000.0f;
	if (Processor_ID == 0)
	{
		std::cout << "after gather Time: " << after_gather_total << std::endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	gather_time_e = MPI_Wtime();
	gather_time_total = 0.0f;
	gather_time_total = (gather_time_e - gather_time_s) * 1000.0f;
	if (Processor_ID == 0)
	{
		Utils::recordCudaRenderTime(save_time_file_path.c_str(),  "gather Time:", std::to_string(Processor_Size - 1), gather_time_total);
	}
	
	std::cout << "PID [ " << Processor_ID << " ] finished IMAGE COMPOSITING " << std::endl;

	obr_rgb = new float[obr_x * obr_y * 3];
	cudaMemcpy(obr_rgb, d_obr_rgb, obr_x * obr_y * 3 * sizeof(float), cudaMemcpyDeviceToHost);
// #if GATHERING_IMAGE == 0
// 	// part II final image gathering
// 	// 计算缓冲区大小
// 	obr_rgb_a.x = arae_a.x; obr_rgb_a.y = arae_a.y;
// 	obr_rgb_b.x = area_b.x; obr_rgb_b.y = area_b.y;
// 	int sendWidth = std::abs(obr_rgb_b.x - obr_rgb_a.x) + 1;
// 	int sendHeight = std::abs(obr_rgb_b.y - obr_rgb_a.y) + 1;
// 	int bufsize = sendWidth * sendHeight * 3;
// 	bufsize += 4;
// 	float* fbuffer = new float[bufsize];// 接收数据的缓冲区



// 	// R, G, B 通道在 obr_rgb 中的起始偏移量
// 	int totalPixels = obr_x * obr_y;
// 	int rOffset_obr = 0;                      // R 通道从 0 开始
// 	int gOffset_obr = totalPixels;            // G 通道从 totalPixels 开始
// 	int bOffset_obr = totalPixels * 2;        // B 通道从 2 * totalPixels 开始 

// 	// 所有进程填充自己的 fbuffer 数据，包括进程 0
// 	fbuffer[0] = obr_rgb_a.x; 
// 	fbuffer[1] = obr_rgb_a.y; 
// 	fbuffer[2] = obr_rgb_b.x; 
// 	fbuffer[3] = obr_rgb_b.y;

// 	int index = 4; // 假设 fbuffer 的前4个字节是一些元数据
// 	for (int j = obr_rgb_a.y; j <= obr_rgb_b.y; j++) {
// 		for (int i = obr_rgb_a.x; i <= obr_rgb_b.x; i++) {
// 			int pixelIndex = (j * obr_x + i) * 1;

// 			// 从 color 数组中读取 RGB 分量
// 			float r_float = obr_rgb[rOffset_obr + pixelIndex];
// 			float g_float = obr_rgb[gOffset_obr + pixelIndex];
// 			float b_float = obr_rgb[bOffset_obr + pixelIndex];

// 			// 将RGB分量存储到 fbuffer 中
// 			fbuffer[index++] = r_float;
// 			fbuffer[index++] = g_float;
// 			fbuffer[index++] = b_float;
// 		}
// 	}

// 	// 进程 0 分配接收缓冲区
// 	int* sendcounts = new int[Processor_Size];
// 	int* displs = new int[Processor_Size];
// 	// 每个进程将自己的 bufsize 发送给根进程
// 	MPI_Gather(&bufsize, 1, MPI_INT, sendcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

// 	float* recvbuf = nullptr;
	
// 	if (Processor_ID == 0) 
// 	{
// 		displs[0] = 0;
// 		for (int i = 1; i < Processor_Size; i++)
// 		{
// 			displs[i] = displs[i - 1] + sendcounts[i - 1];
// 		}
// 		recvbuf = new float[displs[Processor_Size - 1] + sendcounts[Processor_Size - 1]];
// 	}
	
// 	if(1)
// 	{
// 		// 压缩数据
// 		size_t outSize;
// 		size_t nbEle = bufsize - 4;
// 		unsigned char* compressedData = SZx_fast_compress_args(SZx_WITH_BLOCK_FAST_CMPR, SZx_FLOAT, fbuffer + 4, &outSize, REL, gather_error, 0.001, 0, 0, 0, 0, 0, 0, nbEle);

// 		// 定义 MetaData 结构体
// 		struct MetaData
// 		{
// 			int fa_x, fa_y, fb_x, fb_y;  // 数据范围
// 			size_t outSize;              // 压缩后的大小
// 			size_t nbEle;                // 原始数据元素个数
// 		};

// 		// 创建并发送元数据
// 		MetaData meta = {static_cast<int>(obr_rgb_a.x), static_cast<int>(obr_rgb_a.y),
// 						static_cast<int>(obr_rgb_b.x), static_cast<int>(obr_rgb_b.y), outSize, nbEle};

// 		MetaData* recvMeta = nullptr;
// 		if (Processor_ID == 0)
// 		{
// 			recvMeta = new MetaData[Processor_Size];  // 接收所有进程的元数据
// 		}

// 		// 使用 MPI_Gather 收集元数据
// 		MPI_Gather(&meta, sizeof(MetaData), MPI_BYTE, recvMeta, sizeof(MetaData), MPI_BYTE, 0, MPI_COMM_WORLD);

// 		unsigned char* recvbuf_compressed = nullptr;
// 		int* recvOutSizes = nullptr;
// 		int* recvnbEleSize = nullptr;
// 		int* displs = nullptr;
// 		// 进程 0 处理元数据，准备进行 MPI_Gatherv
// 		if (Processor_ID == 0)
// 		{
// 			// 处理元数据：提取每个进程的 outSize 和 nbEle，并计算总的接收大小
// 			recvOutSizes = new int[Processor_Size];  // 每个进程的压缩数据大小 outSize
// 			recvnbEleSize = new int[Processor_Size]; // 每个进程原始数据的元素个数 nbEle
// 			displs = new int[Processor_Size];        // 偏移量

// 			size_t totalCompressedSize = 0;
// 			displs[0] = 0;  // 初始化第一个偏移量

// 			// 处理元数据并计算总压缩大小和偏移量
// 			for (int i = 0; i < Processor_Size; i++)
// 			{
// 				recvOutSizes[i] = static_cast<int>(recvMeta[i].outSize);
// 				recvnbEleSize[i] = static_cast<int>(recvMeta[i].nbEle);
// 				totalCompressedSize += recvOutSizes[i];  // 计算所有压缩数据的总大小

// 				if (i > 0)
// 				{
// 					displs[i] = displs[i - 1] + recvOutSizes[i - 1];
// 				}
// 				// std::cout << "[DOTEST===] " << i << " outSize " << recvOutSizes[i] << " nbELeSize " << recvnbEleSize[i] << std::endl;
// 			}

// 			// 为进程 0 分配接收压缩数据的缓冲区
// 			recvbuf_compressed = new unsigned char[totalCompressedSize];
// 		}

// 		// 使用 MPI_Gatherv 收集压缩数据
// 		// std::cout << "[==========================] before" << std::endl;
// 		MPI_Gatherv(compressedData, static_cast<int>(outSize), MPI_UNSIGNED_CHAR, recvbuf_compressed, recvOutSizes, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
// 		// std::cout << "[==========================] after" << std::endl;
// 		if(Processor_ID == 0)
// 		{
// 			// 进程 0 解压和合成数据
// 			for (int p = 1; p < Processor_Size; p++)
// 			{
// 				// 获取压缩数据和元素个数
// 				unsigned char* recvCompressedData = recvbuf_compressed + displs[p];
// 				size_t recv_outSize = static_cast<size_t>(recvOutSizes[p]);
// 				size_t recv_nbEle = static_cast<size_t>(recvnbEleSize[p]);
// 				rgb_totalSentBytes += sendcounts[p] * sizeof(float);
// 				// 打印调试信息，确保 recvCompressedData 和 recvOutSizes 正确
// 				// std::cout << "[Process " << p << "] Compressed data size: " << recvOutSizes[p] 
// 				// 		<< ", Element count: " << recv_nbEle << std::endl;

// 				// 检查recvCompressedData 和 recvOutSizes[p] 是否有效
// 				if (recvCompressedData == nullptr || recvOutSizes[p] == 0) {
// 					// std::cerr << "Error: Invalid compressed data or size for process " << p << std::endl;
// 					continue; // 跳过错误的进程
// 				}
// 				// MPI_Barrier(MPI_COMM_WORLD);
// 				gather_d_time_start = MPI_Wtime();
// 				// 解压每个进程的数据
// 				// std::cout << "[2222222]:: " << p << "before decompress" << std::endl;
// 				float* decompressedData = (float*)SZx_fast_decompress(SZx_WITH_BLOCK_FAST_CMPR, SZx_FLOAT, recvCompressedData, recv_outSize, 0, 0, 0, 0, recv_nbEle);
// 				// MPI_Barrier(MPI_COMM_WORLD);
// 				gather_d_time_end = MPI_Wtime();
// 				// std::cout << "[2222222]:: " << p << "finished decompress" << std::endl;
// 				// 使用解压后的数据处理 RGB 数据
// 				MetaData& meta = recvMeta[p];
// 				Point2Di fa, fb;
// 				fa.x = meta.fa_x;
// 				fa.y = meta.fa_y;
// 				fb.x = meta.fb_x;
// 				fb.y = meta.fb_y;

// 				int index = 0;  // 解压后的数据从第0项开始
// 				for (int j = fa.y; j <= fb.y; j++)
// 				{
// 					for (int i = fa.x; i <= fb.x; i++)
// 					{
// 						int pixelIndex = (j * obr_x + i) * 1;
// 						float r = decompressedData[index++];
// 						float g = decompressedData[index++];
// 						float b = decompressedData[index++];

// 						// 将解压后的 R, G, B 数据存储到 obr_rgb 中
// 						obr_rgb[rOffset_obr + pixelIndex] = r;
// 						obr_rgb[gOffset_obr + pixelIndex] = g;
// 						obr_rgb[bOffset_obr + pixelIndex] = b;
// 					}
// 				}

				

// 				gather_d_time += (gather_d_time_end - gather_d_time_start) * 1000.0f;
// 				if (p == Processor_Size - 1)
// 				{
// 					Utils::recordCudaRenderTime(save_time_file_path.c_str(),  "gather decompress Time:", std::to_string(Processor_Size - 1), gather_d_time);
// 				}

// 				// 清理解压后的数据
// 				delete[] decompressedData;
// 			}

// 			// 清理内存
// 			delete[] recvOutSizes;
// 			delete[] recvnbEleSize;
// 			delete[] displs;
// 			delete[] recvbuf_compressed;
// 			delete[] recvMeta;
// 		}

// 		// 清理压缩数据
// 		delete[] compressedData;
	
// 	}
	
// #endif

	
	this->reset();

}


void Processor::send_data(unsigned char* buf, int pocet, int dest, int tag)
{
	const int buf_size = 1024;
	unsigned char send_buffer[buf_size];

	int n;//发送/接收缓冲区大小
	if (pocet % buf_size == 0)
		n = pocet / buf_size;
	else
		n = (pocet / buf_size) + 1;

	int send_size;
	for (int i = 0; i < n; i++)
	{
		send_size = buf_size;
		for (int index = 0; index < buf_size; index++)
		{
			if (i * buf_size + index + 1 > pocet)
			{
				send_size = index;
				break;
			} // 如果已经传递了所有数据，停止
			send_buffer[index] = buf[i * buf_size + index];
		}
		// 缓冲区准备好，可以发送

		MPI_Send(send_buffer, send_size, MPI_UNSIGNED_CHAR, dest, tag, MPI_COMM_WORLD);
	}

}

void Processor::recv_data(unsigned char* buf, int pocet, int source, int tag, int& prijate)
{
	const int buf_size = 1024; // 缓冲区大小
	unsigned char send_buffer[buf_size]; // 缓冲区setDimensions

	int n; // 循环次数
	if (pocet % buf_size == 0)
		n = pocet / buf_size;
	else
		n = (pocet / buf_size) + 1;
	int recv;
	MPI_Status status;
	prijate = 0;
	for (int i = 0; i < n; i++)
	{
		MPI_Recv(send_buffer, buf_size, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &recv);
		// 从缓冲区写入大数组
		prijate += recv;
		//std::cout << "Received " << recv << " bytes from process " << odkial << " with tag " << tag << std::endl;
		for (int index = 0; index < recv; index++)
		{
			buf[i * buf_size + index] = send_buffer[index];
		}
	}
}

void Processor::setDimensions(const int u, const Point2Di& a, const Point2Di& b, Point2Di& sa, Point2Di& sb, Point2Di& ra, Point2Di& rb)
{
	ra = sa = a; rb = sb = b;
	if ((u % 2) == 0)
	{
		if (plan[u].over)
		{

			ra.x = a.x + ((b.x - a.x + 1) / 2);
			sb.x = a.x + ((b.x - a.x + 1) / 2) - 1;
			//std::cout << "[setDimensions]:: PID " << Processor_ID << " " << u << " over: " << plan[u].over << " ra : [ " << ra.x << " " << ra.y << " ] rb : [ " << rb.x << " " << rb.y << " ]" <<
			//	" sa : [ " << sa.x << " " << sa.y << " ] sb : [ " << sb.x << " " << sb.y << " ]" << std::endl;
		}
		else
		{

			sa.x = a.x + ((b.x - a.x + 1) / 2);
			rb.x = a.x + ((b.x - a.x + 1) / 2) - 1;
			//std::cout << "[setDimensions]:: PID " << Processor_ID << " " << u << " over: " << plan[u].over << " ra : [ " << ra.x << " " << ra.y << " ] rb : [ " << rb.x << " " << rb.y << " ]" <<
			//	" sa : [ " << sa.x << " " << sa.y << " ] sb : [ " << sb.x << " " << sb.y << " ]" << std::endl;
		}
	}
	else
	{
		if (plan[u].over)
		{
			rb.y = a.y + ((b.y - a.y + 1) / 2) - 1;
			sa.y = a.y + ((b.y - a.y + 1) / 2);
			//std::cout << "[setDimensions]:: PID " << Processor_ID << " " << u << " over: " << plan[u].over << " ra : [ " << ra.x << " " << ra.y << " ] rb : [ " << rb.x << " " << rb.y << " ]" <<
			//	" sa : [ " << sa.x << " " << sa.y << " ] sb : [ " << sb.x << " " << sb.y << " ]" << std::endl;
		}
		else
		{
			sb.y = a.y + ((b.y - a.y + 1) / 2) - 1;
			ra.y = a.y + ((b.y - a.y + 1) / 2);
			//std::cout << "[setDimensions]:: PID " << Processor_ID << " " << u << " over: " << plan[u].over << " ra : [ " << ra.x << " " << ra.y << " ] rb : [ " << rb.x << " " << rb.y << " ]" <<
			//	" sa : [ " << sa.x << " " << sa.y << " ] sb : [ " << sb.x << " " << sb.y << " ]" << std::endl;
		}
	}
}

void Processor::loadBuffer(const Point2Di& sa, const Point2Di& sb)
{
	int index = 0;
	for (int j = sa.y; j <= sb.y; j++)
	{
		for (int i = sa.x; i <= sb.x; i++)
		{
			// Assuming row-major order and 4 floats per pixel (RGBA)
			int pixelIndex = (j * obr_x + i) * 4;

			// Extracting color components from the float array
			float r = obr[pixelIndex + 0];
			float g = obr[pixelIndex + 1];
			float b = obr[pixelIndex + 2];
			float a = obr[pixelIndex + 3];

			sbuffer[index++] = r;
			sbuffer[index++] = g;
			sbuffer[index++] = b;
			sbuffer[index++] = a;
		}
	}
}

void Processor::loadColorBuffer(const Point2Di& sa, const Point2Di& sb)
{
	int index = 0;
	for (int j = sa.y; j <= sb.y; j++)
	{
		for (int i = sa.x; i <= sb.x; i++)
		{
			// Assuming row-major order and 4 floats per pixel (RGBA)
			int pixelIndex = (j * obr_x + i) * 3;

			// Extracting color components from the float array
			float r = obr_rgb[pixelIndex + 0];
			float g = obr_rgb[pixelIndex + 1];
			float b = obr_rgb[pixelIndex + 2];


			rgb_sbuffer[index++] = r;
			rgb_sbuffer[index++] = g;
			rgb_sbuffer[index++] = b;
		}
	}
}

void Processor::loadColorBufferRRGGBB(const Point2Di& sa, const Point2Di& sb)
{
	// R, G, B 通道在 rgb_sbuffer 中的起始位置
	int totalPixels = (std::abs(sb.x - sa.x) + 1) * (std::abs(sb.y - sa.y) + 1);
    int rOffset_sbuffer = 0;                      // R 通道从 0 开始
    int gOffset_sbuffer = totalPixels;            // G 通道从 totalPixels 开始
    int bOffset_sbuffer = totalPixels * 2;        // B 通道从 2 * totalPixels 开始

	// R, G, B 通道在 obr_rgb 中的起始偏移量
    int totalImagePixels = obr_x * obr_y;
    int rOffset_obr = 0;                          // R 通道从 0 开始
    int gOffset_obr = totalImagePixels;           // G 通道从 totalImagePixels 开始
    int bOffset_obr = totalImagePixels * 2;       // B 通道从 2 * totalImagePixels 开始

	int index = 0;
	for (int j = sa.y; j <= sb.y; j++)
	{
		for (int i = sa.x; i <= sb.x; i++)
		{
			// 计算当前像素的索引
			int pixelIndex = (j * obr_x + i) * 1;
			
			// Extracting color components from the float array
			float r = obr_rgb[rOffset_obr + pixelIndex];
			float g = obr_rgb[gOffset_obr + pixelIndex];
			float b = obr_rgb[bOffset_obr + pixelIndex];


			// 将 R, G, B 分别存储到 rgb_sbuffer 中的分段区域
            rgb_sbuffer[rOffset_sbuffer + index] = r;
            rgb_sbuffer[gOffset_sbuffer + index] = g;
            rgb_sbuffer[bOffset_sbuffer + index] = b;
			index++;
		}
	}
}

void Processor::compositng(const int u)
{
	int index = 0;

	for (int y = obr_a.y; y <= obr_b.y; y++)
	{
		for (int x = obr_a.x; x <= obr_b.x; x++)
		{
			int pixelIndex = (y * obr_x + x) * 4;

			// 从 obr 数组中读取 RGBA 分量
			float r_float = obr[pixelIndex + 0];
			float g_float = obr[pixelIndex + 1];
			float b_float = obr[pixelIndex + 2];
			float a_float = obr[pixelIndex + 3];

			if (!plan[u].over)
			{ // nase data su front
				// Convert to back-to-front
				r_float = rbuffer[index + 0] + r_float * (1.0f - rbuffer[index + 3]); // Red
				g_float = rbuffer[index + 1] + g_float * (1.0f - rbuffer[index + 3]); // Green
				b_float = rbuffer[index + 2] + b_float * (1.0f - rbuffer[index + 3]); // Blue
				a_float = rbuffer[index + 3] + a_float * (1.0f - rbuffer[index + 3]); // Alpha
			}
			else
			{ // prijate data su 'nad'
				// Convert to back-to-front
				r_float = r_float + rbuffer[index + 0] * (1.0f - a_float); // Red
				g_float = g_float + rbuffer[index + 1] * (1.0f - a_float); // Green
				b_float = b_float + rbuffer[index + 2] * (1.0f - a_float); // Blue
				a_float = a_float + rbuffer[index + 3] * (1.0f - a_float); // Alpha
			}

			// Combine the modified color components back into the unsigned int
			obr[pixelIndex + 0] = r_float;
			obr[pixelIndex + 1] = g_float;
			obr[pixelIndex + 2] = b_float;
			obr[pixelIndex + 3] = a_float;

			index += 4;
		}
	}
}

void Processor::visibleFunc(const int u)
{
	int index = 0;

	for (int y = obr_alpha_a.y; y <= obr_alpha_b.y; y++)
	{
		for (int x = obr_alpha_a.x; x <= obr_alpha_b.x; x++)
		{
			int pixelIndex = (y * obr_x + x) * 1;

			// 从 obr_alpha 数组中读取 A 分量
			float a_float = obr_alpha[pixelIndex + 0];

			if (!plan[u].over)
			{ // nase data su front
				// Convert to back-to-front
				alpha_values_u[u][y][x] = alpha_rbuffer[index + 0];
				a_float = alpha_rbuffer[index + 0] + a_float * (1.0f - alpha_rbuffer[index + 0]); // Alpha
				
			}
			else
			{ // prijate data su 'nad'
				// Convert to back-to-front
				alpha_values_u[u][y][x] = a_float;
				a_float = a_float + alpha_rbuffer[index + 0] * (1.0f - a_float); // Alpha
				
			}

			// Combine the modified color components back into the float
			obr_alpha[pixelIndex + 0] = a_float;

			index++;
		}
	}
}

void Processor::compositngColor(const int u)
{
	int index = 0;

	for (int y = obr_rgb_a.y; y <= obr_rgb_b.y; y++)
	{
		for (int x = obr_rgb_a.x; x <= obr_rgb_b.x; x++)
		{
			int pixelIndex = (y * obr_x + x) * 1;

			// 从 obr_rgb 数组中读取 rgb 分量
			float r_float = obr_rgb[pixelIndex * 3 + 0];
			float g_float = obr_rgb[pixelIndex * 3 + 1];
			float b_float = obr_rgb[pixelIndex * 3 + 2];

			// 使用完整合成后的 Alpha 值
			//float a_float = obr_alpha[pixelIndex];
			float a_float = alpha_values_u[u][y][x];
			if (!plan[u].over)
			{ // nase data su front
				// Convert to back-to-front
				r_float = rgb_rbuffer[index * 3 + 0] + r_float * (1.0f - a_float); // Red
				g_float = rgb_rbuffer[index * 3 + 1] + g_float * (1.0f - a_float); // Green
				b_float = rgb_rbuffer[index * 3 + 2] + b_float * (1.0f - a_float); // Blue
			}
			else
			{ // prijate data su 'nad'
				// Convert to back-to-front
				r_float = r_float + rgb_rbuffer[index * 3 + 0] * (1.0f - a_float); // Red
				g_float = g_float + rgb_rbuffer[index * 3 + 1] * (1.0f - a_float); // Green
				b_float = b_float + rgb_rbuffer[index * 3 + 2] * (1.0f - a_float); // Blue
			}

			// 将修改后的 Color 分量写回 color 数组
			obr_rgb[pixelIndex * 3 + 0] = r_float;
			obr_rgb[pixelIndex * 3 + 1] = g_float;
			obr_rgb[pixelIndex * 3 + 2] = b_float;

			index++;
		}
	}
}

void Processor::compositngColorRRGGBB(const int u)
{
	int index = 0;

	int totalPixels = obr_x * obr_y;

	// R, G, B 通道在 obr_rgb 中的起始偏移量
    int rOffset_obr = 0;                      // R 通道从 0 开始
    int gOffset_obr = totalPixels;            // G 通道从 totalPixels 开始
    int bOffset_obr = totalPixels * 2;        // B 通道从 2 * totalPixels 开始

	// R, G, B 通道在 rgb_sbuffer 中的起始偏移量
    int rOffset_sbuffer = 0;                     	 // R 通道从 0 开始
    int gOffset_sbuffer = tmpRecvCound / 3;          // G 通道从 rgb_sbuffer 中的1/3开始
    int bOffset_sbuffer = tmpRecvCound / 3 * 2;      // B 通道从 rgb_sbuffer 的2/3处开始
	index = 0;
	for (int y = obr_rgb_a.y; y <= obr_rgb_b.y; y++)
	{
		for (int x = obr_rgb_a.x; x <= obr_rgb_b.x; x++)
		{
			int pixelIndex = (y * obr_x + x) * 1;

			// 从 obr_rgb 数组中读取 rgb 分量
			float r_float = obr_rgb[rOffset_obr + pixelIndex];  // 从 R 通道提取
			float g_float = obr_rgb[gOffset_obr + pixelIndex];  // 从 G 通道提取
			float b_float = obr_rgb[bOffset_obr + pixelIndex];  // 从 B 通道提取


			// 从 rgb_rbuffer 中获取 R, G, B 分量
            float r_float_sbuffer = rgb_rbuffer[rOffset_sbuffer + index];  // 从 rgb_rbuffer 的 R 通道提取
            float g_float_sbuffer = rgb_rbuffer[gOffset_sbuffer + index];  // 从 rgb_rbuffer 的 G 通道提取
            float b_float_sbuffer = rgb_rbuffer[bOffset_sbuffer + index];  // 从 rgb_rbuffer 的 B 通道提取


			// 使用完整合成后的 Alpha 值
			float a_float = alpha_values_u[u][y][x];

			if (!plan[u].over)
			{ // nase data su front
				// Convert to back-to-front
				r_float = r_float_sbuffer + r_float * (1.0f - a_float); // Red
				g_float = g_float_sbuffer + g_float * (1.0f - a_float); // Green
				b_float = b_float_sbuffer + b_float * (1.0f - a_float); // Blue
			}
			else
			{ // prijate data su 'nad'
				// Convert to back-to-front
				r_float = r_float + r_float_sbuffer * (1.0f - a_float); // Red
				g_float = g_float + g_float_sbuffer * (1.0f - a_float); // Green
				b_float = b_float + b_float_sbuffer * (1.0f - a_float); // Blue
			}

			// 将修改后的 Color 分量写回 color 数组
			obr_rgb[rOffset_obr + pixelIndex] = r_float;
			obr_rgb[gOffset_obr + pixelIndex] = g_float;
			obr_rgb[bOffset_obr + pixelIndex] = b_float;

			index++;
		}
	}
}

void Processor::compositngColorRRGGBB(const int u, float* buffer)
{
	int index = 0;

	int totalPixels = obr_x * obr_y;

	// R, G, B 通道在 obr_rgb 中的起始偏移量
    int rOffset_obr = 0;                      // R 通道从 0 开始
    int gOffset_obr = totalPixels;            // G 通道从 totalPixels 开始
    int bOffset_obr = totalPixels * 2;        // B 通道从 2 * totalPixels 开始

	// R, G, B 通道在 rgb_sbuffer 中的起始偏移量
    int rOffset_sbuffer = 0;                     	 // R 通道从 0 开始
    int gOffset_sbuffer = tmpRecvCound / 3;          // G 通道从 rgb_sbuffer 中的1/3开始
    int bOffset_sbuffer = tmpRecvCound / 3 * 2;      // B 通道从 rgb_sbuffer 的2/3处开始
	index = 0;
	for (int y = obr_rgb_a.y; y <= obr_rgb_b.y; y++)
	{
		for (int x = obr_rgb_a.x; x <= obr_rgb_b.x; x++)
		{
			int pixelIndex = (y * obr_x + x) * 1;

			// 从 obr_rgb 数组中读取 rgb 分量
			float r_float = obr_rgb[rOffset_obr + pixelIndex];  // 从 R 通道提取
			float g_float = obr_rgb[gOffset_obr + pixelIndex];  // 从 G 通道提取
			float b_float = obr_rgb[bOffset_obr + pixelIndex];  // 从 B 通道提取


			// 从 buffer 中获取 R, G, B 分量
            float r_float_sbuffer = buffer[rOffset_sbuffer + index];  // 从 buffer 的 R 通道提取
            float g_float_sbuffer = buffer[gOffset_sbuffer + index];  // 从 buffer 的 G 通道提取
            float b_float_sbuffer = buffer[bOffset_sbuffer + index];  // 从 buffer 的 B 通道提取


			// 使用完整合成后的 Alpha 值
			float a_float = alpha_values_u[u][y][x];

			if (!plan[u].over)
			{ // nase data su front
				// Convert to back-to-front
				r_float = r_float_sbuffer + r_float * (1.0f - a_float); // Red
				g_float = g_float_sbuffer + g_float * (1.0f - a_float); // Green
				b_float = b_float_sbuffer + b_float * (1.0f - a_float); // Blue
			}
			else
			{ // prijate data su 'nad'
				// Convert to back-to-front
				r_float = r_float + r_float_sbuffer * (1.0f - a_float); // Red
				g_float = g_float + g_float_sbuffer * (1.0f - a_float); // Green
				b_float = b_float + b_float_sbuffer * (1.0f - a_float); // Blue
			}

			// 将修改后的 Color 分量写回 color 数组
			obr_rgb[rOffset_obr + pixelIndex] = r_float;
			obr_rgb[gOffset_obr + pixelIndex] = g_float;
			obr_rgb[bOffset_obr + pixelIndex] = b_float;

			index++;
		}
	}
}




void Processor::reset()
{
	obr_a.x = 0; obr_a.y = 0;
	obr_b.x = obr_x - 1; obr_b.y = obr_y - 1;

	obr_alpha_a.x = 0; obr_alpha_a.y = 0;
	obr_alpha_b.x = obr_x - 1; obr_alpha_b.y = obr_y - 1;

	obr_rgb_a.x = 0; obr_rgb_a.y = 0;
	obr_rgb_b.x = obr_x - 1; obr_rgb_b.y = obr_y - 1;
}

bool Processor::read_data(const std::string& s, float3& a, float3& b, cudaExtent volumeTotalSize)
{
	std::cout << "[init_master]::MASTER: GET DATA INFO "; std::cout.flush();

	//int err = 0;
	a.x = 0; a.y = 0; a.z = 0;
	b.x = volumeTotalSize.width - 1; b.y = volumeTotalSize.height - 1; b.z = volumeTotalSize.depth - 1;
	std::cout << "OK" << std::endl; std::cout.flush();
	return true;
}

void Processor::createPlan(int ID, int depth, Node* n, const float3& v, Plan*& out)
{
	if (depth == 0)
	{
		printf("one processor. no communication.\n");
		return;
	}

	out = new Plan[depth];
	//对于树的每一层，从最低层开始，计算组成计划（处理器 ID、上方 / 下方）
	for (int i = 0; i < depth; i++)
	{
		int neighborhood = (int)pow(2, i); // 对于每一级，处理器距离 2^level
		Node* t = n;
		for (int j = 0; j < depth - i - 1; j++)
		{// we have 2 options, the correct one is determined by traversing the k-D tree
			if (ID - neighborhood >= t->i1)
			{// if both options are positive
				if ((inInterv(ID - neighborhood, t->back->i1, t->back->i2)) && (inInterv(ID, t->back->i1, t->back->i2)))
					t = t->back;
				else
					t = t->front;
			}
			else
			{// if one option is negative, test only the positive one
				if ((inInterv(ID + neighborhood, t->back->i1, t->back->i2)) && (inInterv(ID, t->back->i1, t->back->i2)))
					t = t->back;
				else
					t = t->front;
			}
		}

		float3 normal; // determine if combining F-B or B-F
		switch (t->split) // calculate the normal for the given data division
		{
		case 0: normal = make_float3(0, 1, 0); break;
		case 1: normal = make_float3(1, 0, 0); break;
		case 2: normal = make_float3(0, 0, 1); break;
		}
		// determine which processor to communicate with
		if (inInterv(ID - neighborhood, t->i1, t->i2))
		{
			out[i].pid = ID - neighborhood;
		}
		else
		{
			out[i].pid = ID + neighborhood;
		}

		// determine the order in which to compose the images
		float s = dot(normal, v); // dot product
		if (s < 0.0f)// front-to-back order, i.e., front is front and vice versa
		{
			if (inInterv(ID, t->back->i1, t->back->i2))
				out[i].over = 0; // order
			else
				out[i].over = 1;
		}
		else// back-to-front
		{
			if (inInterv(ID, t->back->i1, t->back->i2))
				out[i].over = 1;
			else
				out[i].over = 0;
		}

	}
}

void Processor::master_load_data(float3 a, float3 b, std::string s)
{
	// 计算起始点和终止点的索引
	uint64_t startX = a.x;
	uint64_t startY = a.y;
	uint64_t startZ = a.z;

	uint64_t endX = b.x ;
	uint64_t endY = b.y;
	uint64_t endZ = b.z;
	// std::cout << "[master_load_data]::MASTER: [ ================= 1111111" << std::endl;
	uint64_t local_volume_size = (endX - startX + 1) * (endY - startY + 1) * (endZ - startZ + 1) * sizeof(unsigned char);
	// std::cout << "[master_load_data]::MASTER: [ ================= 2222222" << std::endl;
	uint64_t volume_size = (uint64_t)whole_data_len.x * (uint64_t)whole_data_len.y * (uint64_t)whole_data_len.z * sizeof(unsigned char);
	// std::cout << "[master_load_data]::MASTER: [ ================= 3333333" << std::endl;
	uint64_t subW, subH, subD;
	// std::cout << "[master_load_data]::MASTER: [ ================= bbbbbbbbb" << std::endl;
	// std::cout << "jdklfjlaf\n";
	//void* h_volume = FileManager::loadPartialRawFile2(s.c_str(), local_volume_size, startX, endX, startY, endY, startZ, endZ, volume_size);
	// std::cout << s.c_str() << " " << std::endl;
	// std::cout << (uint64_t)whole_data_len.x << " " << (uint64_t)whole_data_len.y << " " << (uint64_t)whole_data_len.z << " " << std::endl;
	// std::cout << startX << " " << endX << " " << startY << " " << endY << " " << startZ << " " << endZ << std::endl;
	// std::cout << "done =============\n"; std::cout << std::flush;
	void* h_volume = FileManager::loadRawFileSubVolume(s, (uint64_t)whole_data_len.x, (uint64_t)whole_data_len.y, (uint64_t)whole_data_len.z, startX, endX, startY, endY, startZ, endZ, subW, subH, subD);
	// std::cout << "[master_load_data]::MASTER: [ ================= 4444444" << std::endl;
	memcpy(this->data, h_volume, local_volume_size);
	// std::cout << "[master_load_data]::MASTER: [ ================= 55555" << std::endl;
}

void Processor::initKDTree()
{
	if (Processor_ID == 0) std::cout << "[initKDTree]::Matster:: [ " << this->Processor_ID << " ] Create KD Tree..... "; std::cout.flush();

	//计算 k - D 树的深度 hlbka，pocprc 表示进程的数量。树的深度取决于进程的数量和以2为底的对数。
	int depth = (int)(std::log(Processor_Size) / std::log(2));
	kdTree = new KDTree(a, b, depth, 0);
	if (depth == 0)
		printf(" [==] only one process, no communication. [==] ");
	else
		this->plan = new Plan[depth];

	if (Processor_ID == 0)
	{
		std::cout << " depth [ " << depth << " ] ok" << std::endl; std::cout.flush();
	}

	//std::cout << "[initKDTree]::PID[ " << Processor_ID << " ]" << a.x << " " << a.y << " " << a.z <<
	//	" " << b.x << " " << b.y << " " << b.z << std::endl;
	
}

void Processor::setCamera()
{
	float up_vector;
	if (cam_vz * cos(cam_dy) >= 0) up_vector = 1.0f;
	else up_vector = 01.0f;
	if (cam_vz < 0) up_vector *= -1.0f;

	float xpos = -1.0f * cam_vz * sin(cam_dx) * cos(cam_dy);
	float ypos = cam_vz * sin(cam_dy);
	float zpos = -1.0f * cam_vz * cos(cam_dx) * cos(cam_dy);
	float3 camPos = make_float3(xpos, ypos, zpos);
	float3 cameraLookAt = make_float3(0.0f, 0.0f, 0.0f);
	float3 cameraWorldUp = make_float3(0.0f, up_vector, 0.0f);

	camera = new Camera(camPos, cameraLookAt, cameraWorldUp, 1.0f, 0.3f, 30.0f);
}

void Processor::setCameraProperty(float cam_dx, float cam_dy, std::optional<float> cam_vz)
{
	this->cam_dx = cam_dx;
	this->cam_dy = cam_dy;
	if (cam_vz.has_value())
	{
		this->cam_vz = cam_vz.value();
	}

	float rozm_x = this->kdTree->root->b.x - this->kdTree->root->a.x + 1;
	float rozm_y = this->kdTree->root->b.y - this->kdTree->root->a.y + 1;
	float rozm_z = this->kdTree->root->b.z - this->kdTree->root->a.z + 1;

	updateCamera(rozm_x, rozm_y, rozm_z);
	setRatioUV();
	
}

void Processor::initImage(int w, int h)
{
	this->obr_x = w; this->obr_y = h;
	this->obr = new float[obr_x * obr_y * 4];

	// 设置图像的边界点
	obr_a.x = 0; obr_a.y = 0;
	obr_b.x = obr_x - 1; obr_b.y = obr_y - 1;

	arae_a.x = 0; arae_a.y = 0;
	area_b.x = obr_x - 1; area_b.y = obr_y - 1;

	obr_alpha_a.x = 0; obr_alpha_a.y = 0;
	obr_alpha_b.x = obr_x - 1; obr_alpha_b.y = obr_y - 1;

	obr_rgb_a.x = 0; obr_rgb_a.y = 0;
	obr_rgb_b.x = obr_x - 1; obr_rgb_b.y = obr_y - 1;

	//计算缓冲区的大小。最坏情况下，需要发送或接收图像的垂直一半 //TODO
	int size = (obr_x / 2 + 1) * (obr_y) * 4;
	sbuffer = new float[size];
	rbuffer = new float[size];
	cudaMalloc(&d_rgba_sbuffer, size * sizeof(float)); 
	cudaMalloc(&d_rgba_rbuffer, size * sizeof(float)); 

	int alpha_size = (obr_x / 2 + 1) * obr_y * 1;
	alpha_sbuffer = new float[alpha_size];
	alpha_rbuffer = new float[alpha_size];

	cudaMalloc(&d_alpha_sbuffer, alpha_size * sizeof(float)); 
	cudaMalloc(&d_alpha_rbuffer, alpha_size * sizeof(float)); 

	int rgb_size = (obr_x / 2 + 1) * obr_y * 3;
	rgb_sbuffer = new float[rgb_size];
	rgb_rbuffer = new float[rgb_size];
	cudaMalloc(&d_rgb_sbuffer, rgb_size * sizeof(float));
	cudaMalloc(&d_rgb_rbuffer, rgb_size * sizeof(float));

	
	int totalSize = kdTree->depth * obr_y * obr_x;
	cudaMalloc(&d_alpha_values_u, totalSize * sizeof(float));


	// init cuda compression stream
	cudaStreamCreate(&stream);
	getLastCudaError("init stream failed");
	cudaMalloc((void**)&d_cmpBytes, sizeof(float) * rgb_size); 
	getLastCudaError("cudaMalloc d_cmpBytes failed");
	cudaMalloc(&d_decData, rgb_size * sizeof(float));
	cudaMalloc(&receivedCompressedBytes, rgb_size * sizeof(float));
}


void Processor::initData(const char* filename)
{
	if (Processor_ID == 0)	std::cout << "[initData]::Matster:: [ " << this->Processor_ID << " ] Init DATA from [ " << filename << " ]" << std::endl;
	if (!this->kdTree->root) return;		// 如果k-D树没有初始化则退出

	int ds = data_size(); //获取每个进程要计算的 数据大小 // 更新了 a b ，大小用的是data_a data_b
	this->data = new unsigned char[ds];
	this->data_len = ds;
	// std::cout << "[init_data]:: PID [ " << Processor_ID << " ] data size " << ds <<
	// 	" [ " << a.x << " , " << a.y << " , " << a.z <<
	// 	" ] -> [ " << b.x << " , " << b.y << " , " << b.z << " ] " << " " <<
	// 	" [ " << data_a.x << " , " << data_a.y << " , " << data_a.z <<
	// 	" ] -> [ " << data_b.x << " , " << data_b.y << " , " << data_b.z << " ] " << std::endl;

	// std::cout << "[init_data]:: PID [ " << Processor_ID << " ] data size " << ds << std::endl;	
	
	//主节点逐步读取子数据并发送给从节点
	if (Processor_ID == 0)
	{
		float3 node_a, node_b; float3 node_compensation; 
		for (auto i = 1; i < this->Processor_Size; ++i)
		{
			Node* t = this->kdTree->root;       // 从根节点开始遍历
			while (t->back != NULL && t->front != NULL)
			{
				if (inInterv(i, t->front->i1, t->front->i2))
					t = t->front;
				else
					t = t->back;
			}
			// 找到k-D树的对应叶节点和子数据尺寸 TODO 
			node_a = t->data_a; node_b = t->data_b; node_compensation = t->compensation;
			// 发送大小
			int ds = (node_b.x - node_a.x + 1) *
				(node_b.y - node_a.y + 1) *
				(node_b.z - node_a.z + 1) *
				sizeof(unsigned char);

			this->data = new unsigned char[ds];
			// std::cout << "[init_data]:: PID [ " << Processor_ID << " ] data loading ===== " << ds << std::endl;	
			master_load_data(node_a, node_b, std::string(filename));

			// 使用 MPI_Send 发送 ds
			MPI_Send(&ds, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			//std::cout << "[init_data]:: MASTER: SEND DATA SIZE " << ds << std::endl;
			// 发生数据
			send_data(data, ds, i, i);
			//std::cout << "[init_data]:: MASTER: SEND DATA CONTENT" << std::endl;
			// 发送补偿
			float compensation_buf[3] = { node_compensation.x, node_compensation.y, node_compensation.z };
			
			MPI_Send(compensation_buf, 3, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			std::cout << "[init_data]:: MASTER: SEND DATA COMPENSATION" << std::endl;
		}
		// 主节点读取自己的数据
		Node* t = this->kdTree->root;
		while (t->back != NULL && t->front != NULL)
		{
			if (inInterv(0, t->front->i1, t->front->i2))
				t = t->front;
			else
				t = t->back;
		}
		node_a = t->data_a; node_b = t->data_b; node_compensation = t->compensation;
		int ds = (node_b.x - node_a.x + 1) *
			(node_b.y - node_a.y + 1) *
			(node_b.z - node_a.z + 1) *
			sizeof(unsigned char);

		this->data = new unsigned char[ds];

		master_load_data(node_a, node_b, std::string(filename));
		data_compensation = node_compensation;
		//std::cout << "init_data:: MASTER: LOADING PID [ 0 ] DATA" << std::endl;
	}
	else
	{
		MPI_Recv(&ds, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		this->data = new unsigned char[ds];
		//std::cout << "[init_data]:: PID [ " << Processor_ID << " ]  RECV DATA SIZE " << ds << std::endl;
		int prijate;
		recv_data(data, ds, 0, this->Processor_ID, prijate);
		std::cout << "[init_data]:: PID [ " << Processor_ID << " ]  RECV DATA CONTENT" << std::endl;
		float compensation_buf[3];
		MPI_Recv(compensation_buf, 3, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		data_compensation = make_float3(compensation_buf[0], compensation_buf[1], compensation_buf[2]);
		std::cout << "[init_data]:: PID [ " << Processor_ID << " ]  RECV DATA COMPENSATION" << std::endl;
	}

	//设置包围盒
	int3 r;
	r.x = kdTree->root->b.x - kdTree->root->a.x + 1;
	r.y = kdTree->root->b.y - kdTree->root->a.y + 1;
	r.z = kdTree->root->b.z - kdTree->root->a.z + 1;

	float dx = 1; 	float dy = 1; 	float dz = 1;
	float x = box_b.x - box_a.x + 1; float y = box_b.y - box_a.y + 1; float z = box_b.z - box_a.z + 1; // 要加载范围的
	float posunx = r.x / -2.0f + box_a.x;  float posuny = r.y / -2.0f + box_a.y;	float posunz = r.z / -2.0f + box_a.z;

	bMin.x = posunx; bMin.y = posuny; bMin.z = posunz;
	bMax.x = float((x - 1) * dx) + posunx; bMax.y = float((y - 1) * dy) + posuny; bMax.z = float((z - 1) * dz) + posunz;

	
	//float3 center;
	//center.x = r.x / 2.0f;
	//center.y = r.y / 2.0f;
	//center.z = r.z / 2.0f;

	//// 计算全局包围盒
	//float3 globalMin, globalMax;
	//globalMin.x = a.x - center.x;
	//globalMin.y = a.y - center.y;
	//globalMin.z = a.z - center.z;

	//globalMax.x = b.x - center.x;
	//globalMax.y = b.y - center.y;
	//globalMax.z = b.z - center.z;

	//bMin = globalMin;
	//bMax = globalMax;
	//if (Processor_Size == 2 && Processor_ID == 0)
	//{
	//	bMin.x = -16.0f; bMin.y = -16.0f; bMin.z = -16.0f;
	//	bMax.x = 15.0f; bMax.y = 0.0f; bMax.z = 15.0f;
	//}
	//else if (Processor_Size == 2 && Processor_ID == 1)
	//{
	//	bMin.x = -16.0f; bMin.y = 0.0f; bMin.z = -16.0f;
	//	bMax.x = 15.0f; bMax.y = 15.0f; bMax.z = 15.0f;
	//}

	//if (Processor_Size == 4 && Processor_ID == 0)
	//{
	//	bMin.x = -16.0f; bMin.y = -16.0f; bMin.z = -16.0f;
	//	bMax.x = 0.0f;  bMax.y = 0.0f;  bMax.z = 15.0f;
	//}
	//else if (Processor_Size == 4 && Processor_ID == 1)
	//{
	//	bMin.x = 0.0f; bMin.y = -16.0f; bMin.z = -16.0f;
	//	bMax.x = 15.0f;  bMax.y = 0.0f;  bMax.z = 15.0f;
	//}
	//else if (Processor_Size == 4 && Processor_ID == 2)
	//{
	//	bMin.x = -16.0f; bMin.y = 0.0f;  bMin.z = -16.0f;
	//	bMax.x = 0.0f;  bMax.y = 15.0f; bMax.z = 15.0f;
	//}
	//else if (Processor_Size == 4 && Processor_ID == 3)
	//{
	//	bMin.x = 0.0f; bMin.y = 0.0f;  bMin.z = -16.0f;
	//	bMax.x = 15.0f;  bMax.y = 15.0f; bMax.z = 15.0f;
	//}

	/*if (Processor_Size == 8 && Processor_ID == 0)
	{
		bMin.x = -16.0f; bMin.y = -16.0f; bMin.z = -16.0f;
		bMax.x = 0.0f;  bMax.y = 0.0f;  bMax.z = 0.0f;
	}
	else if (Processor_Size == 8 && Processor_ID == 1)
	{
		bMin.x =-16.0f; bMin.y = -16.0f; bMin.z = 0.0f;
		bMax.x = 0.0f;  bMax.y = 0.0f;  bMax.z = 15.0f;
	}
	else if (Processor_Size == 8 && Processor_ID == 2)
	{
		bMin.x = 0.0f; bMin.y = -16.0f;  bMin.z = -16.0f;
		bMax.x = 15.0f;  bMax.y = 0.0f; bMax.z = 0.0f;
	}
	else if (Processor_Size == 8 && Processor_ID == 3)
	{
		bMin.x = 0.0f; bMin.y = -16.0f;  bMin.z = 0.0f;
		bMax.x = 15.0f;  bMax.y = 0.0f; bMax.z = 15.0f;
	}
	else if (Processor_Size == 8 && Processor_ID == 4)
	{
		bMin.x = -16.0f; bMin.y = 0.0f;  bMin.z = -16.0f;
		bMax.x = 0.0f;  bMax.y = 15.0f; bMax.z = 0.0f;
	}
	else if (Processor_Size == 8 && Processor_ID == 5)
	{
		bMin.x = -16.0f; bMin.y = 0.0f;  bMin.z = 0.0f;
		bMax.x = 0.0f;  bMax.y = 15.0f; bMax.z = 15.0f;
	}
	else if (Processor_Size == 8 && Processor_ID == 6)
	{
		bMin.x = 0.0f; bMin.y = 0.0f;  bMin.z = -16.0f;
		bMax.x = 15.0f;  bMax.y = 15.0f; bMax.z = 0.0f;
	}
	else if (Processor_Size == 8 && Processor_ID == 7)
	{
		bMin.x = 0.0f; bMin.y = 0.0f;  bMin.z = 0.0f;
		bMax.x = 15.0f;  bMax.y = 15.0f; bMax.z = 15.0f;
	}
	*/
	//std::cout << "[bounding box] PID << " << Processor_ID << " ] " << r.y << " " << a.y << " " << data_a.y << std::endl;
	
	




}

void Processor::initOpti()
{
	float rozm_x = this->kdTree->root->b.x - this->kdTree->root->a.x + 1;
	float rozm_y = this->kdTree->root->b.y - this->kdTree->root->a.y + 1;
	float rozm_z = this->kdTree->root->b.z - this->kdTree->root->a.z + 1;

	initCamera(rozm_x, rozm_y, rozm_z);
	setRatioUV();
}

bool Processor::computeOverlap(const Point2Di& sa, const Point2Di& sb, const Point2Di& ea, const Point2Di& eb, Point2Di& overlap_a, Point2Di& overlap_b)
{
	overlap_a.x = std::max(sa.x, ea.x);
    overlap_a.y = std::max(sa.y, ea.y);
    overlap_b.x = std::min(sb.x, eb.x);
    overlap_b.y = std::min(sb.y, eb.y);
    
    // 如果没有交集
    if (overlap_a.x > overlap_b.x || overlap_a.y > overlap_b.y) {
        return false;
    }
    return true;
}