#pragma once
#include "ggl.h"
#include "mpi.h"
#include "KDTree.h"
#include "BinarySwap.h"
#include "Camera.h"
//一个表示二维点的类或结构，包含 x 和 y 坐标
typedef struct
{
	int x, y;
}Point2Di;

class Processor
{
public:
	int Processor_ID;
	int Processor_Size;
	MPI_Status Processor_status;	//在 mpi_recv - 我们实际收到了多少数据
public:
	// data split
	KDTree* kdTree;
	float3 a, b;						// 数据的起始点和终止点
	float3 data_a, data_b;
	float3 box_a, box_b;
	float3 data_compensation;
	int data_len;
	float3 whole_data_len;
public:
	// binary swap
	float* alpha_sbuffer;
	float* alpha_rbuffer;
	float* d_alpha_sbuffer;
	float* d_alpha_rbuffer;



	float* h_alpha_sbuffer;
	float* h_alpha_rbuffer;
	float h_s_len;
	float h_r_len;

	float* rgb_sbuffer;
	float* rgb_rbuffer;
	float* d_rgb_sbuffer;
	float* d_rgb_rbuffer;

	float* sbuffer;
	float* rbuffer;
	
	Plan* plan;
	
	Point2Di obr_a, obr_b;			// 要发送数据的起始点和终止点（图像上）
	Point2Di obr_alpha_a, obr_alpha_b;
	Point2Di obr_rgb_a, obr_rgb_b;
	int obr_x, obr_y;				//图像尺寸

	size_t tmpRecvCound;
	
	float* obr;						// 图像数组
	float* obr_alpha;				// 图像alpha 数组
	float* d_obr_alpha;				// GPU上的alpha数组
	float* obr_rgb;					// 图像rgb数组
	float* d_obr_rgb;				// GPU上的rgb数组
	float*** alpha_values_u;		// 每次交换的当前的alpha 值
	float* d_alpha_values_u;		// GPU上的每次交换的当前的alpha 值

	size_t alpha_totalSentBytes = 0;
	size_t alpha_totalReceivedBytes = 0;

	size_t totalSentBytes = 0;
	size_t totalReceivedBytes = 0;
public:
	// camera
	unsigned int camera_plane_x, camera_plane_y;
	unsigned int scr_x, scr_y;
	float cam_dx, cam_dy, cam_vz;
	float cam_old_dx, cam_old_dy, cam_old_vz;
	Camera* camera;
public:
	// AABB
	float3 bMin;
	float3 bMax;
	
public:
	unsigned char* data = nullptr;
	unsigned char* data2 = nullptr;

public:
	Processor() :plan(nullptr), kdTree(nullptr), sbuffer(nullptr), rbuffer(nullptr), camera(nullptr)
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &this->Processor_ID);
		MPI_Comm_size(MPI_COMM_WORLD, &this->Processor_Size);
	}
	~Processor()
	{
		if (kdTree) delete kdTree; kdTree = nullptr;
		if (plan) delete[] plan; plan = nullptr;
		if (camera) delete camera; camera = nullptr;

		if (sbuffer) delete[] sbuffer; sbuffer = nullptr;
		if (rbuffer) delete[] rbuffer; rbuffer = nullptr;

		if(alpha_sbuffer) delete[] alpha_sbuffer; alpha_sbuffer = nullptr;
		if(alpha_rbuffer) delete[] alpha_rbuffer; alpha_rbuffer = nullptr;

		if(rgb_sbuffer) delete[] rgb_sbuffer; rgb_sbuffer = nullptr;
		if(rgb_rbuffer) delete[] rgb_rbuffer; rgb_rbuffer = nullptr;

		cudaFree(d_alpha_sbuffer);
		cudaFree(d_alpha_rbuffer);
	}
public:
	void init_node(float3& a, float3& b, int id);
	void init_master(const std::string& s, float3& a, float3& b, cudaExtent volumeTotalSize);
	void initScreen(unsigned int w, unsigned int h);
	void initRayCaster(unsigned int w, unsigned int h);
	void initKDTree();
	void initImage(int w, int h);
	void initData(const char* filename);
	void initOpti();
	void binarySwap(float* img);
	void binarySwap_Alpha(float* img);
	void binarySwap_RGB(float* img, int MinX, int MinY, int MaxX, int MaxY, bool bUseCompression = true);
	void binarySwap_Alpha_GPU(float* d_imgAlpha);
	void binarySwap_RGB_GPU(float* img, int MinX, int MinY, int MaxX, int MaxY, bool bUseCompression = true);
public:
	void setRatioUV();
	void setCamera();
	void setCameraProperty(float cam_dx, float cam_dy, std::optional<float> cam_vz = std::nullopt);
private:
	void initCamera(int rozm_x, int rozm_y, int rozm_z);
	void updateCamera(int rozm_x, int rozm_y, int rozm_z);
	int data_size();
	void master_load_data(float3 a, float3 b, std::string s);
	void send_data(unsigned char* buf, int pocet, int dest, int tag);
	void recv_data(unsigned char* buf, int pocet, int source, int tag, int& friends);
	void createPlan(int ID, int depth, Node* n, const float3& v, Plan*& out);
	void setDimensions(const int u, const Point2Di& a, const Point2Di& b, Point2Di& sa, Point2Di& sb, Point2Di& ra, Point2Di& rb);
	void loadBuffer(const Point2Di& sa, const Point2Di& sb);
	void loadColorBuffer(const Point2Di& sa, const Point2Di& sb);
	void loadColorBufferRRGGBB(const Point2Di& sa, const Point2Di& sb);
	void compositng(const int u);
	void visibleFunc(const int u);
	void compositngColor(const int u);
	void compositngColorRRGGBB(const int u);
	void compositngColorRRGGBB(const int u, float* buffer);
	void reset();
	bool read_data(const std::string& s, float3& a, float3& b, cudaExtent volumeTotalSize);
	
};



