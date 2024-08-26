#pragma once
#include "ggl.h"


typedef struct _Node
{
	int split;						// 划分数据的平面
	float3 a, b;					// 数据的边界点，低点和高点
	float3 data_a, data_b;			// 读取数据的边界点
	float3 bbox_a, bbox_b;
	float3 bbox_data_a, bbox_data_b;
	float3 compensation;			// 补偿值
	std::string binTag;				// 二进制标记
	int i1, i2;						// 节点包含的处理器区间
	struct _Node* front;			// 二叉划分数据的前子节点
	struct _Node* back;				// 后子节点
	_Node() : front(nullptr), back(nullptr) 
	{
		compensation = make_float3(0.0, 0.0, 0.0);
	}
} Node;


class KDTree
{
public:
	Node* root; // 树的根节点
	int depth;	// 树的深度
public:
	KDTree();
	//用 2 个点(3D) 界定体积，划分到什么深度，从哪个平面开始划分[0..2 | , -, \]
	KDTree(const float3& a, const float3& b, int depth, int split);
	~KDTree();
public:
	void freeNode(Node*& n);
	void createNode(Node*& n, const float3& a, const float3& b, int depth, int split, int i1, int i2, const float3& data_a, const float3& data_b, std::string binTag);
	void createNode(Node*& n, const float3& a, const float3& b, int depth, int split, int i1, int i2, const float3& data_a, const float3& data_b, std::string binTag, const float3& box_a, const float3& box_b, const float3& box_data_a, const float3& box_data_b);
	void calcCompensation(Node* node);
};

