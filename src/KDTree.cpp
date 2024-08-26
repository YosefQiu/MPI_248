#include "KDTree.h"

KDTree::KDTree()
{
	root = nullptr;
	depth = 0;
}

KDTree::KDTree(const float3& a, const float3& b, int depth, int split)
{
	root = nullptr;
	this->depth = depth;
	float3 data_a = a;
	float3 data_b = b;

	float3 box_a = a;
	float3 box_b = b;
	float3 box_data_a = a;
	float3 box_data_b = b;
	createNode(root, a, b, depth, split, 0, (int)pow(2, depth) - 1, data_a, data_b, "", box_a, box_b, box_data_a, box_data_b); // 2^depth - 1 = total processor number
	calcCompensation(root);
}

KDTree::~KDTree()
{
	freeNode(root);
}

void KDTree::freeNode(Node*& n)
{
	if (!n) return;
	if (n->front) freeNode(n->front);
	if (n->back) freeNode(n->back);
	if (n != nullptr) delete n; n = nullptr;
}

void KDTree::createNode(Node*& n, const float3& a, const float3& b, int depth, int split, int i1, int i2, const float3& data_a, const float3& data_b, std::string binTag)
{
	//n = new Node;
	//n->a = a;
	//n->b = b;
	//n->split = split;
	//n->front = nullptr;
	//n->back = nullptr;
	//n->i1 = i1;
	//n->i2 = i2;
	//n->data_a = data_a;
	//n->data_b = data_b;
	//n->binTag = binTag;

	//if (depth != 0) {
	//	float3 a1, b1, a2, b2;
	//	float3 data_a1, data_b1, data_a2, data_b2;

	//	switch (split) {
	//	case 0:  // 平行X轴分割（切割Y轴）
	//	{
	//		int midpoint_y = std::floor((a.y + b.y) / 2);

	//		// 上部分（进程1）
	//		a1 = make_float3(a.x, midpoint_y + 1, a.z);
	//		b1 = b;
	//		data_a1 = make_float3(a.x, midpoint_y, a.z); // 向下扩展1个单位（幽灵层）
	//		data_b1 = b;

	//		// 下部分（进程0）
	//		a2 = a;
	//		b2 = make_float3(b.x, midpoint_y, b.z);
	//		data_a2 = a2;
	//		data_b2 = make_float3(b.x, midpoint_y + 1, b.z); // 向上扩展1个单位（幽灵层）

	//		break;
	//	}
	//	case 1:  // 平行Z轴分割（切割X轴）
	//	{
	//		int midpoint_x = std::floor((a.x + b.x) / 2);

	//		// 右部分（进程1）
	//		a1 = make_float3(midpoint_x + 1, a.y, a.z);
	//		b1 = b;
	//		data_a1 = make_float3(midpoint_x, a.y, a.z); // 幽灵层向左扩展1个单位
	//		data_b1 = b;

	//		// 左部分（进程0）
	//		a2 = a;
	//		b2 = make_float3(midpoint_x, b.y, b.z);
	//		data_a2 = a2;
	//		data_b2 = make_float3(midpoint_x + 1, b.y, b.z); // 幽灵层向右扩展1个单位

	//		break;
	//	}
	//	case 2:  // 平行Y轴分割（切割Z轴）
	//	{
	//		int midpoint_z = std::floor((a.z + b.z) / 2);

	//		// 后部分（进程1）
	//		a1 = make_float3(a.x, a.y, midpoint_z + 1);
	//		b1 = b;
	//		data_a1 = make_float3(a.x, a.y, midpoint_z); // 幽灵层向前扩展1个单位
	//		data_b1 = b;

	//		// 前部分（进程0）
	//		a2 = a;
	//		b2 = make_float3(b.x, b.y, midpoint_z);
	//		data_a2 = a2;
	//		data_b2 = make_float3(b.x, b.y, midpoint_z + 1); // 幽灵层向后扩展1个单位

	//		break;
	//	}
	//	}

	//	int fi1, fi2, bi1, bi2;

	//	fi1 = i1 + (std::abs(i1 - i2) + 1) / 2;
	//	fi2 = i2;

	//	bi1 = i1;
	//	bi2 = fi1 - 1;

	//	int next_split = (split + 1) % 3;
	//	createNode(n->back, a1, b1, depth - 1, next_split, fi1, fi2, data_a1, data_b1, binTag + "1"); // 高位部分
	//	createNode(n->front, a2, b2, depth - 1, next_split, bi1, bi2, data_a2, data_b2, binTag + "0"); // 低位部分


	n = new Node;
	n->a = a;
	n->b = b;
	n->split = split;
	n->front = nullptr;
	n->back = nullptr;
	n->i1 = i1;
	n->i2 = i2;
	n->data_a = data_a;
	n->data_b = data_b;
	n->binTag = binTag;

	if (depth != 0)
	{
		float3 a1, b1, a2, b2;
		float3 data_a1, data_b1, data_a2, data_b2;
		switch (split)					// rozdelim data podla prislusnej roviny
		{
		case 0:									// 划分平面平行于X轴
		{
			int midpoint_y = std::floor((data_a.y + data_b.y) / 2);

			// 上部分（进程1）
			a1 = make_float3(data_a.x, midpoint_y + 1, data_a.z);
			b1 = data_b;
			data_a1 = make_float3(data_a.x, midpoint_y, data_a.z); // 向下扩展1个单位（幽灵层）
			data_b1 = data_b;

			// 下部分（进程0）
			a2 = data_a;
			b2 = make_float3(data_b.x, midpoint_y, data_b.z);
			data_a2 = data_a;
			data_b2 = make_float3(data_b.x, midpoint_y + 1, data_b.z); // 向上扩展1个单位（幽灵层）

			break;
		}
		case 1:									// 划分平面平行于Y轴
		{
			int midpoint_x = std::floor((data_a.x + data_b.x) / 2);

			// 右部分（进程1）
			a1 = make_float3(midpoint_x + 1, data_a.y, data_a.z);
			b1 = data_b;
			data_a1 = make_float3(midpoint_x, data_a.y, data_a.z); // 幽灵层向左扩展1个单位
			data_b1 = data_b;

			// 左部分（进程0）
			a2 = data_a;
			b2 = make_float3(midpoint_x, data_b.y, data_b.z);
			data_a2 = data_a;
			data_b2 = make_float3(midpoint_x + 1, data_b.y, data_b.z); // 幽灵层向右扩展1个单位
			break;
		}
		case 2:									// 划分平面平行于Z轴
		{
			int midpoint_z = std::floor((data_a.z + data_b.z) / 2);

			// 后部分（进程1）
			a1 = make_float3(data_a.x, data_a.y, midpoint_z + 1);
			b1 = data_b;
			data_a1 = make_float3(data_a.x, data_a.y, midpoint_z); // 幽灵层向前扩展1个单位
			data_b1 = data_b;

			// 前部分（进程0）
			a2 = data_a;
			b2 = make_float3(data_b.x, data_b.y, midpoint_z);
			data_a2 = data_a;
			data_b2 = make_float3(data_b.x, data_b.y, midpoint_z + 1); // 幽灵层向后扩展1个单位

			break;
		}
		}

		int fi1, fi2, bi1, bi2;

		fi1 = i1 + (std::abs(i1 - i2) + 1) / 2;
		fi2 = i2;

		bi1 = i1;
		bi2 = fi1 - 1;

	

		int next_split = (split + 1) % 3;
		createNode(n->front, a1, b1, depth - 1, next_split, fi1, fi2, data_a1, data_b1, binTag + "1");
		createNode(n->back, a2, b2, depth - 1, next_split, bi1, bi2, data_a2, data_b2, binTag + "0");

	}
}

void KDTree::createNode(Node*& n, const float3& a, const float3& b, int depth, int split, int i1, int i2, const float3& data_a, const float3& data_b, std::string binTag, const float3& box_a, const float3& box_b, const float3& box_data_a, const float3& box_data_b)
{
	n = new Node;
	n->a = a;
	n->b = b;
	n->split = split;
	n->front = nullptr;
	n->back = nullptr;
	n->i1 = i1;
	n->i2 = i2;
	n->data_a = data_a;
	n->data_b = data_b;
	n->binTag = binTag;

	n->bbox_a = box_a;
	n->bbox_b = box_b;
	n->bbox_data_a = box_data_a;
	n->bbox_data_b = box_data_b;

	if (depth != 0)
	{
		float3 a1, b1, a2, b2;
		float3 data_a1, data_b1, data_a2, data_b2;

		float3 box_a1, box_b1, box_a2, box_b2;
		float3 box_data_a1, box_data_b1, box_data_a2, box_data_b2;

		switch (split)					// rozdelim data podla prislusnej roviny
		{
		case 0:									// 划分平面平行于X轴
		{
			int midpoint_y = std::floor((data_a.y + data_b.y) / 2);

			// 上部分（进程1）
			a1 = make_float3(data_a.x, midpoint_y + 1, data_a.z);
			b1 = data_b;
			data_a1 = make_float3(data_a.x, midpoint_y, data_a.z); // 向下扩展1个单位（幽灵层）
			data_b1 = data_b;

			// 下部分（进程0）
			a2 = data_a;
			b2 = make_float3(data_b.x, midpoint_y, data_b.z);
			data_a2 = data_a;
			data_b2 = make_float3(data_b.x, midpoint_y + 1, data_b.z); // 向上扩展1个单位（幽灵层）


			// box data
			box_a1.x = box_a.x; box_a1.y = box_a.y + (std::abs(box_a.y - box_b.y) + 1) / 2; box_a1.z = box_a.z;
			box_b1 = box_b;
			box_data_a1 = box_a1; box_data_b1 = box_b1;
			// back (down)
			box_a2 = box_a;
			box_b2.x = box_b.x; box_b2.y = box_a1.y; box_b2.z = box_b.z;
			box_data_a2 = box_a2; box_data_b2 = box_b2;  box_data_b2.y = box_a1.y;

			break;
		}
		case 1:									// 划分平面平行于Y轴
		{
			int midpoint_x = std::floor((data_a.x + data_b.x) / 2);

			// 右部分（进程1）
			a1 = make_float3(midpoint_x + 1, data_a.y, data_a.z);
			b1 = data_b;
			data_a1 = make_float3(midpoint_x, data_a.y, data_a.z); // 幽灵层向左扩展1个单位
			data_b1 = data_b;

			// 左部分（进程0）
			a2 = data_a;
			b2 = make_float3(midpoint_x, data_b.y, data_b.z);
			data_a2 = data_a;
			data_b2 = make_float3(midpoint_x + 1, data_b.y, data_b.z); // 幽灵层向右扩展1个单位

			// box data
			// front (right)
			box_a1.x = box_a.x + (std::abs(box_a.x - box_b.x) + 1) / 2; box_a1.y = box_a.y; box_a1.z = box_a.z;
			box_b1 = box_b;
			box_data_a1 = box_a1; box_data_b1 = box_b1;
			// back (left)
			box_a2 = box_a;
			box_b2.x = box_a1.x; box_b2.y = box_b.y; box_b2.z = box_b.z;
			box_data_a2 = box_a2; box_data_b2 = box_b2; box_data_b2.x = box_a1.x;
			break;
		}
		case 2:									// 划分平面平行于Z轴
		{
			int midpoint_z = std::floor((data_a.z + data_b.z) / 2);

			// 后部分（进程1）
			a1 = make_float3(data_a.x, data_a.y, midpoint_z + 1);
			b1 = data_b;
			data_a1 = make_float3(data_a.x, data_a.y, midpoint_z); // 幽灵层向前扩展1个单位
			data_b1 = data_b;

			// 前部分（进程0）
			a2 = data_a;
			b2 = make_float3(data_b.x, data_b.y, midpoint_z);
			data_a2 = data_a;
			data_b2 = make_float3(data_b.x, data_b.y, midpoint_z + 1); // 幽灵层向后扩展1个单位


			//box data
			box_a1.x = box_a.x; box_a1.y = box_a.y; box_a1.z = box_a.z + (std::abs(box_a.z - box_b.z) + 1) / 2;
			box_b1 = box_b;
			box_data_a1 = box_a1; box_data_b1 = box_b1;
			box_a2 = box_a;
			box_b2.x = box_b.x; box_b2.y = box_b.y; box_b2.z = box_a1.z;
			box_data_a2 = a2; box_data_b2 = box_b2;

			break;
		}
		}

		int fi1, fi2, bi1, bi2;

		fi1 = i1 + (std::abs(i1 - i2) + 1) / 2;
		fi2 = i2;

		bi1 = i1;
		bi2 = fi1 - 1;



		int next_split = (split + 1) % 3;
		createNode(n->front, a1, b1, depth - 1, next_split, fi1, fi2, data_a1, data_b1, binTag + "1", box_a1, box_b1, box_data_a1, box_data_b1);
		createNode(n->back, a2, b2, depth - 1, next_split, bi1, bi2, data_a2, data_b2, binTag + "0", box_a2, box_b2, box_data_a2, box_data_b2);

	}
}



void KDTree::calcCompensation(Node* node)
{
	if (node == nullptr) return;

	if (node->front == nullptr && node->back == nullptr) 
	{
		// 计算补偿
		for (int i = 0; i < node->binTag.length(); i++) 
		{
			if (node->binTag[i] == '1') {
				if (i % 3 == 0) node->compensation.y = 1.0f; // 影响 Y 轴
				if (i % 3 == 1) node->compensation.x = 1.0f; // 影响 X 轴
				if (i % 3 == 2) node->compensation.z = 1.0f; // 影响 Z 轴
			}
		}
	}

	calcCompensation(node->front);
	calcCompensation(node->back);
}

/*

void KDTree::createNode(Node*& n, const float3& a, const float3& b, int depth, int split, int i1, int i2, const float3& data_a, const float3& data_b, std::string binTag)
{
	n = new Node;
	n->a = a;
	n->b = b;
	n->split = split;
	n->front = nullptr;
	n->back = nullptr;
	n->i1 = i1;
	n->i2 = i2;
	n->data_a = data_a;
	n->data_b = data_b;
	n->binTag = binTag;

	if (depth != 0)
	{
		float3 a1, b1, a2, b2;
		float3 data_a1, data_b1, data_a2, data_b2;
		switch (split)					// rozdelim data podla prislusnej roviny
		{
		case 0:									// 划分平面平行于X轴
		{
			// front (up)  进程1
			//a1.x = a.x; a1.y = a.y + (std::abs(a.y - b.y) + 1) / 2; a1.z = a.z;
			//b1 = b; 
			//data_a1 = a1; data_b1 = b1; 
			//// back (down) 进程0
			//a2 = a;
			//b2.x = b.x; b2.y = a1.y; b2.z = b.z;
			//data_a2 = a2; data_b2 = b2;  data_b2.y = a1.y;

			int ghost_cell = 1;
			int midpoint_y_floor = std::floor(std::abs(a.y - b.y) / 2);  // 向下取整
			float midpoint_y = std::abs(a.y - b.y) / 2;

			// 进程1 (上部分)
			a1.x = a.x;
			a1.y = a.y + midpoint_y;
			a1.z = a.z;
			b1 = b;

			// 原本的分割逻辑
			data_a1 = { a1.x, a1.y, a1.z };
			data_b1 = { b1.x, b1.y, b1.z };

			// 添加幽灵单元: 向下扩展 data_a1，使得范围变为 [0, 15, 0] -> [31, 31, 31]
			data_a1.y = a.y + midpoint_y_floor;;
			if (data_a1.y < 0) data_a1.y = 0;  // 确保不低于 0

			// 进程0 (下部分)
			a2 = a;
			b2.x = b.x;
			b2.y = b.y - midpoint_y;
			b2.z = b.z;

			// 原本的分割逻辑
			data_a2 = a2;
			data_b2 = { b2.x, b2.y, b2.z };

			// 添加幽灵单元: 向上扩展 data_b2，使得范围变为 [0, 0, 0] -> [31, 16, 31]
			data_b2.y = b.y - midpoint_y_floor;
			if (data_b2.y > 31) data_b2.y = 31;  // 确保不超出范围
			break;
		}
		case 1:									// 划分平面平行于Y轴
			// front (right)
			a1.x = a.x + (std::abs(a.x - b.x) + 1) / 2; a1.y = a.y; a1.z = a.z;
			b1 = b;
			data_a1 = a1; data_b1 = b1;
			// back (left)
			a2 = a;
			b2.x = a1.x; b2.y = b.y; b2.z = b.z;
			data_a2 = a2; data_b2 = b2; data_b2.x = a1.x;
			break;
		case 2:									// 划分平面平行于Z轴
			a1.x = a.x; a1.y = a.y; a1.z = a.z + (std::abs(a.z - b.z) + 1) / 2;
			b1 = b;
			data_a1 = a1; data_b1 = b1;
			a2 = a;
			b2.x = b.x; b2.y = b.y; b2.z = a1.z;
			data_a2 = a2; data_b2 = b2;
			break;
		}

		int fi1, fi2, bi1, bi2;

		fi1 = i1 + (std::abs(i1 - i2) + 1) / 2;
		fi2 = i2;

		bi1 = i1;
		bi2 = fi1 - 1;

		data_a1.x = std::max(a1.x - 1, 0);
		data_a1.y = std::max(a1.y - 1, 0);
		data_a1.z = std::max(a1.z - 1, 0);
		data_b1.x = std::min(b1.x + 1, 31);
		data_b1.y = std::min(b1.y + 1, 31);
		data_b1.z = std::min(b1.z + 1, 31);

		data_a2.x = std::max(a2.x - 1, 0);
		data_a2.y = std::max(a2.y - 1, 0);
		data_a2.z = std::max(a2.z - 1, 0);
		data_b2.x = std::min(b2.x + 1, 31);
		data_b2.y = std::min(b2.y + 1, 31);
		data_b2.z = std::min(b2.z + 1, 31);

		int next_split = (split + 1) % 3;
		createNode(n->front, a1, b1, depth - 1, next_split, fi1, fi2, data_a1, data_b1);
		createNode(n->back, a2, b2, depth - 1, next_split, bi1, bi2, data_a2, data_b2);

	}
}*/