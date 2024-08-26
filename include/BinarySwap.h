#pragma once
#include "ggl.h"
#include "KDTree.h"

typedef struct
{
	int pid;		//表示需要进行通信的处理器编号。
	int over;		//over 的值为 1 表示合成方式为“over”，值为 0 表示合成方式为“under”。
}Plan;


bool inInterv(int number, int a, int b); //判断数字 number 是否在区间[a, b] 内


