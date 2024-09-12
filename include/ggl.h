#pragma once
#pragma warning(disable:4267)

//c++ header file
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <array>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <iomanip>
#include <chrono>
#include <limits>
// #include <numbers>
#include <cmath>

//c header file
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <atomic>


// CUDA
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_timer.h>



#if __linux__
// SZX
#include "szx.h"
#include "szx_rw.h"
#include "cuszx_entry.h"
#include <unistd.h>

#include <omp.h>

extern struct timeval startTime;
extern struct timeval endTime;  /* Start and end times */
extern struct timeval costStart; /*only used for recording the cost*/
extern double totalCost;


inline void cost_start()
{
	totalCost = 0;
	gettimeofday(&costStart, NULL);
}

inline void cost_end()
{
	double elapsed;
	struct timeval costEnd;
	gettimeofday(&costEnd, NULL);
	elapsed = ((costEnd.tv_sec*1000000+costEnd.tv_usec)-(costStart.tv_sec*1000000+costStart.tv_usec))/1000000.0;
	totalCost += elapsed;
}
#endif