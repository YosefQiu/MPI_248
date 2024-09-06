#pragma once
#include "ggl.h"
#include "stb_image.h"
#include "stb_image_write.h"
class Utils
{
public:
	static void saveArrayAsBinary(const char* filename, float* data, size_t width, size_t height);
	static void saveArrayAsBinary(const char* filename, float* data, int minX, int minY, int maxX, int maxY, int imageWidth, int imageHeight);
	static void saveArrayAsBinaryRGB(const char* filename, float* data, int minX, int minY, int maxX, int maxY, int imageWidth, int imageHeight);
	static void loadBinaryFile(const char* filename, int width, int height, float* output);
	static void saveImageAsPNG(const char* filename, float* data, int width, int height);
	static void splitRGBA(float* h_output, int image_width, int image_height, float*& h_rgb, float*& h_alpha);
	static void combineRGBA(float* h_rgb, float* h_alpha, int image_width, int image_height, float*& h_output);
	static void convertRGBtoRRRGGGBBB(float* src_buffer, size_t buffer_len, float* dst_buffer);
	static void convertRRRGGGBBBtoRGB(float* src_buffer, size_t buffer_len, float* dst_buffer);

	static void recordCudaRenderTime(const char* filename, int size, int rank, float elapsedTime);
	

};

