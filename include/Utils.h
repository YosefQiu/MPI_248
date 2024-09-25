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
	static void recordCudaRenderTime(const char* filename, std::string title, std::string iterator, float elapsedTime);
	static void recordByte(const char* filename, std::string title, std::string iterator, size_t byte);
	static void recordBSETime(const char* filename, int u, float elapsedTime, float compressTime = -1, float decompressTime = -1);
	static void recordTotalTime(const char* filename, float elapsedTime, std::string iteration);
	static void AddEndline(const char* filename);

	static void parseArguments(int argc, char** argv, 
					std::string& volumeFilename, int& xdim, int& ydim, int& zdim, 
                    unsigned int& image_width, unsigned int& image_height, 
					float& cam_dx, float& cam_dy, float& cam_dz,
                    bool& usecompress, bool& useeffarea, 
					std::string& iteration_str);

};

