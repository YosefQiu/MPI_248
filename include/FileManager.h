#pragma once
#include "ggl.h"

class FileManager
{
public:
    static void* loadRawFile(const char* filename, size_t size);
    // divide by Z 
    static void* loadPartialRawFile(const char* filename, size_t local_volume_size, int start_slice, int end_slice, size_t volumeSize);
    // divide by X - Y - Z
    static void* loadPartialRawFile(const char* filename, size_t local_volume_size,
        int start_x, int end_x, int start_y, int end_y, int start_z, int end_z, size_t volumeSize);
    static void* loadPartialRawFile2(const char* filename, size_t local_volume_size,
        int start_x, int end_x, int start_y, int end_y, int start_z, int end_z, size_t volumeSize);
    

    static void* loadRawFileSubVolume(
        const std::string& fileName, int width, int height, int depth,
        int startX, int endX, int startY, int endY, int startZ, int endZ,
        int& subWidth, int& subHeight, int& subDepth);
    
    static void saveRawFile(const std::string& fileName, const void* data, size_t dataSize);
};
