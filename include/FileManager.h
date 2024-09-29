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
        std::string fileName, uint64_t width, uint64_t height, uint64_t depth,
        uint64_t startX, uint64_t endX, uint64_t startY, uint64_t endY, uint64_t startZ, uint64_t endZ,
        uint64_t& subWidth, uint64_t& subHeight, uint64_t& subDepth);
    
    static void saveRawFile(const std::string& fileName, const void* data, size_t dataSize);
};
