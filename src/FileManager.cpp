#include "FileManager.h"

void *FileManager::loadRawFile(const char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return nullptr;
    }
    
    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

#if defined(_MSC_VER_)
  printf("Read '%s', %Iu bytes\n", filename, read);
#else
  printf("Read '%s', %zu bytes\n", filename, read);
#endif

  return data;
}

void* FileManager::loadPartialRawFile(const char* filename, size_t local_volume_size, int start_slice, int end_slice, size_t volumeSize)
{

    typedef unsigned char VolumeType;
    struct VolumeSize
    {
        int width; int height; int depth;
    };

    VolumeSize vlSize;
    vlSize.width = 32; vlSize.height = 32; vlSize.depth = 32;

    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return nullptr;
    }

    size_t slice_size = vlSize.width * vlSize.height * sizeof(VolumeType);
    size_t offset = start_slice * slice_size;
    size_t size_to_read = (end_slice - start_slice) * slice_size;

    fseek(fp, offset, SEEK_SET);

    void* data = malloc(size_to_read);
    if (!data)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        fclose(fp);
        return nullptr;
    }

    size_t read = fread(data, 1, size_to_read, fp);
    fclose(fp);
#if defined(_MSC_VER_)
    printf("Read '%s', %Iu bytes\n", filename, read);
#else
    printf("Read '%s', %zu bytes\n", filename, read);
#endif
    
    return data;

}

void* FileManager::loadPartialRawFile(const char* filename, size_t local_volume_size, int start_x, int end_x, int start_y, int end_y, int start_z, int end_z, size_t volumeSize)
{
    typedef unsigned char VolumeType;
    struct VolumeSize
    {
        int width; int height; int depth;
    };

    VolumeSize vlSize;
    vlSize.width = 32; vlSize.height = 32; vlSize.depth = 32;

    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return nullptr;
    }

    int width = end_x - start_x;
    int height = end_y - start_y;
    int depth = end_z - start_z;

    VolumeType* data = (VolumeType*)malloc(local_volume_size);
    if (!data)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        fclose(fp);
        return nullptr;
    }

    size_t bytes_per_row = vlSize.width * sizeof(VolumeType);
    size_t bytes_per_slice = bytes_per_row * vlSize.height;
    size_t row_length = width * sizeof(VolumeType);

    fseek(fp, start_z* bytes_per_slice + start_y * bytes_per_row + start_x * sizeof(VolumeType), SEEK_SET);

    for (int z = 0; z < depth; z++)
    {
        for (int y = 0; y < height; y++)
        {
            size_t read = fread(data + (z * height + y) * width, sizeof(VolumeType), width, fp);
            if (read != width)
            {
                fprintf(stderr, "Failed to read data\n");
                free(data);
                fclose(fp);
                return nullptr;
            }
            fseek(fp, bytes_per_row - row_length, SEEK_CUR);
        }
        if (z < depth - 1)
            fseek(fp, bytes_per_slice - height * bytes_per_row, SEEK_CUR);
    }
    fclose(fp);

    return data;
}

void* FileManager::loadPartialRawFile2(const char* filename, size_t local_volume_size, int start_x, int end_x, int start_y, int end_y, int start_z, int end_z, size_t data_volumeSize)
{
    typedef unsigned char VolumeType;
    struct VolumeSize
    {
        int width; int height; int depth;
    };
   

    VolumeSize vlSize;
    vlSize.width = 32; vlSize.height = 32; vlSize.depth = 32;

    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return nullptr;
    }
    int width = end_x - start_x;
    int height = end_y - start_y;
    int depth = end_z - start_z;

    size_t total_bytes = width * height * depth * sizeof(VolumeType);
    VolumeType* data = (VolumeType*)malloc(total_bytes);
    if (!data)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        fclose(fp);
        return nullptr;
    }

   
    for (int z = start_z; z < end_z; ++z)
    {
        for (int y = start_y; y < end_y; ++y)
        {
            size_t offset = (z * vlSize.height + y) * vlSize.width + start_x;
            fseek(fp, offset * sizeof(VolumeType), SEEK_SET);
            if (fread(data + ((z - start_z) * width * height + (y - start_y) * width), sizeof(VolumeType), width, fp) != width)
            {
                fprintf(stderr, "Failed to read data at Z %d, Y : %d\n", z, y);
                fclose(fp);
                delete[] data;
                return nullptr;
            }
        }
    }
    fclose(fp);

    return data;
}

// void* FileManager::loadRawFileSubVolume(
//     const std::string& fileName, int width, int height, int depth,
//     int startX, int endX, int startY, int endY, int startZ, int endZ,
//     int& subWidth, int& subHeight, int& subDepth) 
// {
    
//     std::cout << "done calc sizedfafdafasfd0000000000000adsfsafsafsd\n"; std::cout << std::flush;
//     // Open the file in binary mode
//     std::ifstream file(fileName, std::ios::binary);
//     std::cout << "done calc sizedfafdafasfd1111111111111adsfsafsafsd\n"; std::cout << std::flush;
//     // Check if the file was successfully opened
//     if (!file.is_open()) {
//         std::cerr << "Failed to open file: " << fileName << std::endl; 
//         return nullptr;
//     }

//     // Calculate the size of the entire data
//     size_t dataSize = width * height * depth;
    
//     std::cout << "done calc sizdfafse\n";

//     // Create a buffer to hold the entire file contents
//     std::vector<unsigned char> buffer(dataSize);
// /*
//    //
// */
//     // Read the file contents into the buffer
//     if (file.read(reinterpret_cast<char*>(buffer.data()), dataSize)) 
//     {
//         std::cout << "File loaded successfully!" << std::endl; std::cout << std::flush;
//     }
//     else 
//     {
//         std::cerr << "Failed to read the file." << std::endl; std::cout << std::flush;
//         return nullptr;
//     }

//     // Close the file
//     file.close();

//     // Calculate the dimensions of the subvolume
//     subWidth = endX - startX + 1;
//     subHeight = endY - startY + 1;
//     subDepth = endZ - startZ + 1;
//     size_t subVolumeSize = subWidth * subHeight * subDepth;
//     std::cout << "File loaded 000000--------------------!" << std::endl; std::cout << std::flush;
//     // Create a buffer to hold the subvolume data
//     unsigned char* subVolume = new unsigned char[subVolumeSize];
//     std::cout << "File loaded successfullydfafdsfsd!" << std::endl; std::cout << std::flush;
//     // Fill the subvolume buffer with data from the main buffer
//     size_t index = 0;
//     // for (int z = startZ; z <= endZ; ++z) 
//     // {
//     //     for (int y = startY; y <= endY; ++y) 
//     //     {
//     //         for (int x = startX; x <= endX; ++x) 
//     //         {
//     //             subVolume[index++] = buffer[z * (width * height) + y * width + x];
//     //         }
//     //     }
//     // }
//     for (int z = startZ; z <= endZ; ++z) {
//     for (int y = startY; y <= endY; ++y) {
//         for (int x = startX; x <= endX; ++x) {
//             size_t bufferIndex = z * (width * height) + y * width + x;
//             if (bufferIndex >= buffer.size()) {
//                 std::cerr << "Error: Buffer index out of bounds at "
//                           << "x: " << x << ", y: " << y << ", z: " << z << std::endl;
//                 return nullptr;
//             }
//             subVolume[index++] = buffer[bufferIndex];
//         }
//     }
// }
//     std::cout << "File loaded 111111--------------------!" << std::endl; std::cout << std::flush;
//     return static_cast<void*>(subVolume);
// }

// void FileManager::saveRawFile(const std::string& fileName, const void* data, size_t dataSize) {
//     std::ofstream file(fileName, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open file: " << fileName << std::endl;
//         return;
//     }

//     file.write(reinterpret_cast<const char*>(data), dataSize);
//     file.close();
// }

void* FileManager::loadRawFileSubVolume(
    std::string fileName, uint64_t width, uint64_t height, uint64_t depth,
    uint64_t startX, uint64_t endX, uint64_t startY, uint64_t endY, uint64_t startZ, uint64_t endZ,
    uint64_t& subWidth, uint64_t& subHeight, uint64_t& subDepth) 
{
    // std::cout << "开始加载子体积..." << std::endl;

    // 以二进制模式打开文件
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << fileName << std::endl;
        return nullptr;
    }

    // 计算子体积的尺寸
    subWidth = endX - startX + 1;
    subHeight = endY - startY + 1;
    subDepth = endZ - startZ + 1;
    size_t subVolumeSize = subWidth * subHeight * subDepth;

    // 创建一个缓冲区来存储子体积数据
    unsigned char* subVolume = new unsigned char[subVolumeSize];

    // 我们将分块读取文件，直接将数据存入 subVolume 缓冲区
    size_t index = 0;
    // std::cout << "width " << width << " height " << height << " depth " << depth << std::endl;
    // std::cout << "文件总大小 (bytes): " << width * height * depth * sizeof(unsigned char) << std::endl;
    for (int z = startZ; z <= endZ; ++z) {
        for (int y = startY; y <= endY; ++y) {
            // 计算此行数据在文件中的偏移量
            uint64_t fileOffset = (z * width * height + y * width + startX) * sizeof(unsigned char);
            
            // 调试输出：打印偏移量和文件大小
            // std::cout << "正在定位文件指针到 offset: " << fileOffset << std::endl;
            
            // 定位文件指针到正确的位置
            file.seekg(fileOffset, std::ios::beg);
            if (!file.good()) {
                std::cerr << "文件指针定位错误。fileOffset: " << fileOffset 
                          << ", 文件总大小: " << width * height * depth * sizeof(unsigned char) << std::endl;
                delete[] subVolume;
                return nullptr;
            }

            // 读取从 startX 到 endX 这一行的数据
            size_t rowSize = (endX - startX + 1) * sizeof(unsigned char);
            // std::cout << "读取行数据，大小: " << rowSize << " bytes" << std::endl;
            
            file.read(reinterpret_cast<char*>(&subVolume[index]), rowSize);
            if (!file.good()) {
                std::cerr << "文件读取错误。" << std::endl;
                delete[] subVolume;
                return nullptr;
            }

            // 更新缓冲区索引，向前移动一行的大小
            index += (endX - startX + 1);
        }
    }

    // 关闭文件
    file.close();

    // std::cout << "子体积加载成功！" << std::endl;
    return static_cast<void*>(subVolume);
}