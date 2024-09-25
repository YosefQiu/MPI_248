#include "Utils.h"
#include "mpi.h"
void Utils::saveArrayAsBinary(const char* filename, float* data, size_t width, size_t height)
{
	std::ofstream file(filename, std::ios::out | std::ios::binary);
	if (!file) {
		std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
		return;
	}

	size_t dataSize = width * height * 4 * sizeof(float);
	file.write(reinterpret_cast<char*>(data), dataSize);

	file.close();
	if (!file) {
		std::cerr << "Error: Unable to close file " << filename << std::endl;
	}
}

void Utils::saveArrayAsBinary(const char* filename, float* data, int minX, int minY, int maxX, int maxY, int imageWidth, int imageHeight) {
	std::ofstream file(filename, std::ios::out | std::ios::binary);
	if (!file) {
		std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
		return;
	}

	// 计算新的宽度和高度
	int newWidth = maxX - minX + 1;
	int newHeight = maxY - minY + 1;

	// 只保存这个范围内的数据
	for (int y = minY; y <= maxY; ++y) {
		for (int x = minX; x <= maxX; ++x) {
			int index = (y * imageWidth + x) * 4; // 4 channels (RGBA)
			file.write(reinterpret_cast<char*>(&data[index]), 4 * sizeof(float));
		}
	}

	file.close();
	if (!file) {
		std::cerr << "Error: Unable to close file " << filename << std::endl;
	}

	// 输出新图像的大小
	std::cout << "New image width: " << newWidth << ", New image height: " << newHeight << std::endl;
}

void Utils::saveArrayAsBinaryRGB(const char* filename, float* data, int minX, int minY, int maxX, int maxY, int imageWidth, int imageHeight) {
	std::ofstream file(filename, std::ios::out | std::ios::binary);
	if (!file) {
		std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
		return;
	}

	// 计算新的宽度和高度
	int newWidth = maxX - minX + 1;
	int newHeight = maxY - minY + 1;

	// 创建缓冲区用于存储R, G, B通道的数据
	float* rChannel = new float[newWidth * newHeight];
	float* gChannel = new float[newWidth * newHeight];
	float* bChannel = new float[newWidth * newHeight];

	int index = 0;
	for (int y = minY; y <= maxY; ++y) {
		for (int x = minX; x <= maxX; ++x) {
			int dataIndex = (y * imageWidth + x) * 4; // 原始数据索引 (RGBA)
			rChannel[index] = data[dataIndex];        // R通道
			gChannel[index] = data[dataIndex + 1];    // G通道
			bChannel[index] = data[dataIndex + 2];    // B通道
			index++;
		}
	}

	// 按顺序写入R, G, B通道数据
	file.write(reinterpret_cast<char*>(rChannel), newWidth * newHeight * sizeof(float));
	file.write(reinterpret_cast<char*>(gChannel), newWidth * newHeight * sizeof(float));
	file.write(reinterpret_cast<char*>(bChannel), newWidth * newHeight * sizeof(float));

	// 清理资源
	delete[] rChannel;
	delete[] gChannel;
	delete[] bChannel;

	file.close();
	if (!file) {
		std::cerr << "Error: Unable to close file " << filename << std::endl;
	}

	// 输出新图像的大小
	std::cout << "New image width : " << newWidth << ", New image height : " << newHeight << std::endl;
}


void Utils::loadBinaryFile(const char* filename, int width, int height, float* output)
{
	std::ifstream file(filename, std::ios::in | std::ios::binary);
	if (!file) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		return;
	}

	// 读取R, G, B通道数据
	float* rChannel = new float[width * height];
	float* gChannel = new float[width * height];
	float* bChannel = new float[width * height];

	file.read(reinterpret_cast<char*>(rChannel), width * height * sizeof(float));
	file.read(reinterpret_cast<char*>(gChannel), width * height * sizeof(float));
	file.read(reinterpret_cast<char*>(bChannel), width * height * sizeof(float));

	// 重新组合数据到output数组中，按顺序存放RGB值
	for (int i = 0; i < width * height; ++i) {
		output[i * 3] = rChannel[i];
		output[i * 3 + 1] = gChannel[i];
		output[i * 3 + 2] = bChannel[i];
	}

	delete[] rChannel;
	delete[] gChannel;
	delete[] bChannel;

	file.close();
}

void Utils::saveImageAsPNG(const char* filename, float* data, int width, int height)
{
	// 将float数据转换为uint8类型
	unsigned char* imageData = new unsigned char[width * height * 3];
	for (int i = 0; i < width * height * 3; ++i) {
		imageData[i] = static_cast<unsigned char>(std::clamp(data[i], 0.0f, 1.0f) * 255);
	}

	// 使用stb_image_write保存为PNG文件
	if (!stbi_write_png(filename, width, height, 3, imageData, width * 3)) {
		std::cerr << "Error: Unable to save image to " << filename << std::endl;
	}

	delete[] imageData;
}


void Utils::splitRGBA(float* h_output, int image_width, int image_height, float*& h_rgb, float*& h_alpha)
{
	int num_pixels = image_width * image_height;
	// 分配内存
	h_rgb = new float[num_pixels * 3];    // 存储RGB，每个像素3个通道
	h_alpha = new float[num_pixels];      // 存储ALPHA，每个像素1个通道

	// 拆分数据
	for (int i = 0; i < num_pixels; ++i) {
		h_rgb[i * 3 + 0] = h_output[i * 4 + 0]; // R
		h_rgb[i * 3 + 1] = h_output[i * 4 + 1]; // G
		h_rgb[i * 3 + 2] = h_output[i * 4 + 2]; // B
		h_alpha[i] = h_output[i * 4 + 3];       // Alpha
	}
}

void Utils::combineRGBA(float* h_rgb, float* h_alpha, int image_width, int image_height, float*& h_output)
{
	int num_pixels = image_width * image_height;
	// 分配内存
	h_output = new float[num_pixels * 4]; // 存储RGBA，每个像素4个通道

	// 合并数据
	for (int i = 0; i < num_pixels; ++i) {
		h_output[i * 4 + 0] = h_rgb[i * 3 + 0]; // R
		h_output[i * 4 + 1] = h_rgb[i * 3 + 1]; // G
		h_output[i * 4 + 2] = h_rgb[i * 3 + 2]; // B
		h_output[i * 4 + 3] = h_alpha[i];       // Alpha
	}
}

void Utils::recordCudaRenderTime(const char* filename, int size, int rank, float elapsedTime)
{
	// 所有进程在写文件前等待同步
    MPI_Barrier(MPI_COMM_WORLD);

	std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);  

    if (rank == 0 && outfile.tellp() == 0)
		outfile << "进程总数  进程ID  时间(ms)" << std::endl;
	else 
        outfile << std::endl;

	outfile << size << " " << rank << " " << elapsedTime << std::endl;

    outfile.close();
	// 确保所有进程在写完后同步
    MPI_Barrier(MPI_COMM_WORLD);
}

void Utils::recordCudaRenderTime(const char* filename, std::string title, std::string iterator, float elapsedTime)
{

	std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);  

	outfile << iterator << " " << title << " " << elapsedTime << std::endl;

    outfile.close();

}
void Utils::recordByte(const char* filename, std::string title, std::string iterator, size_t byte)
{

	std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);  

	outfile << iterator << " " << title << " " << byte << std::endl;

    outfile.close();

}

void Utils::recordBSETime(const char* filename, int u, float elapsedTime, float compressTime, float decompressTime)
{
	std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);  
	if (compressTime == -1 && decompressTime == -1)
		outfile << "  " <<  u << " " << "0.0" << " " << "0.0" << " " << elapsedTime << std::endl;
	else
		outfile << "  " <<  u << " " << compressTime << " " << decompressTime << " " << elapsedTime << std::endl;
	
    outfile.close();
}
void Utils::AddEndline(const char* filename)
{
	std::ofstream outfile;
	outfile.open(filename, std::ios_base::app);  
	outfile << std::endl;
	outfile.close();
}

void Utils::recordTotalTime(const char* filename, float elapsedTime, std::string iteration)
{
	std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);  

    

	outfile << iteration << " "  << elapsedTime << std::endl;

    outfile.close();
}

void Utils::convertRGBtoRRRGGGBBB(float* src_buffer, size_t buffer_len, float* dst_buffer)
{
	size_t num_pixels = buffer_len / 3;
    size_t r_index = 0;
    size_t g_index = num_pixels;
    size_t b_index = 2 * num_pixels;

    // 遍历 src_buffer，并将数据重新分配到新的顺序中
    for (size_t i = 0; i < num_pixels; ++i)
    {
        dst_buffer[r_index++] = src_buffer[3 * i + 0];     // R分量
        dst_buffer[g_index++] = src_buffer[3 * i + 1]; // G分量
        dst_buffer[b_index++] = src_buffer[3 * i + 2]; // B分量
    }
}

void Utils::convertRRRGGGBBBtoRGB(float* src_buffer, size_t buffer_len, float* dst_buffer)
{
	size_t num_pixels = buffer_len / 3;
    size_t r_index = 0;
    size_t g_index = num_pixels;
    size_t b_index = 2 * num_pixels;

    // 遍历 src_buffer，并将数据恢复为 rgb rgb rgb 的顺序
    for (size_t i = 0; i < num_pixels; ++i)
    {
        dst_buffer[3 * i + 0] = src_buffer[r_index++]; // R分量
        dst_buffer[3 * i + 1] = src_buffer[g_index++]; // G分量
        dst_buffer[3 * i + 2] = src_buffer[b_index++]; // B分量
    }
}

void Utils::parseArguments(int argc, char** argv, 
					std::string& volumeFilename, int& xdim, int& ydim, int& zdim, 
                    unsigned int& image_width, unsigned int& image_height, 
					float& cam_dx, float& cam_dy, float& cam_dz,
                    bool& usecompress, bool& useeffarea, 
					std::string& iteration_str)
{
	std::string default_volumeFilename = "./res/Bucky.raw";
	volumeFilename = default_volumeFilename;
	xdim = 32; ydim = 32; zdim = 32;
	image_width = 512; image_height = 512;
	cam_dx = 1.0f; cam_dy = 1.0f; cam_dz = 0.35f;
	usecompress = 0; useeffarea = 0;
	iteration_str = "1";

	if (argc >= 2 ) volumeFilename = argv[1];
	if (argc >= 5 ) 
	{
		xdim = atoi(argv[2]);
		ydim = atoi(argv[3]);
		zdim = atoi(argv[4]);
	}
	if(argc >= 7)
	{
		image_width = atoi(argv[5]);
		image_height = atoi(argv[6]);
	}
	if (argc >= 10)
	{
		cam_dx = atof(argv[7]);
		cam_dy = atof(argv[8]);
		cam_dz = atof(argv[9]);
	}
	if (argc >= 12)
	{
		usecompress = atoi(argv[10]);
		useeffarea = atoi(argv[11]);
	}
	if (argc >= 13)
	{
		iteration_str = argv[12];
	}
}