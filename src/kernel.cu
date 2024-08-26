#include "kernel.cuh"
__device__ float cubicInterpolate(float p[4], float x) {
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

__device__ float tricubicInterpolate(float p[4][4][4], float x, float y, float z) {
    float arr[4];
    for (int i = 0; i < 4; i++) {
        float arrY[4];
        for (int j = 0; j < 4; j++) {
            arrY[j] = cubicInterpolate(p[i][j], z);
        }
        arr[i] = cubicInterpolate(arrY, y);
    }
    return cubicInterpolate(arr, x);
}

__device__ float getValue(unsigned char* data, int3 size, int x, int y, int z) {
    // 边界检查
    if (x < 0 || x >= size.x || y < 0 || y >= size.y || z < 0 || z >= size.z) {
        return 0.0f; // 或者返回一个默认值
    }
    return static_cast<float>(data[(z * size.y + y) * size.x + x]) / 255.0f; // 将unsigned char数据转换为float，并归一化
}

__device__ float sampleVolume(unsigned char* data, int3 size, float3 texCoord) {
    int x = floor(texCoord.x);
    int y = floor(texCoord.y);
    int z = floor(texCoord.z);
    float dx = texCoord.x - x;
    float dy = texCoord.y - y;
    float dz = texCoord.z - z;

    float p[4][4][4];
    for (int i = -1; i <= 2; i++) {
        for (int j = -1; j <= 2; j++) {
            for (int k = -1; k <= 2; k++) {
                p[i + 1][j + 1][k + 1] = getValue(data, size, x + i, y + j, z + k);
            }
        }
    }

    return tricubicInterpolate(p, dx, dy, dz);
}

__device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax, float* tnear,
    float* tfar) {
    float3 invDir = 1.0f / r.d;
    float3 tminTemp = (boxmin - r.o) * invDir;
    float3 tmaxTemp = (boxmax - r.o) * invDir;

    float3 tmin = fminf(tminTemp, tmaxTemp);
    float3 tmax = fmaxf(tminTemp, tmaxTemp);

    *tnear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    *tfar = fminf(fminf(tmax.x, tmax.y), tmax.z);

    return *tfar >= *tnear;
}


__device__ uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
        (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

#define M_PI 3.14159265358979323846264338327950288f
__device__ float degrees_to_radians(float angleInDegrees) {
    return ((angleInDegrees)*M_PI / 180.0f);
}

__device__ float mocnina2(float x)
{
    return x * x;
}

__device__ float odmocnina2(float x)
{
    return x >= 0.0f ? powf(x, 0.5f) : 0.0f;
}


__device__ float3 fmaf_float3(float3 a, float3 b, float3 c) {
    float3 result;
    result.x = fmaf(a.x, b.x, c.x);
    result.y = fmaf(a.y, b.y, c.y);
    result.z = fmaf(a.z, b.z, c.z);
    return result;
}

__device__ float3 fmaf_float3(float3 a, float b, float3 c) {
    float3 result;
    result.x = fmaf(a.x, b, c.x);
    result.y = fmaf(a.y, b, c.y);
    result.z = fmaf(a.z, b, c.z);
    return result;
}


__global__ void d_render(float* d_output, uint imageW, uint imageH,
    float3 ccamPos, float3 ccamLookAt, float3 ccam_U, float3 ccam_V, float ccam_dz,
    float density, float brightness, float transferOffset, float transferScale, 
    float3 bboxMin, float3 bboxMax, float3 big_bboxMin, float3 big_bboxMax, float3 data_compensation,
    cudaTextureObject_t tex, cudaTextureObject_t transferTex, uint* d_minMaxXY)
{
    const int maxSteps = 1200;
    //const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imageW || y >= imageH) return;
    //if (!(x == 251 && y == 372)) return;
    
    // Camera setup
    float3 camDir = normalize(ccamLookAt - ccamPos);
   
    // Compute normalized device coordinates (NDC)
    float nu = imageW / (2.0f * length(ccam_U)) * 1.0f * ccam_dz / 2.0f;
    float nv = imageH / (2.0f * length(ccam_V)) * 1.0f * ccam_dz / 2.0f;
    
    float cam_u = 2.0f * (float)x / (float)(imageW - 1) - 1.0f;
    float cam_v = 2.0f * (float)y / (float)(imageH - 1) - 1.0f;

    float3 ray_start;
    ray_start.x = nu * ccam_U.x * cam_u + nv * ccam_V.x * cam_v;
    ray_start.y = nu * ccam_U.y * cam_u + nv * ccam_V.y * cam_v;
    ray_start.z = nu * ccam_U.z * cam_u + nv * ccam_V.z * cam_v;
    //printf("%d %d %f %f\n", x, y, nu, nv);

    Ray eyeRay;
    eyeRay.o = ccamPos + ray_start;
    eyeRay.d = camDir;
    float samplingRate = 1.0f;
    float3 delta = eyeRay.d / samplingRate;

    // find intersection with box
    float tnear, tfar;
    float big_tnear, big_tfar;
    
    float3 bigMin = big_bboxMin;
    float3 bigMax = big_bboxMax;


    int hit = intersectBox(eyeRay, bboxMin, bboxMax, &tnear, &tfar);
    int hit_big = intersectBox(eyeRay, bigMin, bigMax, &big_tnear, &big_tfar);

    if (!hit) return;

    //if (tnear < 0.0f) tnear = 0.1f;  // clamp to near plane
    int idx = y * imageW + x;

    float tstep = 0.05;
    float nTimes = ceil((tnear - big_tnear) / tstep);
    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    //float t = big_tnear + (nTimes * tstep);
    float t = fmaf(nTimes, tstep, big_tnear);
  
   
    //float3 pos = eyeRay.o + normalize(eyeRay.d) * t;
    //float3 pos_temp = eyeRay.o + normalize(eyeRay.d) * t;

    float3 pos = fmaf_float3(normalize(eyeRay.d), t, eyeRay.o);
    float3 pos_temp = fmaf_float3(normalize(eyeRay.d), t, eyeRay.o);
    float3 step = normalize(eyeRay.d) * tstep;

    float total_t_size = ceil((tfar - tnear) / (tstep));
    //printf("%f %f %f %f %f\n", big_tnear, tnear, t, nTimes, total_t_size);
    
    int n = nTimes + 1;
    while (t < tfar )
    {
       
        float3 texCoord;
        
       
        pos_temp.x = truncf( (pos.x) * 1000.0f) / 1000.0f;
        pos_temp.y = truncf( (pos.y) * 1000.0f) / 1000.0f;
        pos_temp.z = truncf( (pos.z) * 1000.0f) / 1000.0f;

        /*pos_temp.x = truncf(roundf(pos.x * 10000.0f) / 10.0f) / 1000.0f;
        pos_temp.y = truncf(roundf(pos.y * 10000.0f) / 10.0f) / 1000.0f;
        pos_temp.z = truncf(roundf(pos.z * 10000.0f) / 10.0f) / 1000.0f;*/

        //printf("%d %d |%f \n", x, y, t);
        //printf("%d %d | %f %f %f | %f %f %f | %f\n", x, y, pos.x, pos.y, pos.z, pos_temp.x, pos_temp.y, pos_temp.z, t);
        //printf("%d %d |%f %f %f | %f\n", x, y, pos_temp.x, pos_temp.y, pos_temp.z, t);
        float3 localPos;
        /*if(addOneFlag == 1)
            localPos = pos_temp - bboxMin + data_compensation;
        else
            localPos = pos_temp - bboxMin + make_float3(0, 0, 0);*/
        localPos = pos_temp - bboxMin + data_compensation;
        //float3 localPos = pos_temp - bboxMin + make_float3(0, 0, 0);

        //// 将局部坐标直接作为纹理坐标
        texCoord = localPos;

        //// 确保 texCoord 在合法范围内
        //texCoord.x = fmaxf(fminf(texCoord.x, (bboxMax.x - bboxMin.x)), 0.0f);
        //texCoord.y = fmaxf(fminf(texCoord.y, (bboxMax.y - bboxMin.y)), 0.0f);
        //texCoord.z = fmaxf(fminf(texCoord.z, (bboxMax.z - bboxMin.z)), 0.0f);
        //texCoord = pos_temp + make_float3(16, 16, 16);

        float sample = tex3D<float>(tex, texCoord.x, texCoord.y, texCoord.z);
        //float sample = sampleVolume(d_volume, d_volumeSize, texCoord);

        //printf("%d %d| %f \n", x, y, sample);
        float4 col = tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);
        //printf("%d %d |%f %f %f %f\n", x, y, col.x, col.y, col.z, col.w);
        col.w *= density;
        //printf("%d %d |%f %f %f %f\n", x, y, col.x, col.y, col.z, col.w);
      
        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        //printf("%d %d |%f %f %f %f\n", x, y, col.x, col.y, col.z, col.w);
        // "over" operator for front-to-back blending

        //float tmpW = 1.0 - sum.w;
        //printf("%d %d |%f %f %f %f | %f\n", x, y, col.x, col.y, col.z, col.w, tmpW);
        
        sum = sum + col * (1.0f - sum.w);
        //printf("sum %d %d |%f %f %f %f\n", x, y, sum.x, sum.y, sum.z, sum.w);
        // exit early if opaque
        //if (sum.w > opacityThreshold) break;

        t = fmaf(n, tstep, big_tnear);
        n++;
        if (t > tfar) break;

        //pos += step;
        pos = fmaf_float3(pos, 1.0, step);
        //pos = fmaf_float3(normalize(eyeRay.d), t, eyeRay.o + step);
    }

    sum *= brightness;


    d_output[idx * 4 + 0] = sum.x;  // R
    d_output[idx * 4 + 1] = sum.y;  // G
    d_output[idx * 4 + 2] = sum.z;  // B
    d_output[idx * 4 + 3] = sum.w;  // A

    // Store min and max positions if pixel intersects AABB
    atomicMin(&d_minMaxXY[0], x);
    atomicMin(&d_minMaxXY[1], y);
    atomicMax(&d_minMaxXY[2], x);
    atomicMax(&d_minMaxXY[3], y);
}



CudaKernel::CudaKernel() : d_volumeArray(nullptr), d_transferFuncArray(nullptr), texObject(0), transferTex(0)
{
}

CudaKernel::~CudaKernel()
{
	freeCudaBuffers();
}

void CudaKernel::initCuda(void* h_volume, cudaExtent volumeSize)
{

    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr =
        make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(VolumeType),
            volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_volumeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;  // access with normalized texture coordinates
    texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

    texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(
        cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

    // create transfer function texture
    /*float4 transferFunc[] = {
        {0.231373,  0.298039,   0.752941,   0.0 },  
        { 0.865,    0.865,      0.865,      0.5 }, 
        { 0.705882, 0.0156863,  0.14902,    1.0 }, 
    };*/

    // float4 transferFunc[] = {
    // {0.0f,       0.0f,       0.5f,       0.0f },  // 深蓝色，完全透明
    // {0.0f,       0.5f,       1.0f,       0.3f },  // 浅蓝色，部分透明
    // {0.0f,       1.0f,       0.5f,       0.5f },  // 绿色，半透明
    // {1.0f,       1.0f,       0.0f,       0.7f },  // 黄色，较高透明度
    // {1.0f,       0.0f,       0.0f,       1.0f },  // 红色，完全不透明
    // };

    float4 transferFunc[] = {
    {0.231373f, 0.298039f, 0.752941f, 0.0f },   // 对应数据值 0.0
    {0.864999f, 0.864999f, 0.864999f, 0.0f },   // 对应数据值 127.5
    {0.705882f, 0.015686f, 0.149020f, 1.0f },   // 对应数据值 255.0
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray* d_transferFuncArray;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2,
        sizeof(transferFunc) / sizeof(float4), 1));
    checkCudaErrors(cudaMemcpy2DToArray(d_transferFuncArray, 0, 0, transferFunc,
        0, sizeof(transferFunc), 1,
        cudaMemcpyHostToDevice));

    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_transferFuncArray;

    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords =
        true;  // access with normalized texture coordinates
    texDescr.filterMode = cudaFilterModeLinear;

    texDescr.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates

    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(
        cudaCreateTextureObject(&transferTex, &texRes, &texDescr, NULL));
}

void CudaKernel::freeCudaBuffers()
{
    checkCudaErrors(cudaDestroyTextureObject(texObject));
    checkCudaErrors(cudaDestroyTextureObject(transferTex));
    checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}

void CudaKernel::render_kernel(dim3 gridSize, dim3 blockSize, float* d_output, uint imageW, uint imageH,
    float3 camPos, float3 camLookAt, float3 cam_U, float3 cam_V, float ccam_dz,
    float3 boxMin, float3 boxMax, float3 big_boxMin, float3 big_boxMax, float3 data_compensation,
    float density, float brightness, float transferOffset, float transferScale,
    uint* d_minMaxXY)
{
    d_render << <gridSize, blockSize >> > (d_output, imageW, imageH,
        camPos, camLookAt, cam_U, cam_V, ccam_dz,
        density, brightness, transferOffset, transferScale,
        boxMin, boxMax, big_boxMin, big_boxMax, data_compensation,
        texObject, transferTex, d_minMaxXY);
}
