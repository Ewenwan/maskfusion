/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 分割==================
 * 深度图双边滤波、阈值二值化、反向 255-x、膨胀、腐蚀、距离凸凹性分割、膨胀腐蚀分割 
 * 
 * 使用周围点 距离图像中心点 像素坐标、颜色、深度值差值作为权值。加权求和深度值后再归一化。
 *  利用 周围9点的 距离、凸凹性计算点的边缘属性 进而 进行分割
 */

#include "cudafuncs.cuh"
#include "convenience.cuh"
#include "operators.cuh"
#include "segmentation.cuh"

// 双边滤波==========是一种可以保边去噪的滤波器==========================
// 使用周围点 距离图像中心点 像素坐标、颜色、深度值差值作为权值。加权求和深度值后再归一化。
__global__ void bilateralFilter_Kernel(int w,
                                       int h,
                                       int radius,
                                       int minValues,
                                       float sigmaDepth,
                                       float sigmaColor,
                                       float sigmaLocation,
                                       const PtrStepSz<uchar4> inputRGBA,
                                       const PtrStepSz<float> inputDepth,
                                       PtrStepSz<float> output)
{
// minValues   没用到....? 
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;

    //output.ptr(y)[x] = inputRGBA.ptr(y)[x].w / 255.0f;//inputDepth.ptr(y)[x];


    //const float sigma_space2_inv_half = 0.024691358; // 0.5 / (sigma_space * sigma_space)
    //const float sigma_color2_inv_half = 555.556; // 0.5 / (sigma_color * sigma_color)
    const float i_sigma_depth_2 = 0.5f / (sigmaDepth*sigmaDepth);// 深度差值 系数  平方导数一半 
    const float i_sigma_color_2 = 0.5f / (sigmaColor*sigmaColor);// 颜色差值 系数
    const float i_sigma_location_2 = 0.5f / (sigmaLocation*sigmaLocation);// 像素坐标差值 系数
    
  // 窗口============================
    const int x1 = max(x-radius,0);
    const int y1 = max(y-radius,0);
    const int x2 = min(x+radius, w-1);
    const int y2 = min(y+radius, h-1);

    float weight, location_diff, color_diff, depth_diff;
    float sum_v = 0;
    float sum_w = 0;

    for(int cy = y1; cy <= y2; ++cy){// 行范围
        for(int cx = x1; cx <= x2; ++cx){// 列范围  边长为 radius的窗口
          // 计算与中心点 cx,cy处的差值=============================
            // 像素坐标 差值 平方和  
            location_diff = (x - cx) * (x - cx) + (y - cy) * (y - cy);// 偏离 图像中心点 的距离平方
            // 颜色 差值 平方和 rgb
            color_diff = (inputRGBA.ptr(y)[x].x - inputRGBA.ptr(cy)[cx].x) * (inputRGBA.ptr(y)[x].x - inputRGBA.ptr(cy)[cx].x) +
                    (inputRGBA.ptr(y)[x].y - inputRGBA.ptr(cy)[cx].y) * (inputRGBA.ptr(y)[x].y - inputRGBA.ptr(cy)[cx].y) +
                    (inputRGBA.ptr(y)[x].z - inputRGBA.ptr(cy)[cx].z) * (inputRGBA.ptr(y)[x].z - inputRGBA.ptr(cy)[cx].z);
            // 深度值差值
            depth_diff = (inputDepth.ptr(y)[x] - inputDepth.ptr(cy)[cx]);
            depth_diff *= depth_diff;
            // 负指数 归一化到 0～1
            weight = exp(-location_diff*i_sigma_location_2 -depth_diff*i_sigma_depth_2 -color_diff*i_sigma_color_2);
            // 中心点深度值 加权求和
            sum_v += weight * inputDepth.ptr(cy)[cx];
            sum_w += weight;// 总权值之和
        }
    }
  
// minValues   没用到....? 
    // TODO if min values
    output.ptr(y)[x] = sum_v / sum_w;// 使用周围点 距离图像中心点 像素坐标、颜色、深度值差值作为权值。家全球和深度值后再归一化

}
// 利用彩色图像中的 颜色差值信息 像素坐标差值信息 和深度图中的深度差值信息 对深度图像进行双边滤波===================
void bilateralFilter(const DeviceArray2D<uchar4> inputRGB,  // 彩色图像
                     const DeviceArray2D<float> inputDepth, // 滤波前深度图像
                     const DeviceArray2D<float> outputDepth,// 滤波后深度图像
                     int radius, int minValues, float sigmaDepth, float sigmaColor, float sigmaLocation)
{

    const int w = inputDepth.cols();// 图像 宽和高
    const int h = inputDepth.rows();
    dim3 block (32, 8);// GPU 线程块
    dim3 grid (1, 1, 1);// 线程格
    grid.x = getGridDim (w, block.x);
    grid.y = getGridDim (h, block.y);
    bilateralFilter_Kernel<<<grid, block>>>(w, h, radius, minValues, sigmaDepth, sigmaColor, sigmaLocation, inputRGB, inputDepth, outputDepth);

    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());

}

// Generate a vertex map 'vmap' based on the depth map 'depth' and camera parameters
// 2Dmap ( 间隔 列数行存储 x，y，z值) 得到 3d坐标点
__device__ __forceinline__ float3 getFloat3(int w, int h, const PtrStepSz<float> img32FC3, int x, int y)
{
    return { img32FC3.ptr(y)[x],      // 间隔 列数行存储 x，y，z值
                img32FC3.ptr(y+h)[x],
                img32FC3.ptr(y+2*h)[x] };
}


// ===================?????????? 凸凹性??
__device__ float getConcavityTerm(int w, int h, 
                                  const PtrStepSz<float> vmap, 
                                  const PtrStepSz<float> nmap, 
                                  const float3& v, 
                                  const float3& n, 
                                  int x_n, int y_n) 
{
    const float3 v_n = getFloat3(w, h, vmap, x_n, y_n);// 3D点
    const float3 n_n = getFloat3(w, h, nmap, x_n, y_n);// 归一化map 相邻三点构成两向量叉乘向量再归一化
    if(dot(v_n-v,n) < 0) return 0;
    return 1-dot(n_n,n);
    //return acos(dot(n_n,n));
}


__device__ float getDistanceTerm(int w, int h, const PtrStepSz<float> vmap, 
                                 const float3& v, const float3& n, int x_n, int y_n)
{
    const float3 v_n = getFloat3(w, h, vmap, x_n, y_n);
    float3 d = v_n - v;
    return fabs(dot(d, n));
}

// 这里估计 借鉴了 3d点云分割算法(pcl中)
// https://blog.csdn.net/hjwang1/article/details/78607038?utm_source=blogxgwz4
// 利用 周围9点的 距离、凸凹性计算点的边缘属性 进而 进行分割=============
//__global__ void computeGeometricSegmentation_Kernel(int w, int h, const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, cudaSurfaceObject_t output, float threshold)
//__global__ void computeGeometricSegmentation_Kernel(int w, int h, const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, PtrStepSz<unsigned char> output, float threshold){
__global__ void computeGeometricSegmentation_Kernel(int w, int h,
                                                    const PtrStepSz<float> vmap,
                                                    const PtrStepSz<float> nmap,
                                                    PtrStepSz<float> output,
                                                    float wD, float wC){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;

    const int radius = 1;
    if (x < radius || x >= w-radius || y < radius || y >= h-radius){
        //surf2Dwrite(1.0f, output, x_out, y);
        output.ptr(y)[x] = 1.0f;
        //output.ptr(y)[x] = 255;
        return;
    }

    //TODO handle special case: missing depth!

    const float3 v = getFloat3(w, h, vmap, x, y);// 中心点 3D点
    const float3 n = getFloat3(w, h, nmap, x, y);// 中心点 归一化3D点
    if(v.z <= 0.0f) { // 深度值 小于0为 噪点====
        output.ptr(y)[x] = 1.0f;
        return;
    }

    float c = 0.0f;
  // 周围9点(8+1) 凸凹性 最大值??????????????=============================
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x-radius, y-radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x, y-radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x+radius, y-radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x-radius, y), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x+radius, y), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x-radius, y+radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x, y+radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x+radius, y+radius), c);
    c = fmax(c,0.0f);// 多余...?
    c *= wC;
    //    if(c < 0.99) c = 0;
    //    else c = 1;

    float d = 0.0f;
  // 周围9点(8+1) 与中心点 距离  最大值??????????????=============================
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x-radius, y-radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x, y-radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x+radius, y-radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x-radius, y), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x+radius, y), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x-radius, y+radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x, y+radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x+radius, y+radius), d);
    d *= wD;

    float edgeness = max(c,d);// 边缘特性

    //surf2Dwrite(edgeness, output, x_out, y);
    output.ptr(y)[x] = fmin(1.0f, edgeness);
}
// 利用 周围9点的 距离、凸凹性计算点的边缘属性 进而 进行分割
void computeGeometricSegmentationMap(const DeviceArray2D<float> vmap,
                                     const DeviceArray2D<float> nmap,
                                     const DeviceArray2D<float> output,
                                     float wD, float wC){
    const int w = vmap.cols();
    const int h = vmap.rows() / 3;
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (w, block.x);
    grid.y = getGridDim (h, block.y);
    //std::cout << "Running block, info: " << grid.x << " " << grid.y << " - wh: " << w << " " << h << std::endl;
    computeGeometricSegmentation_Kernel<<<grid, block>>>(w, h, vmap, nmap, output, wD, wC);

    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}



// 腐蚀=====周围最小值====浮点数==================
__global__ void f_erode_Kernel(int w, int h, 
                               const PtrStepSz<float> intput, 
                               PtrStepSz<float> output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;

    float r = intput.ptr(y)[x];// 中心点深度值
    if (x < 1 || x >= w-1 || y < 1 || y >= h-1) 
    {
      output.ptr(y)[x] = r;
      return;// 这里一个bug==========!!!!!!!!!!!!!!!!!!!=============
    }
  
  // 周围9点 深度值最小值=============================
    r = fmin(intput.ptr(y-1)[x-1], r);
    r = fmin(intput.ptr(y-1)[x], r);
    r = fmin(intput.ptr(y-1)[x+1], r);
    r = fmin(intput.ptr(y)[x-1], r);
    r = fmin(intput.ptr(y)[x+1], r);
    r = fmin(intput.ptr(y+1)[x-1], r);
    r = fmin(intput.ptr(y+1)[x], r);
    r = fmin(intput.ptr(y+1)[x+1], r);
    output.ptr(y)[x] = r;
}
// 腐蚀=====周围最小值====0～255====================不过怎么感觉像是 二值化
__global__ void erode_Kernel(int w, int h, int radius, 
                             const PtrStepSz<unsigned char> intput, 
                             PtrStepSz<unsigned char> output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;
    const int x1 = max(x-radius,0);
    const int y1 = max(y-radius,0);
    const int x2 = min(x+radius, w-1);
    const int y2 = min(y+radius, h-1);
    output.ptr(y)[x] = 255;
    for (int cy = y1; cy <= y2; ++cy)
    {
        for (int cx = x1; cx <= x2; ++cx)
        {
            if (cy == y && cx == x) continue;
            if (intput.ptr(cy)[cx] == 0) 
            {
                output.ptr(y)[x] = 0;
                return;
            }
        }
    }
}

// 膨胀====周围最大值======浮点数=============
__global__ void f_dilate_Kernel(int w, int h, const PtrStepSz<float> intput, PtrStepSz<float> output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;

    float r = intput.ptr(y)[x];
    if (x < 1 || x >= w-1 || y < 1 || y >= h-1) 
    {
      output.ptr(y)[x] = r;
      return;// 这里一个bug==========!!!!!!!!!!!!!!!!!!!=============
    }
  // 周围9点 深度值最大值=============================
    r = fmax(intput.ptr(y-1)[x-1], r);
    r = fmax(intput.ptr(y-1)[x], r);
    r = fmax(intput.ptr(y-1)[x+1], r);
    r = fmax(intput.ptr(y)[x-1], r);
    r = fmax(intput.ptr(y)[x+1], r);
    r = fmax(intput.ptr(y+1)[x-1], r);
    r = fmax(intput.ptr(y+1)[x], r);
    r = fmax(intput.ptr(y+1)[x+1], r);
    output.ptr(y)[x] = r;
}
// 膨胀====周围最大值======0～255=============不过怎么感觉像是 二值化
__global__ void dilate_Kernel(int w, int h, int radius, 
                              const PtrStepSz<unsigned char> intput, 
                              PtrStepSz<unsigned char> output) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;
    const int x1 = max(x-radius,0);
    const int y1 = max(y-radius,0);
    const int x2 = min(x+radius, w-1);
    const int y2 = min(y+radius, h-1);
    output.ptr(y)[x] = 0;
    for (int cy = y1; cy <= y2; ++cy)
    {
        for (int cx = x1; cx <= x2; ++cx)
        {
            if (cy == y && cx == x) continue;
            if (intput.ptr(cy)[cx] == 255) 
            {
                output.ptr(y)[x] = 255;
                return;
            }
        }
    }
}

// 阈值二值化=============================================
__global__ void threshold_Kernel(const PtrStepSz<float> input, 
                                 PtrStepSz<unsigned char> output, 
                                 float threshold)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= input.rows || x >= input.cols) return;
    output.ptr(y)[x] = input.ptr(y)[x] > threshold ? 255 : 0;
}
// 阈值二值化
void thresholdMap(const DeviceArray2D<float> input,
                  const DeviceArray2D<unsigned char> output,
                  float threshold){
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (input.cols(), block.x);
    grid.y = getGridDim (input.rows(), block.y);
    threshold_Kernel<<<grid, block>>>(input, output, threshold);
}


// 反向 255-x============================================
__global__ void invert_Kernel(const PtrStepSz<unsigned char> input, 
                              PtrStepSz<unsigned char> output) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= input.rows || x >= input.cols) return;
    output.ptr(y)[x] = 255 - input.ptr(y)[x];
}
// 反向
void invertMap(const DeviceArray2D<unsigned char> input,
               const DeviceArray2D<unsigned char> output){
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (input.cols(), block.x);
    grid.y = getGridDim (input.rows(), block.y);
    invert_Kernel<<<grid, block>>>(input, output);
}


//__global__ void morphGeometricSegmentation_Kernel(int w, int h, const PtrStepSz<float> input, const PtrStepSz<float> output)
//{

//}





// 使用 3次膨胀、腐蚀 来分割=============================
void morphGeometricSegmentationMap(const DeviceArray2D<float> data,
                                   const DeviceArray2D<float> buffer){
    const int w = data.cols();
    const int h = data.rows();
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (w, block.x);
    grid.y = getGridDim (h, block.y);
    f_dilate_Kernel<<<grid, block>>>(w, h, data, buffer);//膨胀
    f_erode_Kernel<<<grid, block>>>(w, h, buffer, data);//腐蚀
    f_dilate_Kernel<<<grid, block>>>(w, h, data, buffer);
    f_erode_Kernel<<<grid, block>>>(w, h, buffer, data);
    f_dilate_Kernel<<<grid, block>>>(w, h, data, buffer);
    f_erode_Kernel<<<grid, block>>>(w, h, buffer, data);

    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}

void morphGeometricSegmentationMap(const DeviceArray2D<unsigned char> data,
                                const DeviceArray2D<unsigned char> buffer,
                                int radius,
                                int iterations){

    const int w = data.cols();
    const int h = data.rows();
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (w, block.x);
    grid.y = getGridDim (h, block.y);
    for (int i = 0; i < iterations; ++i) {
        dilate_Kernel<<<grid, block>>>(w, h, radius, data, buffer);
         cudaSafeCall (cudaDeviceSynchronize ());
        erode_Kernel<<<grid, block>>>(w, h, radius, buffer, data);
         cudaSafeCall (cudaDeviceSynchronize ());
        //erode_Kernel<<<grid, block>>>(w, h, radius, buffer, data);
    }
    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}



//__global__ void computeGeometricSegmentation_Kernel(const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, PtrStepSz<float> output, float threshold)
//{
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;

//    if (x < 1 || x >= output.cols-1 || y < 1 || y >= output.rows-1){
//        output.ptr(y)[x] = 1;
//        return;
//    }

//    float c = 1;
//    c = min(getConvexityTerm(vmap, nmap, x, x-1, y, y-1), c);

//    output.ptr(y)[x] = c;
//}

//void computeGeometricSegmentationMap(const DeviceArray2D<float> vmap,
//                                     const DeviceArray2D<float> nmap,
//                                     DeviceArray2D<float> segmentationMap,
//                                     float threshold){
//    dim3 block (32, 8);
//    dim3 grid (1, 1, 1);
//    grid.x = getGridDim (segmentationMap.cols (), block.x);
//    grid.y = getGridDim (segmentationMap.rows (), block.y);
//    computeGeometricSegmentation_Kernel<<<grid, block>>>(vmap, nmap, segmentationMap, threshold);

//    //cudaSafeCall (cudaGetLastError ());

//    cudaCheckError();
//    cudaSafeCall (cudaDeviceSynchronize ());
//}
