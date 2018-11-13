/*
 * This file is part of https://github.com/martinruenz/maskfusion
 *  分割==================
 *  深度图双边滤波、阈值二值化、反向 255-x、膨胀、腐蚀、距离凸凹性分割、膨胀腐蚀分割
 */



#ifndef CUDA_SEGMENTATION_CUH_
#define CUDA_SEGMENTATION_CUH_

// Normal and verterx maps should already be available, see nmaps_curr_, vmaps_curr_
// Also, already generated, since performTracking was called. (initialises values)
#include "containers/device_array.hpp"
#include "types.cuh"

//  利用 周围9点的 距离、凸凹性计算点的边缘属性 进而 进行分割
void computeGeometricSegmentationMap(const DeviceArray2D<float> vmap,
                                     const DeviceArray2D<float> nmap,
                                     const DeviceArray2D<float> output,
                                     //const DeviceArray2D<unsigned char> output,
                                     float wD, float wC);
// 阈值二值化
void thresholdMap(const DeviceArray2D<float> input,
                  const DeviceArray2D<unsigned char> output,
                  float threshold);
// 反向 255-x
void invertMap(const DeviceArray2D<unsigned char> input,
               const DeviceArray2D<unsigned char> output);
// 使用 3次膨胀、腐蚀 来分割
void morphGeometricSegmentationMap(const DeviceArray2D<float> data,
                                   const DeviceArray2D<float> buffer);

void morphGeometricSegmentationMap(const DeviceArray2D<unsigned char> data,
                                   const DeviceArray2D<unsigned char> buffer,
                                   int radius,
                                   int iterations);
// 深度图双边滤波：使用周围点 距离图像中心点 像素坐标、颜色、深度值差值作为权值。加权求和深度值后再归一化。
void bilateralFilter(const DeviceArray2D<uchar4> inputRGB,
                     const DeviceArray2D<float> inputDepth,
                     const DeviceArray2D<float> outputDepth,
                     int radius, int minValues, float sigmaDepth, float sigmaColor, float sigmaLocation);

#endif
