/*
 * This file is part of https://github.com/martinruenz/maskfusion
 *  分割==================
 *  
 */



#ifndef CUDA_SEGMENTATION_CUH_
#define CUDA_SEGMENTATION_CUH_

// Normal and verterx maps should already be available, see nmaps_curr_, vmaps_curr_
// Also, already generated, since performTracking was called. (initialises values)
#include "containers/device_array.hpp"
#include "types.cuh"

void computeGeometricSegmentationMap(const DeviceArray2D<float> vmap,
                                     const DeviceArray2D<float> nmap,
                                     const DeviceArray2D<float> output,
                                     //const DeviceArray2D<unsigned char> output,
                                     float wD, float wC);

void thresholdMap(const DeviceArray2D<float> input,
                  const DeviceArray2D<unsigned char> output,
                  float threshold);

void invertMap(const DeviceArray2D<unsigned char> input,
               const DeviceArray2D<unsigned char> output);

void morphGeometricSegmentationMap(const DeviceArray2D<float> data,
                                   const DeviceArray2D<float> buffer);

void morphGeometricSegmentationMap(const DeviceArray2D<unsigned char> data,
                                   const DeviceArray2D<unsigned char> buffer,
                                   int radius,
                                   int iterations);

void bilateralFilter(const DeviceArray2D<uchar4> inputRGB,
                     const DeviceArray2D<float> inputDepth,
                     const DeviceArray2D<float> outputDepth,
                     int radius, int minValues, float sigmaDepth, float sigmaColor, float sigmaLocation);

#endif
