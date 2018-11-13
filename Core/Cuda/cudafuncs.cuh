/*
 * This file is part of ElasticFusion.
 * 
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef CUDA_CUDAFUNCS_CUH_
#define CUDA_CUDAFUNCS_CUH_

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

#include "containers/device_array.hpp"
#include "types.cuh"

// ICP配准============================
void icpStep(const mat33& Rcurr,
             const float3& tcurr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr,
             const mat33& Rprev_inv,
             const float3& tprev,
             const CameraModel& intr,
             const DeviceArray2D<float>& vmap_g_prev,
             const DeviceArray2D<float>& nmap_g_prev,
             float distThres,// 距离阈值
             float angleThres,// 角度阈值
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads,
             int blocks,           
// =============== 比 ElasticFusion多的地方==============
             const cudaSurfaceObject_t& icpErrorSurface,
             const DeviceArray2D<unsigned char>& nextMask,
             unsigned char maskID);

// 
void rgbStep(const DeviceArray2D<DataTerm> & corresImg,
             const float & sigma,
             const DeviceArray2D<float3> & cloud,
             const float & fx,
             const float & fy,
             const DeviceArray2D<short> & dIdx,
             const DeviceArray2D<short> & dIdy,
             const float & sobelScale,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             int threads,
             int blocks);
// 
void so3Step(const DeviceArray2D<unsigned char> & lastImage,
             const DeviceArray2D<unsigned char> & nextImage,
             const mat33 & imageBasis,
             const mat33 & kinv,
             const mat33 & krlr,
             DeviceArray<JtJJtrSO3> & sum,
             DeviceArray<JtJJtrSO3> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads,
             int blocks);

// 残差=================================
void computeRgbResidual(const float & minScale,
                        const DeviceArray2D<short> & dIdx,
                        const DeviceArray2D<short> & dIdy,
                        const DeviceArray2D<float> & lastDepth,
                        const DeviceArray2D<float> & nextDepth,
                        const DeviceArray2D<unsigned char> & lastImage,
                        const DeviceArray2D<unsigned char> & nextImage,
                    // ===============
                        const DeviceArray2D<unsigned char> & lastMask,
                        const DeviceArray2D<unsigned char> & nextMask,
                    // ===============
                        DeviceArray2D<DataTerm> & corresImg,
                        DeviceArray<int2> & sumResidual,
                        const float maxDepthDelta,
                        const float3 & kt,
                        const mat33 & krkinv,
                        int & sigmaSum,
                        int & count,
                        int threads,
                        int blocks,
                    // ===============
                        const cudaSurfaceObject_t& icpErrorSurface,
                        unsigned char maskID);

// 深度图 创建 3D点 地图============================
// 一行深度变3行X，Y，Z=============================
void createVMap(const CameraModel& intr,
                const DeviceArray2D<float> & depth,
                DeviceArray2D<float> & vmap,
                const float depthCutoff);
// 3D点 地图归一化，减去均值，除方差==================
void createNMap(const DeviceArray2D<float>& vmap,
                DeviceArray2D<float>& nmap);
// 3D点地图 刚体变换================================
void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src,
                  const mat33& Rmat,
                  const float3& tvec,
                  DeviceArray2D<float>& vmap_dst,
                  DeviceArray2D<float>& nmap_dst);
// 复制3D点云map地图
void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst);
// 改变map形状
void resizeVMap(const DeviceArray2D<float>& input,
                DeviceArray2D<float>& output);
// 改变归一化map形状
void resizeNMap(const DeviceArray2D<float>& input,
                DeviceArray2D<float>& output);
// 彩色图转会赌徒
void imageBGRToIntensity(cudaArray * cuArr,
                         DeviceArray2D<unsigned char> & dst);
// 2Dmap 转深度值
void verticesToDepth(DeviceArray<float>& vmap_src,
                     DeviceArray2D<float> & dst,
                     float cutOff);

// 2D深度值 转 3D地图
// 2D to 3D: input is depth image, output is cloud
void projectToPointCloud(const DeviceArray2D<float> & depth,
                         const DeviceArray2D<float3> & cloud,
                         CameraModel & intrinsics,
                         const int & level);
// 金字塔下采样
void pyrDown(const DeviceArray2D<unsigned short> & src,
             DeviceArray2D<unsigned short> & dst);


//FIXME
//void testCuda(cudaSurfaceObject_t surface);

// 5×5高斯核下采样 浮点数
void pyrDownGaussF(const DeviceArray2D<float> & src,
                   DeviceArray2D<float> & dst);

// 5×5高斯核下采样 0～255
void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src,
                       DeviceArray2D<unsigned char> & dst);

//void pyrDown2(const DeviceArray2D<unsigned char> & src,
//             DeviceArray2D<unsigned char> & dst);

// 计算图像水平和垂直梯度
void computeDerivativeImages(DeviceArray2D<unsigned char>& src,
                             DeviceArray2D<short>& dx,
                             DeviceArray2D<short>& dy);

#endif /* CUDA_CUDAFUNCS_CUH_ */
