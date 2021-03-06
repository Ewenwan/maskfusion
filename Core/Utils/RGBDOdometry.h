/*
 * This file is part of ElasticFusion.
 * 
 *
 */

#ifndef RGBDODOMETRY_H_
#define RGBDODOMETRY_H_

#include "Stopwatch.h"
#include "../GPUTexture.h"
#include "../Cuda/cudafuncs.cuh"
#include "OdometryProvider.h"
#include "GPUConfig.h"

#include <vector>
#include <vector_types.h>

class RGBDOdometry {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RGBDOdometry(int width, int height, 
               float cx, float cy, 
               float fx, float fy, 
               unsigned char maskID = 0, //=========================add
               float distThresh = 0.10f,  // TODO Check, hardcoded scale?
               float angleThresh = sin(20.f * 3.14159254f / 180.f));

  virtual ~RGBDOdometry();

  //FIXME clean up this mess

  // Prepare current frame data for CUDA ICP execution
  // void initICP(GPUTexture * filteredDepth, const float depthCutoff, GPUTexture * mask); // frame to model
//  void initICP(const std::vector<DeviceArray2D<float> >& depthPyramid, const std::vector<DeviceArray2D<unsigned char> >& maskPyramid,
//               const float depthCutoff);
//  void initICP(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff);

  void initICP(const std::vector<DeviceArray2D<float>>* vertexMapPyramid,
               const std::vector<DeviceArray2D<float>>* normalMapPyramid,
               const std::vector<DeviceArray2D<unsigned char>>* prevMaskPyramid); // frame to model, used normally

  //void generateCurrentMaps(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff); // model to model

  // Prepare model data for CUDA ICP execution. Information from the last frame.
  //void initICPModel(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff, const Eigen::Matrix4f& modelPose);
  void initICPModel(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff, const Eigen::Matrix4f& modelPose);

  void initRGB(GPUTexture* rgb);

  void initRGBModel(GPUTexture* rgb);

  void initFirstRGB(GPUTexture* rgb);

  // Get relative transformation, executes optimisation
  Eigen::Matrix4f getIncrementalTransformation(Eigen::Vector3f& trans, 
                                               Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& rot,
                                               const bool& rgbOnly,
                                               const float& icpWeight, 
                                               const bool& pyramid, 
                                               const bool& fastOdom, 
                                               const bool& so3,
                                               const cudaSurfaceObject_t& icpErrorSurface, 
                                               const cudaSurfaceObject_t& rgbErrorSurface);

  Eigen::MatrixXd getCovariance();

  float lastICPError;
  float lastICPCount;
  float lastRGBError;
  float lastRGBCount;
  float lastSO3Error;
  float lastSO3Count;

  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> lastA;
  Eigen::Matrix<double, 6, 1> lastb;

  static const int NUM_PYRS = 3;

 private:
  void populateRGBDData(GPUTexture* rgb, 
                        DeviceArray2D<float>* destDepths, 
                        DeviceArray2D<unsigned char>* destImages,
                        DeviceArray2D<unsigned char>* destMasks);

  DeviceArray<float> vmaps_tmp;
  DeviceArray<float> nmaps_tmp;

  // Prediction pyramid (projected)
  std::vector<DeviceArray2D<float> > vmaps_g_prev_;
  std::vector<DeviceArray2D<float> > nmaps_g_prev_;

  // Current frame pyramid
//  std::vector<DeviceArray2D<float> > vmaps_curr_;
//  std::vector<DeviceArray2D<float> > nmaps_curr_;
  const std::vector<DeviceArray2D<float>>* vertexMapPyramid;
  const std::vector<DeviceArray2D<float>>* normalMapPyramid;
  const std::vector<DeviceArray2D<unsigned char>>* prevMaskPyramid;

  CameraModel intr;

  DeviceArray<JtJJtrSE3> sumDataSE3;
  DeviceArray<JtJJtrSE3> outDataSE3;
  DeviceArray<int2> sumResidualRGB;

  DeviceArray<JtJJtrSO3> sumDataSO3;
  DeviceArray<JtJJtrSO3> outDataSO3;

  const int sobelSize;
  const float sobelScale;
  const float maxDepthDeltaRGB;
  const float maxDepthRGB;

  std::vector<int2> pyrDims;

  // Used during optimisation, rgb-related
  DeviceArray2D<short> nextdIdx[NUM_PYRS];
  DeviceArray2D<short> nextdIdy[NUM_PYRS];

  // Handle textures logic?
  DeviceArray2D<float> lastDepth[NUM_PYRS];
  DeviceArray2D<float> nextDepth[NUM_PYRS];

  DeviceArray2D<unsigned char> lastMask[NUM_PYRS];
  DeviceArray2D<unsigned char> nextMask[NUM_PYRS];

  DeviceArray2D<unsigned char> lastImage[NUM_PYRS];
  DeviceArray2D<unsigned char> nextImage[NUM_PYRS];
  DeviceArray2D<unsigned char> lastNextImage[NUM_PYRS];

  DeviceArray2D<DataTerm> corresImg[NUM_PYRS];

  DeviceArray2D<float3> pointClouds[NUM_PYRS];

  std::vector<int> iterations;
  std::vector<float> minimumGradientMagnitudes;

  float distThres_;
  float angleThres_;

  Eigen::Matrix<double, 6, 6> lastCov;

  const int width;
  const int height;
  const float cx, cy, fx, fy;

  unsigned char maskID;
};

#endif /* RGBDODOMETRY_H_ */
