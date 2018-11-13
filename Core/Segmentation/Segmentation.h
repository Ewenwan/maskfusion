/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 
 */

#pragma once

#include "../Cuda/types.cuh"
#include "Slic.h"
#include "SegmentationPerformer.h"
#include "MfSegmentation.h"
#include "CfSegmentation.h"
#include <Eigen/Core>
#include <thread>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>

class Model;
class GPUTexture;

class Segmentation
{
 public:
  enum class Method { CO_FUSION, MASK_FUSION, PRECOMPUTED };

  void init(int width,
            int height,
            Method method,
            const CameraModel& cameraIntrinsics,
            std::shared_ptr<GPUTexture> textureRGB,
            std::shared_ptr<GPUTexture> depthMetric,
            bool usePrecomputedMasks,
            GlobalProjection* globalProjection,
            std::queue<FrameDataPointer>* pQueue = NULL);

  std::vector<std::pair<std::string, std::shared_ptr<GPUTexture> > > getDrawableTextures();

  SegmentationResult performSegmentation(std::list<std::shared_ptr<Model>>& models, 
                                         FrameDataPointer frame, unsigned char nextModelID,
                                         bool allowNew);

//  SegmentationResult performSegmentationPrecomputed(std::list<std::shared_ptr<Model>>& models, FrameDataPointer frame, unsigned char nextModelID,
//                                         bool allowNew);

  MfSegmentation* getMfSegmentationPerformer() { return dynamic_cast<MfSegmentation*>(segmentationPerformer.get()); }
  CfSegmentation* getCfSegmentationPerformer() { return dynamic_cast<CfSegmentation*>(segmentationPerformer.get()); }
  SegmentationPerformer* getSegmentationPerformer() { return segmentationPerformer.get(); }

  void cleanup();

 private:

  // General
  Method method;

  std::unique_ptr<SegmentationPerformer> segmentationPerformer;
};
