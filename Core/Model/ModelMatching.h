/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 
 */

#pragma once

#include <memory>

#include "Model.h"

class IModelMatcher {
 public:
  /// Try to detect one of the inactive models in the specified segmented region
  virtual ModelDetectionResult detectInRegion(const FrameData& frame, const cv::Rect& rect) = 0;

  /// Build model description. Returns true if succeeded else false (for instance, because the model is not having enough
  /// points to create descr.)
  virtual bool buildModelDescription(Model* model) = 0;
};

// [Removed matching code]
