/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 帧 数据结构====================================================
 */

#pragma once

#include <stdint.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <utility>
#include <memory>

struct FrameData 
{
  // Allocate memory for rgb and depth image
  void allocateRGBD(unsigned width, unsigned height) 
  {
    rgb = cv::Mat(height, width, CV_8UC3);// 0～255 RGB 数据
    depth = cv::Mat(height, width, CV_32FC1);// float 数据 深度值
  }

  int64_t timestamp = 0;
  int64_t index = 0;

  cv::Mat mask;   // 语义分割mask 0~255 External segmentation (optional!), CV_8UC1
  cv::Mat rgb;    // 色彩 RGB data, CV_8UC3
  cv::Mat depth;  // 深度值   Depth data, CV_32FC1

  std::vector<int> classIDs;  // 目标类别id 序列 It is assumed that mask-labels are consecutive and that classIDs[mask.data[i]] provides the class for each pixel in the mask.
  std::vector<cv::Rect> rois; // 目标 框 序列

  // RGB 变 BGR===========================
  void flipColors() 
  {
#pragma omp parallel for
    for (unsigned i = 0; i < rgb.total() * 3; i += 3) std::swap(rgb.data[i + 0], rgb.data[i + 2]);
  }
};

typedef std::shared_ptr<FrameData> FrameDataPointer;
