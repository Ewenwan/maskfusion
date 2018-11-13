/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 
 */

#pragma once
#include <Eigen/Core>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include <memory>
#include "../Utils/BoundingBox.h"

class Model;
typedef std::list<std::shared_ptr<Model>> ModelList;// 模型指针 列表
typedef ModelList::iterator ModelListIterator;// 迭代器

// TODO Separate specific data (CRF) to somewhere else


struct SegmentationResult 
{
  cv::Mat fullSegmentation;

  bool hasNewLabel = false;
  float depthRange;

  // Optional
  cv::Mat lowCRF;
  cv::Mat lowRGB;
  cv::Mat lowDepth;

  struct ModelData 
  {
    // Warning order makes a difference here!
    unsigned id; // FIXME this should not be part of the result
    ModelListIterator modelListIterator;

    cv::Mat lowICP;
    cv::Mat lowConf;

    bool isNonStatic = false;
    bool isEmpty = true;
    unsigned superPixelCount = 0; // TODO refactor this
    unsigned pixelCount = 0;
    float avgConfidence = 0;
    int classID = -1;

    float depthMean = 0;
    float depthStd = 0;

    // The following values are only approximations:
    BoundingBox boundingBox;

    // Required for partially supported C++14 (in g++ 4.9.4)
    ModelData(unsigned t_id);

    ModelData(unsigned t_id, 
              ModelListIterator const& t_modelListIterator, 
              cv::Mat const& t_lowICP, 
              cv::Mat const& t_lowConf,
              unsigned t_superPixelCount = 0, 
              float t_avgConfidence = 0);
  };
  std::vector<ModelData> modelData;
};
