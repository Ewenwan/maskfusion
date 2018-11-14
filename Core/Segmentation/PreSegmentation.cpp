/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 预分割==================
 */

#include <list>
#include <tuple>

#include "../Model/Model.h"
#include "PreSegmentation.h"


PreSegmentation::PreSegmentation(){}
PreSegmentation::~PreSegmentation() {}

SegmentationResult PreSegmentation::performSegmentation(std::list<std::shared_ptr<Model> > &models,
                                                       FrameDataPointer frame,
                                                       unsigned char nextModelID,
                                                       bool allowNew)
{
    assert(frame->mask.type() == CV_8UC1);
    assert(frame->mask.isContinuous());
    static std::vector<unsigned char> mapping(256, 0);  // FIXME

    SegmentationResult result;
    result.hasNewLabel = false;
    result.fullSegmentation = cv::Mat::zeros(frame->mask.rows, frame->mask.cols, CV_8UC1);

    unsigned char modelIdToIndex[256];
    unsigned char mIndex = 0;
    for (auto m : models) modelIdToIndex[m->getID()] = mIndex++;
    modelIdToIndex[nextModelID] = mIndex;

    std::vector<unsigned> outIdsArray(256, 0);  // Should be faster than using a set

    // Replace unseen with zeroes (except new label)
    for (unsigned i = 0; i < frame->mask.total(); i++) 
    {
      unsigned char& vIn = frame->mask.data[i];
      if (vIn) 
      {
        unsigned char& vOut = result.fullSegmentation.data[i];
        if (mapping[vIn] != 0)
        {
          vOut = mapping[vIn];
          outIdsArray[vOut]++;
        } 
        else if (allowNew && !result.hasNewLabel)
        {
          vOut = nextModelID;
          mapping[vIn] = nextModelID;
          result.hasNewLabel = true;
          outIdsArray[vOut]++;
        }
      }
      else 
      {
        outIdsArray[0]++;
      }
    }

    for (ModelListIterator m = models.begin(); m != models.end(); m++)
      result.modelData.push_back({(*m)->getID(), m, cv::Mat(), cv::Mat(), outIdsArray[(*m)->getID()] / (16 * 16), 0.4});
    if (result.hasNewLabel)
      result.modelData.push_back({nextModelID, ModelListIterator(), cv::Mat(), cv::Mat(),
                                  unsigned(std::max((float)(outIdsArray[nextModelID] / (16 * 16)), 1.0f)), 0.4});

    std::vector<unsigned> cnts(result.modelData.size(), 0);
    for (unsigned i = 0; i < frame->mask.total(); i++) 
    {
      const size_t index = modelIdToIndex[result.fullSegmentation.data[i]];
      result.modelData[index].depthMean += ((const float*)frame->depth.data)[i];
      cnts[index]++;
    }
    for (size_t index = 0; index < result.modelData.size(); ++index) result.modelData[index].depthMean /= cnts[index] ? cnts[index] : 1;

    for (unsigned i = 0; i < frame->mask.total(); i++) 
    {
      const size_t index = modelIdToIndex[result.fullSegmentation.data[i]];
      result.modelData[index].depthStd += std::abs(result.modelData[index].depthMean - ((const float*)frame->depth.data)[i]);
    }
    for (size_t iindex = 0; iindex < result.modelData.size(); ++iindex)
      result.modelData[iindex].depthStd /= cnts[iindex] ? cnts[iindex] : 1;

    return result;
}
