/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 
 */

#pragma once

#include "SegmentationResult.h"
#include "../FrameData.h"

class GPUTexture;

class SegmentationPerformer {
 public:

    virtual SegmentationResult performSegmentation(std::list<std::shared_ptr<Model>>& models,
                                              FrameDataPointer frame,
                                              unsigned char nextModelID,
                                              bool allowNew) = 0;

    virtual std::vector<std::pair<std::string, std::shared_ptr<GPUTexture>>> getDrawableTextures() { return {}; }

    inline void setNewModelMinRelativeSize(float v) { minRelSizeNew = v; }
    inline void setNewModelMaxRelativeSize(float v) { maxRelSizeNew = v; }

 protected:
    // post-processing
    float maxRelSizeNew = 0.4;
    float minRelSizeNew = 0.07;
};
