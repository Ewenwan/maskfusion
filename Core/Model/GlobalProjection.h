/*
 * This file is part of https://github.com/martinruenz/maskfusion
 * 
 */


#pragma once

#include <list>

#include "../Shaders/Shaders.h"
#include "../Shaders/Uniform.h"
#include "../Shaders/Vertex.h"
#include "../GPUTexture.h"
#include "../Utils/Resolution.h"
#include "../Utils/Intrinsics.h"
#include <pangolin/gl/gl.h>

#include "Buffers.h"

class Model;

class GlobalProjection 
{
 public:
  GlobalProjection(int w, int h);
  virtual ~GlobalProjection();

  void project(const std::list<std::shared_ptr<Model>>& models, 
               int time, int maxTime, 
               int timeDelta, float depthCutoff);

  void downloadDirect(); // Speed: This could be accelerated with PBOs

  inline cv::Mat getProjectedModelIDs() const { return idBuffer; }
  inline cv::Mat getProjectedDepth() const { return depthBuffer; }

 private:

  std::shared_ptr<Shader> program;
  pangolin::GlFramebuffer framebuffer;
  pangolin::GlRenderBuffer renderbuffer;
  GPUTexture texDepth; // GL_R16F
  GPUTexture texID; // GL_R8UI

  cv::Mat depthBuffer;
  cv::Mat idBuffer;
};
