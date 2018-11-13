/*
 * This file is part of ElasticFusion.
 *
 *
 */

#pragma once

#include <pangolin/pangolin.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgproc/imgproc.hpp>

class GPUTexture {
 public:

  GPUTexture(const int width, const int height,
             const GLint internalFormat, 
             const GLenum format, 
             const GLenum dataType,
             const bool drawAndSampleLinear = false, 
             const bool cuda = false, 
             unsigned int cudaFlags = cudaGraphicsRegisterFlagsReadOnly);

  virtual ~GPUTexture();

  // Cuda functions
  void cudaMap();
  void cudaUnmap();
  cudaArray* getCudaArray();
  const cudaSurfaceObject_t& getCudaSurface();

  int getTypeCV() const;
  cv::Mat downloadTexture();
  void downloadTexture(unsigned char* dst);

  void save(const std::string& file);

  pangolin::GlTexture* texture;

  const bool draw;

  const int width;
  const int height;
  const GLint internalFormat;
  const GLenum format;
  const GLenum dataType;

 private:
  GPUTexture() : texture(nullptr), draw(false),
                 width(0), height(0), internalFormat(0), 
                 format(0), dataType(0), cudaRes(nullptr), cudaSurfHandle(0) {}

  bool cudaIsMapped = false;
  cudaGraphicsResource* cudaRes;
  cudaSurfaceObject_t cudaSurfHandle;
};
