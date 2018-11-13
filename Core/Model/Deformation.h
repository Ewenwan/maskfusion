/*
 * This file is part of ElasticFusion.
 * 
 *
 */

#pragma once

#include "../Utils/DeformationGraph.h"
#include "../Shaders/Shaders.h"
#include "../Shaders/Uniform.h"
#include "../Shaders/Vertex.h"
#include "../GPUTexture.h"
#include "../Utils/Resolution.h"
#include "../Utils/Intrinsics.h"
#include "../Ferns.h"

#include <pangolin/gl/gl.h>

#include "Buffers.h"

class Deformation 
{
 public:
  Deformation();
  virtual ~Deformation();

  std::vector<GraphNode*>& getGraph();

  void getRawGraph(std::vector<float>& graph);

  void sampleGraphModel(const OutputBuffer& model);

  void sampleGraphFrom(Deformation& other);

  class Constraint {
   public:
    Constraint(const Eigen::Vector3f& src, 
               const Eigen::Vector3f& target,
               const uint64_t& srcTime,
               const uint64_t& targetTime,
               const bool relative, const bool pin = false)
        : src(src),
          target(target),
          srcTime(srcTime),
          targetTime(targetTime),
          relative(relative),
          pin(pin),
          srcPointPoolId(-1),
          tarPointPoolId(-1) {}

    Eigen::Vector3f src;
    Eigen::Vector3f target;
    uint64_t srcTime;
    uint64_t targetTime;
    bool relative;
    bool pin;
    int srcPointPoolId;
    int tarPointPoolId;
  };

  void addConstraint(const Eigen::Vector4f& src, 
                     const Eigen::Vector4f& target, 
                     const uint64_t& srcTime, 
                     const uint64_t& targetTime,
                     const bool pinConstraints);

  void addConstraint(const Constraint& constraint);

  bool constrain(std::vector<Ferns::Frame*>& ferns,
                 std::vector<float>& rawGraph,
                 int time, const bool fernMatch,
                 /*std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > & poseGraph,*/
                 const bool relaxGraph, 
                 std::vector<Constraint>* newRelativeCons = 0);

  Eigen::Vector4f* getVertices() { return vertices; }

  int getCount() { return int(count); }

  int getLastDeformTime() { return lastDeformTime; }

 private:
  DeformationGraph def;

  std::vector<unsigned long long int> vertexTimes;
  std::vector<Eigen::Vector3f> pointPool;
  int originalPointPool;
  int firstGraphNode;

  std::shared_ptr<Shader> sampleProgram;
  GLuint vbo;
  GLuint fid;
  const int bufferSize;
  GLuint countQuery;
  unsigned int count;
  Eigen::Vector4f* vertices;

  std::vector<std::pair<uint64_t, Eigen::Vector3f> > poseGraphPoints;
  std::vector<unsigned long long int> graphPoseTimes;
  std::vector<Eigen::Vector3f>* graphPosePoints;

  std::vector<Constraint> constraints;
  int lastDeformTime;
};
