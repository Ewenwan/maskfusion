/*
 * This file is part of ElasticFusion.
 * 图 节点
 *
 */

#ifndef GRAPHNODE_H_
#define GRAPHNODE_H_

#include <Eigen/Dense>

class GraphNode {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // EIGEN 矩阵对其=======
  GraphNode() {}

  int id;// 身份证号码
  Eigen::Vector3f position;// 位置
  Eigen::Matrix3f rotation;// 姿态 3×3 旋转矩阵
  Eigen::Vector3f translation;// 平移向量t
  std::vector<int> neighbours;// 邻居点id序列
  bool enabled;
};

#endif /* GRAPHNODE_H_ */
