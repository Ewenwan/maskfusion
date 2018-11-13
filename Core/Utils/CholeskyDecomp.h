/*
 * This file is part of ElasticFusion.
 * Cholesky 矩阵分解，求解线性方程
 *
 */

#ifndef UTILS_CHOLESKYDECOMP_H_
#define UTILS_CHOLESKYDECOMP_H_

#include <cholmod.h>
#include <Eigen/Core>

#include "Jacobian.h"

class CholeskyDecomp {
 public:
  CholeskyDecomp();
  virtual ~CholeskyDecomp();

  void freeFactor();

  Eigen::VectorXd solve(const Jacobian& jacobian, const Eigen::VectorXd& residual, const bool firstRun);

 private:
  cholmod_common Common;
  cholmod_factor* L;
};

#endif /* UTILS_CHOLESKYDECOMP_H_ */
