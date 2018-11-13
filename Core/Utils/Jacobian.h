/*
 * This file is part of ElasticFusion.
 * 二维数组
 */

#ifndef UTILS_JACOBIAN_H_
#define UTILS_JACOBIAN_H_

#include <vector>

#include "OrderedJacobianRow.h"

class Jacobian {
 public:
  Jacobian() : columns(0) {}

  virtual ~Jacobian() { reset(); }

  void assign(std::vector<OrderedJacobianRow*>& rows, const int columns) {
    reset();
    this->rows = rows;
    this->columns = columns;
  }

  int cols() const { return columns; }

  int nonZero() const {
    int count = 0;
    for (size_t i = 0; i < rows.size(); i++) {
      count += rows[i]->nonZeros();
    }
    return count;
  }

  std::vector<OrderedJacobianRow*> rows;

 private:
  int columns;

  void reset() {
    for (size_t i = 0; i < rows.size(); i++) {
      delete rows[i];
    }
    rows.clear();
  }
};

#endif /* UTILS_JACOBIAN_H_ */
