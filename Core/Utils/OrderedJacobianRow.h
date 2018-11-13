/*
 * This file is part of ElasticFusion.
 *
 *
 */

#ifndef UTILS_ORDEREDJACOBIANROW_H_
#define UTILS_ORDEREDJACOBIANROW_H_

#include <cassert>
#include <unordered_map>

class OrderedJacobianRow {
 public:
  OrderedJacobianRow(const int nonZeros)
      : indices(new int[nonZeros]), 
        vals(new double[nonZeros]), 
        lastSlot(0), lastIndex(-1), maxNonZero(nonZeros) {}

  virtual ~OrderedJacobianRow() {
    delete[] indices;
    delete[] vals;
  }

  // You have to use this in an ordered fashion for efficiency :)
  void append(const int index, const double value) {
    assert(index > lastIndex);
    indexSlotMap[index] = lastSlot;
    indices[lastSlot] = index;
    vals[lastSlot] = value;
    lastSlot++;
    lastIndex = index;
  }

  // To add to an existing and already weighted value
  void addTo(const int index, const double value, const double weight) {
    double& val = vals[indexSlotMap[index]];
    val = ((val / weight) + value) * weight;
  }

  int nonZeros() { return lastSlot; }

  int* indices;
  double* vals;

 private:
  int lastSlot;
  int lastIndex;
  const int maxNonZero;
  std::unordered_map<int, int> indexSlotMap;
};

#endif /* UTILS_ORDEREDJACOBIANROW_H_ */
