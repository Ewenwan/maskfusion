/*
 * This file is part of ElasticFusion.
 * 图像Img类
 * 分配空间，复制初始化，获取制定位置值
 */

#ifndef UTILS_IMG_H_
#define UTILS_IMG_H_

#include <Eigen/Core>

template <class T>
class Img {
 public:
  // 初始化 图像 申请新内存，亲生的
  Img(const int rows, const int cols) : 
              rows(rows), 
              cols(cols), 
              data(new unsigned char[rows * cols * sizeof(T)]),
              owned(true) {}
  // 从其他内存 改造成图像，非亲生的
  Img(const int rows, const int cols, T* data) : 
              rows(rows), cols(cols), 
              data((unsigned char*)data), // 其他内存数据
              owned(false) {}
 
  // 析构
  virtual ~Img() {
    if (owned) {
      delete[] data; // 要是自己的，可以做主，直接消灭
    }
  }

  const int rows;// 行
  const int cols;// 列
  unsigned char* data;// 数据区域
  const bool owned;   // 是否是本单位申请的

  template <typename V>
  inline V& at(const int i) 
  {// 0<i<rows*cols ============bug
    return ((V*)data)[i];// 全局索引访问
  }

  template <typename V>
  inline V& at(const int row, const int col) // 从0开始索引
  {
    return ((V*)data)[cols * row + col];// 
  }

  template <typename V>
  inline const V& at(const int row, const int col) const {
    return ((const V*)data)[cols * row + col];
  }
};

#endif /* UTILS_IMG_H_ */
