/*
 * Software License Agreement (BSD License)
 * 
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef KERNEL_CONTAINERS_HPP_
#define KERNEL_CONTAINERS_HPP_

#include <cstddef>

#if defined(__CUDACC__) // Defined when compiling CUDA source files. 
#define GPU_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
#define GPU_HOST_DEVICE__
#endif

// 设备 结构体数据
template <typename T>
struct DevPtr {
  typedef T elem_type;//数据 类型
  const static size_t elem_size = sizeof(elem_type);//数据单位长度 字节

  T* data;// 数据起始地址
// cpu数据??  主GPU数据？？
  GPU_HOST_DEVICE__ DevPtr() : data(0) {}
  GPU_HOST_DEVICE__ DevPtr(T* data_arg) : data(data_arg) {}

  GPU_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
  GPU_HOST_DEVICE__ operator T*() { return data; }
  GPU_HOST_DEVICE__ operator const T*() const { return data; }
};

template <typename T>
struct PtrSz : public DevPtr<T> {
  GPU_HOST_DEVICE__ PtrSz() : size(0) {}
  GPU_HOST_DEVICE__ PtrSz(T* data_arg, size_t size_arg) : DevPtr<T>(data_arg), size(size_arg) {}

  size_t size;
};

template <typename T>
struct PtrStep : public DevPtr<T> {
  GPU_HOST_DEVICE__ PtrStep() : step(0) {}
  GPU_HOST_DEVICE__ PtrStep(T* data_arg, size_t step_arg) : DevPtr<T>(data_arg), step(step_arg) {}

  /** \brief stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!! */
  size_t step;

  GPU_HOST_DEVICE__ T* ptr(int y = 0) { return (T*)((char*)DevPtr<T>::data + y * step); }
  GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)((const char*)DevPtr<T>::data + y * step); }
};

template <typename T>
struct PtrStepSz : public PtrStep<T> {
  GPU_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
  GPU_HOST_DEVICE__ PtrStepSz(int rows_arg, int cols_arg, T* data_arg, size_t step_arg)
      : PtrStep<T>(data_arg, step_arg), cols(cols_arg), rows(rows_arg) {}

  int cols;
  int rows;
};

#endif /* KERNEL_CONTAINERS_HPP_ */
