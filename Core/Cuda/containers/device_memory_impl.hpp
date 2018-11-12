/*
 * Software License Agreement (BSD License)
 * gpu数据 指针、数据大小 获取
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef DEVICE_MEMORY_IMPL_HPP_
#define DEVICE_MEMORY_IMPL_HPP_

/////////////////////  Inline implementations of DeviceMemory ////////////////////////////////////////////
// 1维 数据  针、数据大小获取
template <class T>
inline T* DeviceMemory::ptr() {
  return (T*)data_;//数据 指针 
}
template <class T>
inline const T* DeviceMemory::ptr() const {
  return (const T*)data_;//数据 常量指针 
}

// 获取 指针 + 元素数量
template <class U>
inline DeviceMemory::operator PtrSz<U>() const {
  PtrSz<U> result;
  result.data = (U*)ptr<U>(); // 指针 
  result.size = sizeBytes_ / sizeof(U);// 元素数量
  return result;
}

/////////////////////  Inline implementations of DeviceMemory2D ////////////////////////////////////////////
// 2维 数据  针、数据大小获取
//  cuda 中这样分配的二维数组内存保证了数组每一行首元素的地址值都按照 256 或 512 的倍数对齐，提高访问效率，
// 但使得每行末尾元素与下一行首元素地址可能不连贯，使用指针寻址时要注意考虑尾部。
template <class T>
T* DeviceMemory2D::ptr(int y_arg) {
  return (T*)((char*)data_ + y_arg * step_);// 数据首地址 考虑 CUDA内存对其齐
}
template <class T>
const T* DeviceMemory2D::ptr(int y_arg) const {
  return (const T*)((const char*)data_ + y_arg * step_);
}

template <class U>
DeviceMemory2D::operator PtrStep<U>() const {
  PtrStep<U> result;
  result.data = (U*)ptr<U>();// 地址
  result.step = step_;// 内存对齐参数
  return result;
}

template <class U>
DeviceMemory2D::operator PtrStepSz<U>() const {
  PtrStepSz<U> result;
  result.data = (U*)ptr<U>();// 地址
  result.step = step_;       // 内存对齐参数
  result.cols = colsBytes_ / sizeof(U);// 列宽度
  result.rows = rows_;                 // 行数量
  return result;
}

#endif /* DEVICE_MEMORY_IMPL_HPP_ */
