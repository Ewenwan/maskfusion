/*
 * Software License Agreement (BSD License)
 * 对应 device_array.hpp 头文件中声明函数的实现    implementation
 * cUDA 设备内存数据数组 * 1维数组 * 2维数组
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef DEVICE_ARRAY_IMPL_HPP_
#define DEVICE_ARRAY_IMPL_HPP_

// 内联函数  GPU设备内存 1维数组
/////////////////////  Inline implementations of DeviceArray ////////////////////////////////////////////

template <class T>
inline DeviceArray<T>::DeviceArray() {}// 空构造函数
template <class T>
inline DeviceArray<T>::DeviceArray(size_t size) : DeviceMemory(size * elem_size) {}// 元素数量×单元素字节数量 cuda内存
template <class T>
inline DeviceArray<T>::DeviceArray(T* ptr, size_t size) : DeviceMemory(ptr, size * elem_size) {}// 指定地址处的内存块
template <class T>
inline DeviceArray<T>::DeviceArray(const DeviceArray& other) : DeviceMemory(other) {}// 浅拷贝，拷贝初始化
template <class T>
inline DeviceArray<T>& DeviceArray<T>::operator=(const DeviceArray& other) {// 浅拷贝，赋值初始化
  DeviceMemory::operator=(other);
  return *this;
}

template <class T>
inline void DeviceArray<T>::create(size_t size) {// 申请内存空间
  DeviceMemory::create(size * elem_size);
}
template <class T>
inline void DeviceArray<T>::release() {// 释放内存空间
  DeviceMemory::release();
}

template <class T>
inline void DeviceArray<T>::copyTo(DeviceArray& other) const {// 深拷贝，完全复制一个对象
  DeviceMemory::copyTo(other);
}
template <class T>
inline void DeviceArray<T>::upload(const T* host_ptr, size_t size) {// cpu内存数据 到 GPU内存数据
  DeviceMemory::upload(host_ptr, size * elem_size);
}
template <class T>
inline void DeviceArray<T>::download(T* host_ptr) const {//  GPU内存数据 到 cpu内存数据
  DeviceMemory::download(host_ptr);
}

template <class T>
void DeviceArray<T>::swap(DeviceArray& other_arg) {// GPU内存数据 互相交换
  DeviceMemory::swap(other_arg);
}

template <class T>
inline DeviceArray<T>::operator T*() {// 返回gpu数据地址指针
  return ptr();
}
template <class T>
inline DeviceArray<T>::operator const T*() const {// 返回gpu数据地址常量指针
  return ptr();
}
template <class T>
inline size_t DeviceArray<T>::size() const {// 返回元素数量
  return sizeBytes() / elem_size;
}

template <class T>
inline T* DeviceArray<T>::ptr() {// DeviceMemory::ptr 方式实现
  return DeviceMemory::ptr<T>();
}
template <class T>
inline const T* DeviceArray<T>::ptr() const {
  return DeviceMemory::ptr<T>();
}

template <class T>
template <class A>
inline void DeviceArray<T>::upload(const std::vector<T, A>& data) {// cpu内存中的数组数据 上传到 GPU
  upload(&data[0], data.size());
}
template <class T>
template <class A>
inline void DeviceArray<T>::download(std::vector<T, A>& data) const {// GPU内存中的数组数据 上传到 CPU
  data.resize(size());
  if (!data.empty()) download(&data[0]);
}


// 内联函数  GPU设备内存 1维数组
/////////////////////  Inline implementations of DeviceArray2D ////////////////////////////////////////////

// 数据对齐=============
template <class T>
inline DeviceArray2D<T>::DeviceArray2D() {}
template <class T>
inline DeviceArray2D<T>::DeviceArray2D(int rows, int cols) : DeviceMemory2D(rows, cols * elem_size) {}// 行，列数×单元素字节量
template <class T>
inline DeviceArray2D<T>::DeviceArray2D(int rows, int cols, void* data, size_t stepBytes)//指定地址
    : DeviceMemory2D(rows, cols * elem_size, data, stepBytes) {}
template <class T>
inline DeviceArray2D<T>::DeviceArray2D(const DeviceArray2D& other) : DeviceMemory2D(other) {}
template <class T>
inline DeviceArray2D<T>& DeviceArray2D<T>::operator=(const DeviceArray2D& other) {
  DeviceMemory2D::operator=(other);
  return *this;
}

template <class T>
inline void DeviceArray2D<T>::create(int rows, int cols) {// 申请内存 行 列数
  DeviceMemory2D::create(rows, cols * elem_size);
}
template <class T>
inline void DeviceArray2D<T>::release() {// 释放内存
  DeviceMemory2D::release();
}

template <class T>
inline void DeviceArray2D<T>::copyTo(DeviceArray2D& other) const {// 深拷贝
  DeviceMemory2D::copyTo(other);
}
template <class T>
inline void DeviceArray2D<T>::upload(const void* host_ptr, size_t host_step, int rows, int cols) {
  DeviceMemory2D::upload(host_ptr, host_step, rows, cols * elem_size);
}
template <class T>
inline void DeviceArray2D<T>::download(void* host_ptr, size_t host_step) const {
  DeviceMemory2D::download(host_ptr, host_step);
}

template <class T>
template <class A>
inline void DeviceArray2D<T>::upload(const std::vector<T, A>& data, int cols) {
  upload(&data[0], cols * elem_size, data.size() / cols, cols);
}

template <class T>
template <class A>
inline void DeviceArray2D<T>::download(std::vector<T, A>& data, int& elem_step) const {
  elem_step = cols();
  data.resize(cols() * rows());
  if (!data.empty()) download(&data[0], colsBytes());
}

template <class T>
void DeviceArray2D<T>::swap(DeviceArray2D& other_arg) {
  DeviceMemory2D::swap(other_arg);
}

template <class T>
inline T* DeviceArray2D<T>::ptr(int y) {
  return DeviceMemory2D::ptr<T>(y);
}
template <class T>
inline const T* DeviceArray2D<T>::ptr(int y) const {
  return DeviceMemory2D::ptr<T>(y);
}

template <class T>
inline DeviceArray2D<T>::operator T*() {
  return ptr();
}
template <class T>
inline DeviceArray2D<T>::operator const T*() const {
  return ptr();
}

template <class T>
inline int DeviceArray2D<T>::cols() const {
  return DeviceMemory2D::colsBytes() / elem_size;
}
template <class T>
inline int DeviceArray2D<T>::rows() const {
  return DeviceMemory2D::rows();
}

template <class T>
inline size_t DeviceArray2D<T>::elem_step() const {
  return DeviceMemory2D::step() / elem_size;
}

#endif /* DEVICE_ARRAY_IMPL_HPP_ */
