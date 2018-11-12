/*
 * Software License Agreement (BSD License)
 * device_array 的父类 gpu内存
 * 实际 gpu内存 分配和释放的 类
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef DEVICE_MEMORY_HPP_
#define DEVICE_MEMORY_HPP_

#include "kernel_containers.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b DeviceMemory class
  *
  * \note This is a BLOB container class with reference counting for GPU memory.
  *
  * \author Anatoly Baksheev
  */

class DeviceMemory {
 public:
  /** \brief Empty constructor. */
  DeviceMemory();// 空的构造函数

  /** \brief Destructor. */
  ~DeviceMemory();// 析构函数  内存清理

  /** \brief Allocates internal buffer in GPU memory
    * \param sizeBytes_arg: amount of memory to allocate
    * */
  DeviceMemory(size_t sizeBytes_arg);// 传入 总字节数量

  /** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
    * \param ptr_arg: pointer to buffer
    * \param sizeBytes_arg: buffer size
    * */
  DeviceMemory(void* ptr_arg, size_t sizeBytes_arg);// 从指定地址，对齐的 sizeBytes_arg 个字节 的GPU内存地址

  /** \brief Copy constructor. Just increments reference counter. */
  DeviceMemory(const DeviceMemory& other_arg);// 浅拷贝，拷贝初始化，仅增加引用计数

  /** \brief Assigment operator. Just increments reference counter. */
  DeviceMemory& operator=(const DeviceMemory& other_arg);// 浅拷贝，赋值初始化，仅增加引用计数

  /** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with
   * new size. If new and old sizes are equal it does nothing.
    * \param sizeBytes_arg: buffer size
    * */
  void create(size_t sizeBytes_arg);// 申请GPU内存单元，出入总字节数量

  /** \brief Decrements reference counter and releases internal buffer if needed. */
  void release();// 释放GPU内存

  /** \brief Performs data copying. If destination size differs it will be reallocated.
    * \param other_arg: destination container
    * */
  void copyTo(DeviceMemory& other) const;// 深拷贝，完全复制另一个对象，克隆

  /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is
   * enough.
    * \param host_ptr_arg: pointer to buffer to upload
    * \param sizeBytes_arg: buffer size
    * */
  void upload(const void* host_ptr_arg, size_t sizeBytes_arg);// CPU 到 GPU 

  /** \brief Downloads data from internal buffer to CPU memory
    * \param host_ptr_arg: pointer to buffer to download
    * */
  void download(void* host_ptr_arg) const;// gpu 到 CPU

  /** \brief Performs swap of data pointed with another device memory.
    * \param other: device memory to swap with
    * */
  void swap(DeviceMemory& other_arg);// 交换 GPU内存中 两个对象的数据

  /** \brief Returns pointer for internal buffer in GPU memory. */
  template <class T>
  T* ptr();// 返回GPU数据 的地址(指针)

  /** \brief Returns constant pointer for internal buffer in GPU memory. */
  template <class T>
  const T* ptr() const;// 返回常量指针

  /** \brief Conversion to PtrSz for passing to kernel functions. */
  template <class U>
  operator PtrSz<U>() const;// 传递给核函数

  /** \brief Returns true if unallocated otherwise false. */
  bool empty() const;// 内存未分配

  size_t sizeBytes() const;// 返回 使用的总字节数量

 private:
  /** \brief Device pointer. */
  void* data_;// 数据起始指针

  /** \brief Allocated size in bytes. */
  size_t sizeBytes_;// 数据字节总数量

  /** \brief Pointer to reference counter in CPU memory. */
  int* refcount_;// 数据内存区域 引用计数器
};


// gpu内存 2维数组 实现
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b DeviceMemory2D class
  *
  * \note This is a BLOB container class with reference counting for pitched GPU memory.
  *
  * \author Anatoly Baksheev
  */

class DeviceMemory2D {
 public:
  /** \brief Empty constructor. */
  DeviceMemory2D();

  /** \brief Destructor. */
  ~DeviceMemory2D();

  /** \brief Allocates internal buffer in GPU memory
    * \param rows_arg: number of rows to allocate
    * \param colsBytes_arg: width of the buffer in bytes
    * */
  DeviceMemory2D(int rows_arg, int colsBytes_arg);

  /** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
    * \param rows_arg: number of rows
    * \param colsBytes_arg: width of the buffer in bytes
    * \param data_arg: pointer to buffer
    * \param stepBytes_arg: stride between two consecutive rows in bytes
    * */
  DeviceMemory2D(int rows_arg, int colsBytes_arg, void* data_arg, size_t step_arg);

  /** \brief Copy constructor. Just increments reference counter. */
  DeviceMemory2D(const DeviceMemory2D& other_arg);

  /** \brief Assigment operator. Just increments reference counter. */
  DeviceMemory2D& operator=(const DeviceMemory2D& other_arg);

  /** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with
   * new size. If new and old sizes are equal it does nothing.
     * \param ptr_arg: number of rows to allocate
     * \param sizeBytes_arg: width of the buffer in bytes
     * */
  void create(int rows_arg, int colsBytes_arg);

  /** \brief Decrements reference counter and releases internal buffer if needed. */
  void release();

  /** \brief Performs data copying. If destination size differs it will be reallocated.
    * \param other_arg: destination container
    * */
  void copyTo(DeviceMemory2D& other) const;

  /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is
   * enough.
    * \param host_ptr_arg: pointer to host buffer to upload
    * \param host_step_arg: stride between two consecutive rows in bytes for host buffer
    * \param rows_arg: number of rows to upload
    * \param sizeBytes_arg: width of host buffer in bytes
    * */
  void upload(const void* host_ptr_arg, size_t host_step_arg, int rows_arg, int colsBytes_arg);

  /** \brief Downloads data from internal buffer to CPU memory. User is resposible for correct host buffer size.
    * \param host_ptr_arg: pointer to host buffer to download
    * \param host_step_arg: stride between two consecutive rows in bytes for host buffer
    * */
  void download(void* host_ptr_arg, size_t host_step_arg) const;

  /** \brief Performs swap of data pointed with another device memory.
    * \param other: device memory to swap with
    * */
  void swap(DeviceMemory2D& other_arg);

  /** \brief Returns pointer to given row in internal buffer.
    * \param y_arg: row index
    * */
  template <class T>
  T* ptr(int y_arg = 0);

  /** \brief Returns constant pointer to given row in internal buffer.
    * \param y_arg: row index
    * */
  template <class T>
  const T* ptr(int y_arg = 0) const;

  /** \brief Conversion to PtrStep for passing to kernel functions. */
  template <class U>
  operator PtrStep<U>() const;

  /** \brief Conversion to PtrStepSz for passing to kernel functions. */
  template <class U>
  operator PtrStepSz<U>() const;

  /** \brief Returns true if unallocated otherwise false. */
  bool empty() const;

  /** \brief Returns number of bytes in each row. */
  int colsBytes() const;

  /** \brief Returns number of rows. */
  int rows() const;

  /** \brief Returns stride between two consecutive rows in bytes for internal buffer. Step is stored always and everywhere
   * in bytes!!! */
  size_t step() const;

 private:
  /** \brief Device pointer. */
  void* data_;

  /** \brief Stride between two consecutive rows in bytes for internal buffer. Step is stored always and everywhere in
   * bytes!!! */
  size_t step_;

  /** \brief Width of the buffer in bytes. */
  int colsBytes_;

  /** \brief Number of rows. */
  int rows_;

  /** \brief Pointer to reference counter in CPU memory. */
  int* refcount_;
};

#include "device_memory_impl.hpp"

#endif /* DEVICE_MEMORY_HPP_ */
