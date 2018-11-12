/*
* Software License Agreement (BSD License)
*  	device_array 的父类 gpu内存 * 实际 gpu内存 分配和释放的 类 
*
*  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
*/

#include "device_memory.hpp"
#include "../convenience.cuh"// 版本??

#include "cuda_runtime_api.h"// 运行时间
#include "assert.h"          // 断言宏

//////////////////////////    XADD    ///////////////////////////////
// __GNUC__             代表gcc的主版本号
// __GNUC_MINOR__       代表gcc的次版本号
// __GNUC_PATCHLEVEL__  代表gcc的修正版本号

#ifdef __GNUC__

#if __GNUC__ * 10 + __GNUC_MINOR__ >= 42 // gcc 4.2版本

#if !defined WIN32 && (defined __i486__ || defined __i586__ || defined __i686__ || defined __MMX__ || defined __SSE__ || defined __ppc__)

// 以count = 4为例，调用__sync_fetch_and_add(&count,1),之后，返回值是4，然后，count变成了5.
// https://blog.csdn.net/hzhsan/article/details/25124901
// 有 __sync_fetch_and_add , 自然也就有 __sync_add_and_fetch，呵呵这个的意思就很清楚了，先自加，再返回。
// 他们哥俩的关系与i++和++i的关系是一样的。
// 有了这个宝贝函数，我们就有新的解决办法了。对于多线程对全局变量进行自加，我们就再也不用理线程锁了。

#define CV_XADD __sync_fetch_and_add
#else
#include <ext/atomicity.h>
#define CV_XADD __gnu_cxx::__exchange_and_add
#endif
#else
#include <bits/atomicity.h>

#if __GNUC__ * 10 + __GNUC_MINOR__ >= 34// gcc 3.4
#define CV_XADD __gnu_cxx::__exchange_and_add
#else
#define CV_XADD __exchange_and_add
#endif
#endif

#elif defined WIN32 || defined _WIN32
#include <intrin.h>
// 是一个 Windows API 函数，用于对一个32位数值执行加法的原子操作====================
#define CV_XADD(addr, delta) _InterlockedExchangeAdd((long volatile*)(addr), (delta))
#else

// 加法 函数  主要用于 对引用计数 + 1
template <typename _Tp>
static inline _Tp CV_XADD(_Tp* addr, _Tp delta) {
  int tmp = *addr;
  *addr += delta;
  return tmp;
}

#endif

////////////////////////    DeviceArray  1维数组  /////////////////////////////
// 类 私有变量 初始化===============
// sizeBytes_数据字节总数量  data_数据起始地址 refcount_引用计数
DeviceMemory::DeviceMemory() : data_(0), sizeBytes_(0), refcount_(0) {}
DeviceMemory::DeviceMemory(void* ptr_arg, size_t sizeBytes_arg) : data_(ptr_arg), sizeBytes_(sizeBytes_arg), refcount_(0) {}
DeviceMemory::DeviceMemory(size_t sizeBtes_arg) : data_(0), sizeBytes_(0), refcount_(0) { create(sizeBtes_arg); }
DeviceMemory::~DeviceMemory() { release(); }

// 拷贝初始化==================
DeviceMemory::DeviceMemory(const DeviceMemory& other_arg)
    : data_(other_arg.data_), sizeBytes_(other_arg.sizeBytes_), refcount_(other_arg.refcount_) {
  if (refcount_) CV_XADD(refcount_, 1);// 引用计数+1===============
}

// 赋值初始化==================
DeviceMemory& DeviceMemory::operator=(const DeviceMemory& other_arg) 
{
  if (this != &other_arg) {// 不是同一个对象====
    if (other_arg.refcount_) CV_XADD(other_arg.refcount_, 1);// 引用计数+1
    release();

    data_ = other_arg.data_;// 数据起始地址
    sizeBytes_ = other_arg.sizeBytes_;// 数据总字节数
    refcount_ = other_arg.refcount_;  // 引用计数
  }
  return *this;
}

// cudaMalloc 申请内存空间===========================
void DeviceMemory::create(size_t sizeBytes_arg) {
  if (sizeBytes_arg == sizeBytes_) return;

  if (sizeBytes_arg > 0) {
    if (data_) release();// 先清空

    sizeBytes_ = sizeBytes_arg;

    cudaSafeCall(cudaMalloc(&data_, sizeBytes_));// cudaMalloc 申请GPU内存

    refcount_ = new int;
    *refcount_ = 1;// 引用次数 设置为1
  }
}

// cudaMemcpy深拷贝============================================
void DeviceMemory::copyTo(DeviceMemory& other) const {
  if (empty())
    other.release();
  else {
    other.create(sizeBytes_);
    cudaSafeCall(cudaMemcpy(other.data_, data_, sizeBytes_, cudaMemcpyDeviceToDevice));
    cudaSafeCall(cudaDeviceSynchronize());
  }
}

// cudaFree释放内存=====================================================
void DeviceMemory::release() {
  if (refcount_ && CV_XADD(refcount_, -1) == 1) {
    delete refcount_;
    cudaSafeCall(cudaFree(data_));
  }
  data_ = 0;
  sizeBytes_ = 0;
  refcount_ = 0;
}


// cudaMemcpy cpu 到 GPU  cudaMemcpyHostToDevice =======================
void DeviceMemory::upload(const void* host_ptr_arg, size_t sizeBytes_arg) {
  create(sizeBytes_arg);
  cudaSafeCall(cudaMemcpy(data_, host_ptr_arg, sizeBytes_, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaDeviceSynchronize());
}

// cudaMemcpy GPU 到 CPU  cudaMemcpyDeviceToHost =======================
void DeviceMemory::download(void* host_ptr_arg) const {
  cudaSafeCall(cudaMemcpy(host_ptr_arg, data_, sizeBytes_, cudaMemcpyDeviceToHost));
  cudaSafeCall(cudaDeviceSynchronize());
}

// 交换两者数据指针，数据量，引用计数========================================
void DeviceMemory::swap(DeviceMemory& other_arg) {
  std::swap(data_, other_arg.data_);
  std::swap(sizeBytes_, other_arg.sizeBytes_);
  std::swap(refcount_, other_arg.refcount_);
}

bool DeviceMemory::empty() const { return !data_; }
size_t DeviceMemory::sizeBytes() const { return sizeBytes_; }

////////////////////////    DeviceArray2D    /////////////////////////////

DeviceMemory2D::DeviceMemory2D() : data_(0), step_(0), colsBytes_(0), rows_(0), refcount_(0) {}

DeviceMemory2D::DeviceMemory2D(int rows_arg, int colsBytes_arg) : data_(0), step_(0), colsBytes_(0), rows_(0), refcount_(0) {
  create(rows_arg, colsBytes_arg);
}

DeviceMemory2D::DeviceMemory2D(int rows_arg, int colsBytes_arg, void* data_arg, size_t step_arg)
    : data_(data_arg), step_(step_arg), colsBytes_(colsBytes_arg), rows_(rows_arg), refcount_(0) {}

DeviceMemory2D::~DeviceMemory2D() { release(); }

DeviceMemory2D::DeviceMemory2D(const DeviceMemory2D& other_arg)
    : data_(other_arg.data_),
      step_(other_arg.step_),
      colsBytes_(other_arg.colsBytes_),
      rows_(other_arg.rows_),
      refcount_(other_arg.refcount_) {
  if (refcount_) CV_XADD(refcount_, 1);
}

DeviceMemory2D& DeviceMemory2D::operator=(const DeviceMemory2D& other_arg) {
  if (this != &other_arg) {
    if (other_arg.refcount_) CV_XADD(other_arg.refcount_, 1);
    release();

    colsBytes_ = other_arg.colsBytes_;
    rows_ = other_arg.rows_;
    data_ = other_arg.data_;
    step_ = other_arg.step_;

    refcount_ = other_arg.refcount_;
  }
  return *this;
}


// 二维数组 cudaMallocPitch() 和三维数组 cudaMalloc3D() 的使用 =============
// 使用函数 cudaMallocPitch()创建内存 和配套的函数 拷贝函数 cudaMemcpy2D() 来使用二维数组。
// https://www.cnblogs.com/cuancuancuanhao/p/7805892.html
// cudaMallocPitch 申请内存
// cuda 中这样分配的二维数组内存保证了数组每一行首元素的地址值都按照 256 或 512 的倍数对齐，提高访问效率，
// 但使得每行末尾元素与下一行首元素地址可能不连贯，使用指针寻址时要注意考虑尾部。

void DeviceMemory2D::create(int rows_arg, int colsBytes_arg) {
  if (colsBytes_ == colsBytes_arg && rows_ == rows_arg) return;

  if (rows_arg > 0 && colsBytes_arg > 0) {
    if (data_) release();

    colsBytes_ = colsBytes_arg;
    rows_ = rows_arg;
// cudaMAllocPitch() 传入存储器指针 **devPtr，偏移值的指针 *pitch，数组行字节数 widthByte，数组行数 height。
    cudaSafeCall(cudaMallocPitch((void**)&data_, &step_, colsBytes_, rows_));

    refcount_ = new int;
    *refcount_ = 1;
  }
}

// cudaFree 释放内存====================================
void DeviceMemory2D::release() {
  if (refcount_ && CV_XADD(refcount_, -1) == 1) {
    delete refcount_;
    cudaSafeCall(cudaFree(data_));
  }

  colsBytes_ = 0;
  rows_ = 0;
  data_ = 0;
  step_ = 0;
  refcount_ = 0;
}


// 创建cudaMallocPitch  拷贝cudaMemcpy2D 深拷贝===============================================
void DeviceMemory2D::copyTo(DeviceMemory2D& other) const {
  if (empty())
    other.release();
  else {
    other.create(rows_, colsBytes_);// 创建新内存空间
    // 拷贝
    cudaSafeCall(cudaMemcpy2D(other.data_, other.step_, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToDevice));
    cudaSafeCall(cudaDeviceSynchronize());
  }
}

void DeviceMemory2D::upload(const void* host_ptr_arg, size_t host_step_arg, int rows_arg, int colsBytes_arg) {
  create(rows_arg, colsBytes_arg);
  cudaSafeCall(cudaMemcpy2D(data_, step_, host_ptr_arg, host_step_arg, colsBytes_, rows_, cudaMemcpyHostToDevice));
}

void DeviceMemory2D::download(void* host_ptr_arg, size_t host_step_arg) const {
  cudaSafeCall(cudaMemcpy2D(host_ptr_arg, host_step_arg, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToHost));
}


// 交换
void DeviceMemory2D::swap(DeviceMemory2D& other_arg) {
  std::swap(data_, other_arg.data_);
  std::swap(step_, other_arg.step_);

  std::swap(colsBytes_, other_arg.colsBytes_);
  std::swap(rows_, other_arg.rows_);
  std::swap(refcount_, other_arg.refcount_);
}

bool DeviceMemory2D::empty() const { return !data_; }
int DeviceMemory2D::colsBytes() const { return colsBytes_; }
int DeviceMemory2D::rows() const { return rows_; }
size_t DeviceMemory2D::step() const { return step_; }
