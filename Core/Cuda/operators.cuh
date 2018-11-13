/*
 * This file is part of ElasticFusion.
 * 自定义类float3、mat33的基本运算操作
 * 加 减 叉积 点积 模长 归一化
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include <vector_functions.h>

#ifndef CUDA_OPERATORS_CUH_
#define CUDA_OPERATORS_CUH_

// float3 相减=====
__device__ __host__ __forceinline__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
// float3 相加=====
__device__ __host__ __forceinline__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
// float3 叉乘=====
// a.x a.y a.z 
// b.x b.y b.z
// =======> 上中开始 a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x
__device__ __host__ __forceinline__ float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// 向量点积，对应元素乘积之和==========
__device__ __host__ __forceinline__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 向量模长=========================
__device__ __host__ __forceinline__ float norm(const float3& a)
{
    return sqrtf(dot(a, a));
}

// 向量归一化，除以模长===============
__device__ __host__ __forceinline__ float3 normalized(const float3& a)
{
    const float rn = rsqrtf(dot(a, a));// 快速计算模长
    return make_float3(a.x * rn, a.y * rn, a.z * rn);
}

// 3×3矩阵和 向量相乘===============
__device__ __forceinline__ float3 operator*(const mat33& m, const float3& a)
{
  return make_float3(dot(m.data[0], a), dot(m.data[1], a), dot(m.data[2], a));
}

#endif /* CUDA_OPERATORS_CUH_ */
