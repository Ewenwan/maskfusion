/*
 * This file is part of ElasticFusion.
 * 自定义的基本结构
 * 3×3矩阵 3维向量 相机模型 CameraModel 雅可比SE3 
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef CUDA_TYPES_CUH_
#define CUDA_TYPES_CUH_

#include <vector_types.h>

#if !defined(__CUDACC__)
#include <Eigen/Core>
#endif

//  3×3矩阵 3个 float3 实现
struct mat33
{
    mat33() {}

#if !defined(__CUDACC__)
    // 可由 Eigen::Matrix 拷贝过来
    mat33(Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & e)
    {
        memcpy(data, e.data(), sizeof(mat33));
    }
#endif

    float3 data[3];// Eigen下的 float3
};

struct DataTerm
{
    short2 zero;
    short2 one;
    float diff;
    bool valid;
};

// 相机模型 CameraModel
struct CameraModel
{
    float fx, fy, cx, cy;
    CameraModel()
     : fx(0), fy(0), cx(0), cy(0)
    {}

    CameraModel(float fx_, float fy_, float cx_, float cy_)
     : fx(fx_), fy(fy_), cx(cx_), cy(cy_)
    {}
    
    // 金字塔下采样尺度 对应的内参数
    CameraModel operator()(int level) const
    {
        int div = 1 << level;
        return (CameraModel (fx / div, fy / div, cx / div, cy / div));
    }
};

// 雅可比SE3================
struct JtJJtrSE3
{
    //27 floats for each product (27)
    float aa, ab, ac, ad, ae, af, ag,
              bb, bc, bd, be, bf, bg,
                  cc, cd, ce, cf, cg,
                      dd, de, df, dg,
                          ee, ef, eg,
                              ff, fg;

    //Extra data needed (29)
    float residual, inliers;

    __device__ inline void add(const JtJJtrSE3 & a)
    {
        aa += a.aa;
        ab += a.ab;
        ac += a.ac;
        ad += a.ad;
        ae += a.ae;
        af += a.af;
        ag += a.ag;

        bb += a.bb;
        bc += a.bc;
        bd += a.bd;
        be += a.be;
        bf += a.bf;
        bg += a.bg;

        cc += a.cc;
        cd += a.cd;
        ce += a.ce;
        cf += a.cf;
        cg += a.cg;

        dd += a.dd;
        de += a.de;
        df += a.df;
        dg += a.dg;

        ee += a.ee;
        ef += a.ef;
        eg += a.eg;

        ff += a.ff;
        fg += a.fg;

        residual += a.residual;
        inliers += a.inliers;
    }
};

// 雅可比SO3================
struct JtJJtrSO3
{
    //9 floats for each product (9)
    float aa, ab, ac, ad,
              bb, bc, bd,
                  cc, cd;

    //Extra data needed (11)
    float residual, inliers;

    __device__ inline void add(const JtJJtrSO3 & a)
    {
        aa += a.aa;
        ab += a.ab;
        ac += a.ac;
        ad += a.ad;

        bb += a.bb;
        bc += a.bc;
        bd += a.bd;

        cc += a.cc;
        cd += a.cd;

        residual += a.residual;
        inliers += a.inliers;
    }
};

#endif /* CUDA_TYPES_CUH_ */
