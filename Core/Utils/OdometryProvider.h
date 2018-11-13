/*
 * This file is part of ElasticFusion.
 * 里程记
 * 罗德里格斯(Rodrigues)旋转向量与矩阵的变换、
 */

#ifndef ODOMETRYPROVIDER_H_
#define ODOMETRYPROVIDER_H_

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <float.h>

class OdometryProvider {
 public:
  OdometryProvider() {}

  virtual ~OdometryProvider() {}
 
// 罗德里格斯(Rodrigues)旋转向量与矩阵的变换============
  static inline 
   Eigen::Matrix<double, 3, 3, Eigen::RowMajor> 
     rodrigues(const Eigen::Vector3d& src) 
  {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> dst = 
          Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity(); // 3×3 单位矩阵

    double rx, ry, rz, theta;

    rx = src(0);// 旋转向量
    ry = src(1);
    rz = src(2);

    theta = src.norm();

    if (theta >= DBL_EPSILON) {
      const double I[] = {1, 0, 0, 
                          0, 1, 0, 
                          0, 0, 1};

      double c = cos(theta);
      double s = sin(theta);
      double c1 = 1. - c;
      double itheta = theta ? 1. / theta : 0.;

      rx *= itheta;
      ry *= itheta;
      rz *= itheta;

      double rrt[] = {rx * rx, rx * ry, rx * rz, 
                      rx * ry, ry * ry, ry * rz, 
                      rx * rz, ry * rz, rz * rz};
      double _r_x_[] = {0, -rz, ry, 
                        rz, 0, -rx, 
                        -ry, rx, 0};
      double R[9];
      // 转旋转矩阵==========
      for (int k = 0; k < 9; k++) {
        R[k] = c * I[k] + c1 * rrt[k] + s * _r_x_[k];
      }

      memcpy(dst.data(), &R[0], sizeof(Eigen::Matrix<double, 3, 3, Eigen::RowMajor>));
    }

    return dst;
  }

 // 更新 变换矩阵
  static inline void computeUpdateSE3(Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& resultRt,
                                      const Eigen::Matrix<double, 6, 1>& result,//6自由度位姿
                                      Eigen::Isometry3f& rgbOdom) 
  {
    // for infinitesimal transformation
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = 
           Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

    Eigen::Vector3d rvec(result(3), result(4), result(5));// 3维旋转向量

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = rodrigues(rvec);// 旋转矩阵

    Rt.topLeftCorner(3, 3) = R;
    Rt(0, 3) = result(0);// 平移向量
    Rt(1, 3) = result(1);
    Rt(2, 3) = result(2);

    resultRt = Rt * resultRt;// 更新 变换矩阵

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotation = resultRt.topLeftCorner(3, 3);// 旋转矩阵
    rgbOdom.setIdentity();
    rgbOdom.rotate(rotation.cast<float>().eval());
    rgbOdom.translation() = resultRt.cast<float>().eval().topRightCorner(3, 1);
  }
};

#endif /* ODOMETRYPROVIDER_H_ */
