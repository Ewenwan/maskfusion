/*
 * This file is part of ElasticFusion.
 * 相机内参数 fx fy cx cy   Intrinsics静态类型
 *
 */

#ifndef INTRINSICS_H_
#define INTRINSICS_H_

#include <cassert>

class Intrinsics {
 public:
  static const Intrinsics& getInstance() { return getInstancePrivate(); }
  static void setIntrinics(float fx = 0, float fy = 0, float cx = 0, float cy = 0) {
    getInstancePrivate().fx_ = fx;
    getInstancePrivate().fy_ = fy;
    getInstancePrivate().cx_ = cx;
    getInstancePrivate().cy_ = cy;
    getInstancePrivate().checkSet();
  }

  const float& fx() const {
    checkSet();
    return fx_;
  }

  const float& fy() const {
    checkSet();
    return fy_;
  }

  const float& cx() const {
    checkSet();
    return cx_;
  }

  const float& cy() const {
    checkSet();
    return cy_;
  }

 private:
  Intrinsics() {}
  static Intrinsics& getInstancePrivate() {
    static Intrinsics instance;
    return instance;
  }
  void checkSet() const { assert(fx_ != 0 && fy_ != 0 && "You haven't initialised the Intrinsics class!"); }

  float fx_, fy_, cx_, cy_;
};

#endif /* INTRINSICS_H_ */
