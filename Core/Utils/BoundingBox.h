/*
 * This file is part of https://github.com/martinruenz/maskfusion
 *  2d 检测框===============
 * 2d框融合、交集、包含关系、扩展、绘制矩形框
 */

#pragma once

// Todo make template
struct BoundingBox {
    // top----    y_min
    // bottom---  y_max
    // left-----  x_min
    // right----  x_max
    int top = std::numeric_limits<int>::max(); // order matters
    int right = std::numeric_limits<int>::min();
    int bottom = std::numeric_limits<int>::min();
    int left = std::numeric_limits<int>::max();

    inline int w() const { return right-left; }
    inline int h() const { return bottom-top; }

    static inline BoundingBox fromLeftTopWidthHeight(int l, int t, int w, int h)
    {
        return BoundingBox({t,l+w,t+h,l});// x_min，y_min，宽度，高度
    }
    
    // 两个2D框融合 变成一个大框======
    inline void merge(const BoundingBox& other) 
    {
        if (other.left < left) left = other.left;// 左上 选小的
        if (other.top < top) top = other.top;
        if (other.right > right) right = other.right;  // 右下选大的
        if (other.bottom > bottom) bottom = other.bottom;
    }
    inline void mergeLeftTopWidthHeight(int l, int t, int w, int h){
        merge(fromLeftTopWidthHeight(l,t,w,h));
      // merge(BoundingBox({t,l+w,t+h,l}));
    }
    
    // 2D框 包含关系======== 包含 other 框=======================
    inline bool includes(const BoundingBox& other) const {
        return (other.left > left && other.right < right && other.top > top && other.bottom < bottom);
    }
    inline bool includesLeftTopWidthHeight(int l, int t, int w, int h) const {
        //return includes(BoundingBox({t,l+w,t+h,l}));
        return includes(fromLeftTopWidthHeight(l,t,w,h));
    }
    
    // 包含点=========================
    inline void include(int y, int x) 
    {
        top = std::min(y, top);
        right = std::max(x, right);
        bottom = std::max(y, bottom);
        left = std::min(x, left);
    }
    
    // 2d边框 扩展 border_size ，加边框，装裱，加相框，拉警戒线横幅
    inline BoundingBox extended(int border_size) const 
    {
        return BoundingBox({top-border_size, right+border_size, bottom+border_size, left-border_size});
    }
    
    // 交集框
    inline BoundingBox intersection(const BoundingBox& other) const 
    {
        return BoundingBox({std::max(top,other.top), 
                            std::min(right,other.right), 
                            std::min(bottom,other.bottom), 
                            std::max(left,other.left)});
    }
    // 检验交集框是否合格
    inline bool intersects(const BoundingBox& other) const {
        return intersection(other).isPositive();
    }
    inline bool isPositive() const 
    {// 2d框是否 合格
        return top <= bottom && left <= right;
    }

    // OpenCV interface 转化成 opencv格式===========
    inline cv::Rect toCvRect() const {
        return cv::Rect(left,top,w(),h());
    }
    // opencv绘制 矩形框
    inline void draw(cv::Mat img, cv::Scalar color=cv::Scalar(255,0,0), int thickness=1) const {
        cv::rectangle(img, toCvRect(), color, thickness);
    }

};
