#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace util {

class ImageProcessor {
 public:
  // 双线性灰度插值
  static inline double GetBilinearInterpolatedValue(const cv::Mat& img,
                                                    const Eigen::Vector2d& pt) {
    uchar* d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    const double xx = pt(0, 0) - floor(pt(0, 0));
    const double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1. - xx) * (1. - yy) * double(d[0]) +
            xx * (1. - yy) * double(d[1]) +
            (1. - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) /
           255.0;
  }
};

}  // util