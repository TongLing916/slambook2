#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace util {

class Reader {
 public:
  // 从 REMODE 数据集读取数据
  static void ReadRemode(const int width, const int height,
                         const std::string& path,
                         std::vector<std::string>* const color_image_files,
                         std::vector<Sophus::SE3d>* const poses,
                         cv::Mat* const ref_dept);
};

}  //  util