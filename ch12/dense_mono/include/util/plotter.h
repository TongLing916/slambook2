#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace util {

class Plotter {
 public:
  // 显示极线匹配
  static void ShowEpipolarMatch(const cv::Mat& ref, const cv::Mat& curr,
                                const Eigen::Vector2d& px_ref,
                                const Eigen::Vector2d& px_curr);

  // 显示极线
  static void ShowEpipolarLine(const cv::Mat& ref, const cv::Mat& curr,
                               const Eigen::Vector2d& px_ref,
                               const Eigen::Vector2d& px_min_curr,
                               const Eigen::Vector2d& px_max_curr);

  // 显示估计的深度图
  static void PlotDepth(const cv::Mat& depth_truth,
                        const cv::Mat& depth_estimate);
};

}  // util