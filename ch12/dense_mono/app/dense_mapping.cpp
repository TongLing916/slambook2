#include <fstream>
#include <iostream>
#include <vector>

#include <boost/timer.hpp>
#include <opencv2/opencv.hpp>

#include "include/depth_predictor.h"
#include "include/util/image_processor.h"
#include "include/util/plotter.h"
#include "include/util/reader.h"

using namespace util;
using namespace dense_mapper;

/**********************************************
* 本程序演示了单目相机在已知轨迹下的稠密深度估计
* 使用极线搜索 + NCC 匹配的方式，与书本的 12.2 节对应
* 请注意本程序并不完美，你完全可以改进它——我其实在故意暴露一些问题(这是借口)。
***********************************************/

// ------------------------------------------------------------------
// parameters
const int border = 20;     // 边缘宽度
const int width = 640;     // 图像宽度
const int height = 480;    // 图像高度
const double fx = 481.2;   // 相机内参
const double fy = -480.0;  // WHY NEGATIVE?
const double cx = 319.5;
const double cy = 239.5;
const int ncc_window_size = 3;  // NCC 取的窗口半宽度
const double min_cov = 0.1;     // 收敛判定：最小方差
const double max_cov = 10.;     // 发散判定：最大方差

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: dense_mapping path_to_test_dataset" << std::endl;
    return -1;
  }

  // 从数据集读取数据
  std::vector<std::string> color_image_files;
  std::vector<Sophus::SE3d> poses_Twc;
  cv::Mat ref_depth;
  Reader::ReadRemode(width, height, argv[1], &color_image_files, &poses_Twc,
                     &ref_depth);
  std::cout << "read total " << color_image_files.size() << " files."
            << std::endl;

  DepthPredictor predictor(border, width, height, fx, fy, cx, cy,
                           ncc_window_size, min_cov, max_cov);

  // 第一张图
  const cv::Mat ref = cv::imread(color_image_files[0], 0);  // gray-scale image
  const Sophus::SE3d pose_ref_Twc = poses_Twc[0];           // reference pose
  const double init_depth = 3.0;                            // 深度初始值
  const double init_cov2 = 3.0;                             // 方差初始值
  cv::Mat depth(height, width, CV_64F, init_depth);         // 深度图
  cv::Mat depth_cov2(height, width, CV_64F, init_cov2);     // 深度图方差

  for (size_t index = 1; index < color_image_files.size(); ++index) {
    std::cout << "*** loop " << index << " ***" << std::endl;
    cv::Mat curr = cv::imread(color_image_files[index], 0);
    if (curr.data == nullptr) {
      continue;
    }
    Sophus::SE3d pose_curr_Twc = poses_Twc[index];

    // 坐标转换关系： T_cw * T_wr = T_cr
    Sophus::SE3d pose_T_cr = pose_curr_Twc.inverse() * pose_ref_Twc;
    predictor.Update(ref, curr, pose_T_cr, &depth, &depth_cov2);
    predictor.EvaludateDepth(ref_depth, depth);
    Plotter::PlotDepth(ref_depth, depth);
    cv::imshow("image", curr);
    cv::waitKey(1);
  }

  std::cout << "estimation returns, saving depth map ..." << std::endl;
  cv::imwrite("depth.png", depth);
  std::cout << "done." << std::endl;

  return 0;
}
