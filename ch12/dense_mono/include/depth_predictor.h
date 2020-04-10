#pragma once

#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace dense_mapper {

class DepthPredictor {
 public:
  DepthPredictor(const int border, const int width, const int height,
                 const double fx, const double fy, const double cx,
                 const double cy, const int ncc_window_size,
                 const double min_cov, const double max_cov)
      : border_(border),
        width_(width),
        height_(height),
        fx_(fx),
        fy_(fy),
        cx_(cx),
        cy_(cy),
        ncc_window_size_(ncc_window_size),
        ncc_area_((2 * ncc_window_size + 1) * (2 * ncc_window_size + 1)),
        min_cov_(min_cov),
        max_cov_(max_cov) {
    CHECK_GE(border, 0);
    CHECK_GT(width, 0);
    CHECK_GT(height, 0);
    CHECK_GT(fx, 0);
    // CHECK_GT(fy, 0);
    CHECK_GT(cx, 0);
    CHECK_GT(cy, 0);
    CHECK_GT(ncc_window_size, 0);
    CHECK_GT(min_cov, 0);
    CHECK_GT(max_cov, 0);
  }

  /**
   * 根据新的图像更新深度估计
   * @param ref           参考图像
   * @param curr          当前图像
   * @param Tcr         参考图像到当前图像的位姿
   * @param depth         深度
   * @param depth_cov     深度方差
   */
  void Update(const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &Tcr,
              cv::Mat *const depth, cv::Mat *const depth_cov2);

  // 评测深度估计
  void EvaludateDepth(const cv::Mat &depth_truth,
                      const cv::Mat &depth_estimate);

 private:
  /**
   * 极线搜索
   * @param ref           参考图像
   * @param curr          当前图像
   * @param Tcr         位姿
   * @param pt_ref        参考图像中点的位置
   * @param depth_mu      深度均值
   * @param depth_cov     深度方差
   * @param pt_curr       当前点
   * @param epipolar_direction  极线方向
   * @return              是否成功
   */
  bool EpipolarSearch(const cv::Mat &ref, const cv::Mat &curr,
                      const Sophus::SE3d &Tcr, const Eigen::Vector2d &pt_ref,
                      const double depth_mu, const double depth_cov,
                      Eigen::Vector2d *const pt_curr,
                      Eigen::Vector2d *const epipolar_direction);

  /**
   * 更新深度滤波器
   * @param pt_ref    参考图像点
   * @param pt_curr   当前图像点
   * @param Tcr     位姿
   * @param epipolar_direction 极线方向
   * @param depth     深度均值
   * @param depth_cov2    深度方向
   * @return          是否成功
   */
  bool UpdateDepthFilter(const Eigen::Vector2d &pt_ref,
                         const Eigen::Vector2d &pt_curr,
                         const Sophus::SE3d &Tcr,
                         const Eigen::Vector2d &epipolar_direction,
                         cv::Mat *const depth, cv::Mat *const depth_cov2);

  /**
   * 计算 NCC 评分
   * @param ref       参考图像
   * @param curr      当前图像
   * @param pt_ref    参考点
   * @param pt_curr   当前点
   * @return          NCC评分
   */
  double NCC(const cv::Mat &ref, const cv::Mat &curr,
             const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr);

  // 像素到相机坐标系
  Eigen::Vector3d Pixel2Cam(const Eigen::Vector2d px) {
    return Eigen::Vector3d((px(0, 0) - cx_) / fx_, (px(1, 0) - cy_) / fy_, 1.);
  }

  // 相机坐标系到像素
  Eigen::Vector2d Cam2Pixel(const Eigen::Vector3d p_cam) {
    return Eigen::Vector2d(p_cam(0, 0) * fx_ / p_cam(2, 0) + cx_,
                           p_cam(1, 0) * fy_ / p_cam(2, 0) + cy_);
  }

  // 检测一个点是否在图像边框内
  bool IsInside(const Eigen::Vector2d &pt) {
    return pt(0, 0) >= border_ && pt(1, 0) >= border_ &&
           pt(0, 0) + border_ < width_ && pt(1, 0) + border_ <= height_;
  }

 private:
  const int border_;  // 边缘宽度
  const int width_;   // 图像宽度
  const int height_;  // 图像高度
  const double fx_;   // 相机内参
  const double fy_;
  const double cx_;
  const double cy_;
  const int ncc_window_size_;  // NCC 取的窗口半宽度
  const int ncc_area_;         // NCC窗口面积
  const double min_cov_;       // 收敛判定：最小方差
  const double max_cov_;       // 发散判定：最大方差
};

}  // dense_mapper