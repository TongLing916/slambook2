#include "include/depth_predictor.h"

#include <iostream>

#include "include/util/image_processor.h"
#include "include/util/plotter.h"

namespace dense_mapper {

// 对整个深度图进行更新
void DepthPredictor::Update(const cv::Mat &ref, const cv::Mat &curr,
                            const Sophus::SE3d &Tcr, cv::Mat *const depth,
                            cv::Mat *const depth_cov2) {
  CHECK_NOTNULL(depth);
  CHECK_NOTNULL(depth_cov2);

  for (int x = border_; x < width_ - border_; ++x) {
    for (int y = border_; y < height_ - border_; ++y) {
      // 遍历每个像素
      if (depth_cov2->ptr<double>(y)[x] < min_cov_ ||
          depth_cov2->ptr<double>(y)[x] > max_cov_) {
        // 深度已收敛或发散
        continue;
      }
      // 在极线上搜索 (x,y) 的匹配
      Eigen::Vector2d pt_curr;
      Eigen::Vector2d epipolar_direction;
      const bool ret = EpipolarSearch(
          ref, curr, Tcr, Eigen::Vector2d(x, y), depth->ptr<double>(y)[x],
          sqrt(depth_cov2->ptr<double>(y)[x]), &pt_curr, &epipolar_direction);

      if (ret == false) {
        // 匹配失败
        continue;
      }

      // 取消该注释以显示匹配
      // Plotter::ShowEpipolarMatch(ref, curr, Eigen::Vector2d(x, y), pt_curr);

      // 匹配成功，更新深度图
      DepthPredictor::UpdateDepthFilter(Eigen::Vector2d(x, y), pt_curr, Tcr,
                                        epipolar_direction, depth, depth_cov2);
    }
  }
}

// 极线搜索
// 方法见书 12.2 12.3 两节
bool DepthPredictor::EpipolarSearch(const cv::Mat &ref, const cv::Mat &curr,
                                    const Sophus::SE3d &Tcr,
                                    const Eigen::Vector2d &pt_ref,
                                    const double depth_mu,
                                    const double depth_cov,
                                    Eigen::Vector2d *const pt_curr,
                                    Eigen::Vector2d *const epipolar_direction) {
  CHECK_NOTNULL(pt_curr);
  CHECK_NOTNULL(epipolar_direction);

  Eigen::Vector3d f_ref = Pixel2Cam(pt_ref);
  f_ref.normalize();
  const Eigen::Vector3d P_ref = f_ref * depth_mu;  // 参考帧的 P 向量

  // 按深度均值投影的像素
  const Eigen::Vector2d px_mean_curr = Cam2Pixel(Tcr * P_ref);

  double d_min = depth_mu - 3 * depth_cov;
  double d_max = depth_mu + 3 * depth_cov;
  if (d_min < 0.1) {
    d_min = 0.1;
  }

  // 按最小深度投影的像素
  Eigen::Vector2d px_min_curr = Cam2Pixel(Tcr * (f_ref * d_min));

  // 按最大深度投影的像素
  Eigen::Vector2d px_max_curr = Cam2Pixel(Tcr * (f_ref * d_max));

  // 极线（线段形式）
  Eigen::Vector2d epipolar_line = px_max_curr - px_min_curr;
  *epipolar_direction = epipolar_line;  // 极线方向
  epipolar_direction->normalize();
  double half_length = 0.5 * epipolar_line.norm();  // 极线线段的半长度
  if (half_length > 100.) {
    // 我们不希望搜索太多东西
    half_length = 100.;
  }

  // 取消此句注释以显示极线（线段）
  // Plotter::ShowEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

  // 在极线上搜索，以深度均值点为中心，左右各取半长度
  double best_ncc = -1.0;
  Eigen::Vector2d best_px_curr;

  // l+=sqrt(2)
  for (double l = -half_length; l <= half_length; l += 0.7) {
    // 待匹配点
    Eigen::Vector2d px_curr = px_mean_curr + l * (*epipolar_direction);
    if (!IsInside(px_curr)) {
      continue;
    }
    // 计算待匹配点与参考帧的 NCC
    const double ncc = NCC(ref, curr, pt_ref, px_curr);
    if (ncc > best_ncc) {
      best_ncc = ncc;
      best_px_curr = px_curr;
    }
  }
  if (best_ncc < 0.85) {
    // 只相信 NCC 很高的匹配
    return false;
  }
  *pt_curr = best_px_curr;
  return true;
}

double DepthPredictor::NCC(const cv::Mat &ref, const cv::Mat &curr,
                           const Eigen::Vector2d &pt_ref,
                           const Eigen::Vector2d &pt_curr) {
  // 零均值-归一化互相关
  // 先算均值
  double mean_ref = 0, mean_curr = 0;
  std::vector<double> values_ref, values_curr;  // 参考帧和当前帧的均值
  for (int x = -ncc_window_size_; x <= ncc_window_size_; x++) {
    for (int y = -ncc_window_size_; y <= ncc_window_size_; y++) {
      const double value_ref =
          double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) /
          255.0;
      mean_ref += value_ref;

      double value_curr = util::ImageProcessor::GetBilinearInterpolatedValue(
          curr, pt_curr + Eigen::Vector2d(x, y));
      mean_curr += value_curr;

      values_ref.emplace_back(value_ref);
      values_curr.emplace_back(value_curr);
    }
  }

  mean_ref /= ncc_area_;
  mean_curr /= ncc_area_;

  // 计算 Zero mean NCC
  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (size_t i = 0; i < values_ref.size(); ++i) {
    const double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
    numerator += n;
    demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
    demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);  // 防止分母出现零
}

bool DepthPredictor::UpdateDepthFilter(
    const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr,
    const Sophus::SE3d &Tcr, const Eigen::Vector2d &epipolar_direction,
    cv::Mat *const depth, cv::Mat *const depth_cov2) {
  CHECK_NOTNULL(depth);
  CHECK_NOTNULL(depth_cov2);

  // 用三角化计算深度
  const Sophus::SE3d Trc = Tcr.inverse();
  Eigen::Vector3d f_ref = Pixel2Cam(pt_ref);
  f_ref.normalize();
  Eigen::Vector3d f_curr = Pixel2Cam(pt_curr);
  f_curr.normalize();

  // 方程
  // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
  // f2 = R_RC * f_cur
  // 转化成下面这个矩阵方程组
  // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
  //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
  const Eigen::Vector3d t = Trc.translation();  // Oc -> Or
  const Eigen::Vector3d f2 = Trc.so3() * f_curr;
  const Eigen::Vector2d b = Eigen::Vector2d(t.dot(f_ref), t.dot(f2));
  Eigen::Matrix2d A;
  A(0, 0) = f_ref.dot(f_ref);
  A(0, 1) = -f_ref.dot(f2);
  A(1, 0) = -A(0, 1);
  A(1, 1) = -f2.dot(f2);
  const Eigen::Vector2d ans = A.inverse() * b;
  const Eigen::Vector3d xm = ans[0] * f_ref;   // ref 侧的结果
  const Eigen::Vector3d xn = t + ans[1] * f2;  // cur 结果
  const Eigen::Vector3d p_esti = (xm + xn) / 2.0;  // P的位置，取两者的平均
  const double depth_estimation = p_esti.norm();  // 深度值

  // 计算不确定性（以一个像素为误差）
  const double t_norm = t.norm();
  const double alpha = acos(f_ref.dot(t) / t_norm);
  Eigen::Vector3d f_curr_prime = Pixel2Cam(pt_curr + epipolar_direction);
  f_curr_prime.normalize();
  const double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
  const double gamma = M_PI - alpha - beta_prime;
  const double p_prime = t_norm * sin(beta_prime) / sin(gamma);
  const double d_cov = fabs(p_prime - depth_estimation);
  const double d_cov2 = d_cov * d_cov;

  // 高斯融合
  const int v = static_cast<int>(pt_ref(1));
  const int u = static_cast<int>(pt_ref(0));
  const double mu = depth->ptr<double>(v)[u];
  const double sigma2 = depth_cov2->ptr<double>(v)[u];

  const double mu_fuse =
      (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
  const double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

  depth->ptr<double>(v)[u] = mu_fuse;
  depth_cov2->ptr<double>(v)[u] = sigma_fuse2;

  return true;
}

void DepthPredictor::EvaludateDepth(const cv::Mat &depth_truth,
                                    const cv::Mat &depth_estimate) {
  double ave_depth_error = 0.;     // 平均误差
  double ave_depth_error_sq = 0.;  // 平方误差
  int cnt_depth_data = 0;
  for (int y = border_; y < depth_truth.rows - border_; ++y) {
    for (int x = border_; x < depth_truth.cols - border_; ++x) {
      const double error =
          depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
      ave_depth_error += error;
      ave_depth_error_sq += error * error;
      ++cnt_depth_data;
    }
  }
  ave_depth_error /= cnt_depth_data;
  ave_depth_error_sq /= cnt_depth_data;

  std::cout << "Average squared error = " << ave_depth_error_sq
            << ", average error: " << ave_depth_error << std::endl;
}

}  // dense_mapper