#include "include/util/plotter.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace util {

void Plotter::ShowEpipolarMatch(const cv::Mat &ref, const cv::Mat &curr,
                                const Eigen::Vector2d &px_ref,
                                const Eigen::Vector2d &px_curr) {
  cv::Mat ref_show, curr_show;
  cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
  cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

  cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5,
             cv::Scalar(0, 0, 250), 2);
  cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5,
             cv::Scalar(0, 0, 250), 2);

  cv::imshow("ref", ref_show);
  cv::imshow("curr", curr_show);
  cv::waitKey(1);
}

void Plotter::ShowEpipolarLine(const cv::Mat &ref, const cv::Mat &curr,
                               const Eigen::Vector2d &px_ref,
                               const Eigen::Vector2d &px_min_curr,
                               const Eigen::Vector2d &px_max_curr) {
  cv::Mat ref_show, curr_show;
  cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
  cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

  cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5,
             cv::Scalar(0, 255, 0), 2);
  cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5,
             cv::Scalar(0, 255, 0), 2);
  cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5,
             cv::Scalar(0, 255, 0), 2);
  cv::line(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)),
           cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
           cv::Scalar(0, 255, 0), 1);

  cv::imshow("ref", ref_show);
  cv::imshow("curr", curr_show);
  cv::waitKey(1);
}

void Plotter::PlotDepth(const cv::Mat &depth_truth,
                        const cv::Mat &depth_estimate) {
  cv::imshow("depth_truth", depth_truth * 0.4);
  cv::imshow("depth_estimate", depth_estimate * 0.4);
  cv::imshow("depth_error", depth_truth - depth_estimate);
  cv::waitKey(1);
}

}  // util