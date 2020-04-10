#include "include/util/reader.h"

#include <fstream>

#include <glog/logging.h>

namespace util {

void Reader::ReadRemode(const int width, const int height,
                        const std::string& path,
                        std::vector<std::string>* const color_image_files,
                        std::vector<Sophus::SE3d>* const poses,
                        cv::Mat* const ref_depth) {
  CHECK_NOTNULL(color_image_files);
  CHECK_NOTNULL(poses);
  CHECK_NOTNULL(ref_depth);

  const std::string path_pose =
      path + "/first_200_frames_traj_over_table_input_sequence.txt";
  std::ifstream fin(path_pose);
  LOG_IF(FATAL, !fin) << "WRONG PATH " << path_pose;

  std::string image;
  std::vector<double> data(7, 0.);
  while (!fin.eof()) {
    // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 Twc 而非 Tcw

    fin >> image;
    for (double& d : data) {
      fin >> d;
    };

    color_image_files->emplace_back(path + "/images/" + image);
    poses->emplace_back(
        Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                     Eigen::Vector3d(data[0], data[1], data[2])));

    if (!fin.good()) {
      break;
    }
  }
  fin.close();

  // load reference depth
  const std::string path_depth = path + "/depthmaps/scene_000.depth";
  fin.open(path_depth);
  *ref_depth = cv::Mat(height, width, CV_64F);
  LOG_IF(FATAL, !fin) << "WRONG PATH " << path_depth;
  double depth = 0.;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      fin >> depth;
      ref_depth->ptr<double>(y)[x] = depth / 100.;
    }
  }
}

}  // util