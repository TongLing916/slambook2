#include <fstream>
#include <iostream>
#include <string>

#include <glog/logging.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using std::cerr;
using std::endl;
using std::cout;
using std::vector;
using std::ifstream;
using std::string;
using PointT = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointT>;

int main(int argc, char **argv) {
  LOG_IF(FATAL, argc < 3)
      << "Usage: ./pointcloud_mapping path_2_pose_file path_2_data";

  ifstream fin(argv[1]);
  LOG_IF(FATAL, !fin) << "Cannot find pose file " << argv[1];

  vector<cv::Mat> imgs_color, imgs_depth;  // color images and depth images
  vector<Eigen::Isometry3d> poses;         // Twc

  string prefix(argv[2]);
  if (prefix.back() != '/') {
    prefix.append("/");
  }

  std::vector<double> pose;
  for (int i = 0; i < 5; ++i) {
    boost::format fmt(prefix + "%s/%d.%s");  // image file format
    imgs_color.emplace_back(
        cv::imread((fmt % "color" % (i + 1) % "png").str()));
    imgs_depth.emplace_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(),
                                       cv::IMREAD_UNCHANGED));

    for (int j = 0; j < 7; ++j) {
      fin >> pose[j];
    }
    Eigen::Quaterniond q(pose[6], pose[3], pose[4], pose[5]);
    Eigen::Isometry3d T(q);
    T.pretranslate(Eigen::Vector3d(pose[0], pose[1], pose[2]));
    poses.emplace_back(T);
  }

  // Intrinsic parameters
  const double cx = 319.5;
  const double cy = 239.5;
  const double fx = 481.2;
  const double fy = -480.0;
  const double depth_scale = 5000.0;

  cout << "Converting images to point cloud..." << endl;

  // Create a new point cloud
  PointCloud::Ptr point_cloud(new PointCloud);
  for (int i = 0; i < 5; ++i) {
    PointCloud::Ptr current(new PointCloud);
    cout << "Converting image: " << i + 1 << endl;
    const cv::Mat color = imgs_color[i];
    const cv::Mat depth = imgs_depth[i];
    const Eigen::Isometry3d T = poses[i];
    for (int v = 0; v < color.rows; ++v) {
      for (int u = 0; u < color.cols; ++u) {
        const unsigned int d = depth.ptr<unsigned short>(v)[u];  // depth value
        if (d == 0) {
          // 0 means data unavailable
          continue;
        }
        Eigen::Vector3d pc;
        pc[2] = static_cast<double>(d) / depth_scale;
        pc[0] = (u - cx) * pc[2] / fx;
        pc[1] = (v - cy) * pc[2] / fy;
        const Eigen::Vector3d pw = T * pc;

        PointT p;
        p.x = pw[0];
        p.y = pw[1];
        p.z = pw[2];
        p.b = color.data[v * color.step + u * color.channels()];
        p.g = color.data[v * color.step + u * color.channels() + 1];
        p.r = color.data[v * color.step + u * color.channels() + 2];
        current->points.emplace_back(p);
      }
    }
    // depth filter and statistical removal
    PointCloud::Ptr tmp(new PointCloud);
    pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    statistical_filter.setMeanK(50);
    statistical_filter.setStddevMulThresh(1.0);
    statistical_filter.setInputCloud(current);
    statistical_filter.filter(*tmp);
    (*point_cloud) += *tmp;
  }

  point_cloud->is_dense = false;
  cout << "There are " << point_cloud->size() << " points in our point cloud."
       << endl;

  // voxel filter
  pcl::VoxelGrid<PointT> voxel_filter;
  double resolution = 0.03;
  voxel_filter.setLeafSize(resolution, resolution, resolution);  // resolution
  PointCloud::Ptr tmp(new PointCloud);
  voxel_filter.setInputCloud(point_cloud);
  voxel_filter.filter(*tmp);
  tmp->swap(*point_cloud);

  cout << "After filtering，there are total " << point_cloud->size()
       << " points in our point cloud." << endl;

  pcl::io::savePCDFileBinary("map.pcd", *point_cloud);

  return 0;
}