//
// Created by ccfy on 18-7-15.
//

#ifndef MYRGBD_SLAM_SLAMBASE_H
#define MYRGBD_SLAM_SLAMBASE_H

#endif //MYRGBD_SLAM_SLAMBASE_H

#include <fstream>
#include <vector>
#include <iostream>
#include "json.hpp"

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

using json = nlohmann::json;

class Config{
public:
	json config_para;
	Config() {}

public:
	void setParameterFile()
	{
		std::string filename = "../config/rgbd_slam.json";
		std::ifstream config_file(filename);
		config_file >> config_para;
		config_file.close();
	}

	json getConfigPara()
	{
		setParameterFile();
		return config_para;
	}

};


struct CAMERA_INTRINSIC_PARAMETERS{
	double cx, cy;
	double fx,fy;
	double scale;
};

struct FRAME{
	int FrameId;
	cv::Mat rgb, depth;
	cv::Mat desp;
	vector<cv::KeyPoint> kp;
};

struct RESULT_OF_PNP{
	cv::Mat rvec, tvec;
	int inliers;
	int goodMatch;
};

//rgb image to pointcloud
PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera);

//image point(u,v,d) to three dimension point
cv::Point3f point2dTo3d(cv::Point3f& point,CAMERA_INTRINSIC_PARAMETERS& camera);

void computeKeyPointAndDesc(FRAME& frame,string detector, string describtor);

RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera);

//convert rotate and translate to T matrix
Eigen::Isometry3d cvMat2Eigen(cv::Mat& tvec, cv::Mat& rvec);

//combine pointcloud from original and new
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newframe, CAMERA_INTRINSIC_PARAMETERS& camera, Eigen::Isometry3d T);

CAMERA_INTRINSIC_PARAMETERS get_camera_intrinsic(Config& config);