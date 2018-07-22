//
// Created by ccfy on 18-7-15.
//
#include <iostream>
using namespace std;

#include "slamBase.h"

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>


int main(int argc, char** argv)
{
	Config config;
	string detector = config.getConfigPara().at("detector").get<string>();
	string descriptor = config.getConfigPara().at("descriptor").get<string>();
	FRAME frame1,frame2;
	frame1.rgb = cv::imread("../data/rgb1.png");
	frame2.rgb = cv::imread("../data/rgb2.png");
	frame1.depth = cv::imread("../data/depth1.png", -1);
	frame2.depth = cv::imread("../data/depth2.png", -1);

	computeKeyPointAndDesc(frame1,detector,descriptor);
	computeKeyPointAndDesc(frame2,detector,descriptor);

	CAMERA_INTRINSIC_PARAMETERS camera;
	camera.cx = config.getConfigPara().at("camera_cx").get<double>();
	camera.cy = config.getConfigPara().at("camera_cy").get<double>();
	camera.fx = config.getConfigPara().at("camera_fx").get<double>();
	camera.fy = config.getConfigPara().at("camera_fy").get<double>();
	camera.scale = config.getConfigPara().at("camera_scale").get<double>();

	RESULT_OF_PNP result = estimateMotion(frame1,frame2,camera);
	cout<<result.rvec<<endl;
	cout<<result.tvec<<endl;

	cv::Mat R;
	cv::Rodrigues(result.rvec,R);
	//transform opencv mat to eigen matrix
	Eigen::Matrix3d r;
	cv::cv2eigen(R,r);

	Eigen::Isometry3d T =Eigen::Isometry3d::Identity();
	Eigen::AngleAxisd angle(r);
	T.rotate(angle);
	T.pretranslate(Eigen::Vector3d(result.tvec.at<double>(0,0),
	                               result.tvec.at<double>(0,1),
								   result.tvec.at<double>(0,2)));
	PointCloud::Ptr cloud1 = image2PointCloud(frame1.rgb,frame1.depth,camera);
	PointCloud::Ptr cloud2 = image2PointCloud(frame2.rgb,frame2.depth,camera);

	PointCloud::Ptr output(new PointCloud());
	pcl::transformPointCloud(*cloud1,*output,T.matrix());
	*output += *cloud2;
	pcl::io::savePCDFile("../data/result.pcd",*output);

	pcl::visualization::CloudViewer viewer("viewer");
	viewer.showCloud(output);
	while(!viewer.wasStopped())
	{

	}
	return 0;

}

