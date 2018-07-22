//
// Created by ccfy on 18-7-15.
//

#include "slamBase.h"
#include <opencv2/core/eigen.hpp>

PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	PointCloud::Ptr pointcloud(new PointCloud);
	for(int v=0;v<rgb.rows;v++)
	{
		for(int u=0;u<rgb.cols;u++)
		{
			ushort d = depth.ptr<ushort>(v)[u];
			if(d == 0)
				continue;
			PointT point;
			point.z = double(d) / camera.scale;
			point.x = (u - camera.cx)/camera.fx*point.z;
			point.y = (v - camera.cy)/camera.fy*point.z;
			point.b = rgb.ptr<uchar>(v)[3*u];
			point.g = rgb.ptr<uchar>(v)[3*u+1];
			point.r = rgb.ptr<uchar>(v)[3*u+2];
			pointcloud->points.push_back(point);

		}
	}
	pointcloud->height = 1;
	pointcloud->is_dense = false;
	pointcloud->width = pointcloud->points.size();
	return pointcloud;
}

cv::Point3f point2dTo3d(cv::Point3f& point,CAMERA_INTRINSIC_PARAMETERS& camera)
{
	cv::Point3f point3d;
	point3d.z = double(point.z)/camera.scale;
	point3d.x = (point.x - camera.cx)/camera.fx*point3d.z;
	point3d.y = (point.y - camera.cy)/camera.fy*point3d.z;
	return point3d;
}

void computeKeyPointAndDesc(FRAME& frame,string detector, string describtor)
{
	cv::Ptr<cv::FeatureDetector> _detector;
	cv::Ptr<cv::DescriptorExtractor> _descriptor;

	//sift init for detector and descriptor
	cv::initModule_nonfree();
	_detector = cv::FeatureDetector::create(detector.c_str());
	_descriptor = cv::DescriptorExtractor::create(describtor.c_str());
    if(!_detector || !_descriptor)
	{
		cerr<<"unknown detector type and descibe type"<<detector<<", "<<describtor<<endl;
		return;
	}
	vector<cv::KeyPoint> kp;
	_detector->detect(frame.rgb,frame.kp);

	cv::Mat desp;
	_descriptor->compute(frame.rgb, frame.kp, frame.desp);
	return;
}

RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	RESULT_OF_PNP result;
	vector<cv::DMatch> matches;
	cv::FlannBasedMatcher matcher;
	matcher.match(frame1.desp,frame2.desp,matches);
	Config config;
	double good_match_threshold = config.getConfigPara().at("good_match_threshold").get<double>();
	double min_good_match = config.getConfigPara().at("min_good_match").get<int>();
	vector<cv::DMatch> goodMatches;
	double mindis = 10000;
	for(int i=0;i<matches.size();i++)
	{
		if(matches[i].distance < mindis)
			mindis = matches[i].distance;
	}
	for(auto match : matches)
	{
		if(match.distance < good_match_threshold * mindis)
			goodMatches.push_back(match);
	}

	//calc the r and t (pose) for two image based on pnp
	vector<cv::Point3f> pt3d_imageone;
	vector<cv::Point2f> pt2d_imagetwo;
    if(goodMatches.size() < min_good_match)
	{
		result.inliers = 0;
		return result;
	}
	for(int i=0;i<goodMatches.size();i++)
	{
		cv::Point2f imageone_point = frame1.kp[goodMatches[i].queryIdx].pt;
		ushort d = frame1.depth.ptr<ushort>(int(imageone_point.y))[int(imageone_point.x)];
		if(d == 0)
			continue;
		cv::Point3f p(imageone_point.x, imageone_point.y, d);
		pt3d_imageone.push_back(point2dTo3d(p, camera));
		pt2d_imagetwo.push_back(cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt));
	}
	double camera_matrix_data[3][3]{
			{camera.fx, 0, camera.cx},
			{0, camera.fy, camera.cy},
			{0, 0, 1}
	};
	cv::Mat camera_matrix_mat(3,3,CV_64F,camera_matrix_data);
	cv::Mat rvec, tvec, inliers;
	cv::solvePnPRansac(pt3d_imageone,pt2d_imagetwo,camera_matrix_mat,cv::Mat(),rvec,tvec,false,100,1.0,100,inliers);
	result.rvec = rvec;
	result.tvec = tvec;
	result.inliers = inliers.rows;
	return result;
}


//convert rotate and translate to T matrix
Eigen::Isometry3d cvMat2Eigen(cv::Mat& tvec, cv::Mat& rvec)
{
	cv::Mat R;
	cv::Rodrigues(rvec,R);
	//transform opencv mat to eigen matrix
	Eigen::Matrix3d r;
	cv::cv2eigen(R,r);

	Eigen::Isometry3d T =Eigen::Isometry3d::Identity();
	Eigen::AngleAxisd angle(r);
	T.rotate(angle);
	T.pretranslate(Eigen::Vector3d(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2)));
	return T;
}

//combine pointcloud from original and new
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newframe, CAMERA_INTRINSIC_PARAMETERS& camera, Eigen::Isometry3d T)
{
	PointCloud::Ptr newPointCloud = image2PointCloud(newframe.rgb,newframe.depth,camera);
	PointCloud::Ptr output(new PointCloud());
	pcl::transformPointCloud(*original,*output,T.matrix());
	*newPointCloud += *output;

	//voxel grid filter down-sample
	Config config;
	static pcl::VoxelGrid<PointT> voxel;
	double gridsize = config.getConfigPara().at("voxel_grid").get<double>();
	voxel.setLeafSize(gridsize,gridsize,gridsize);
	voxel.setInputCloud(newPointCloud);
	PointCloud::Ptr tmp(new PointCloud());
	voxel.filter(*tmp);
	return tmp;
}

CAMERA_INTRINSIC_PARAMETERS get_camera_intrinsic(Config& config)
{
	CAMERA_INTRINSIC_PARAMETERS camera;
	camera.cx = config.getConfigPara().at("camera_cx").get<double>();
	camera.cy = config.getConfigPara().at("camera_cy").get<double>();
	camera.fx = config.getConfigPara().at("camera_fx").get<double>();
	camera.fy = config.getConfigPara().at("camera_fy").get<double>();
	camera.scale=config.getConfigPara().at("camera_scale").get<double>();
	return camera;
}