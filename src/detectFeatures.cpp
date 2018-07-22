//
// Created by ccfy on 18-7-15.
//

#include <iostream>
#include "slamBase.h"

using namespace std;

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main(int argc, char** argv)
{
	cv::Mat rgb1 = cv::imread("../data/rgb1.png");
	cv::Mat rgb2 = cv::imread("../data/rgb2.png");
	cv::Mat depth1 = cv::imread("../data/depth1.png", -1);
	cv::Mat depth2 = cv::imread("../data/depth2.png", -1);

	cv::Ptr<cv::FeatureDetector> _detector;
	cv::Ptr<cv::DescriptorExtractor> _descriptor;

	//sift init for detector and descriptor
	cv::initModule_nonfree();
	_detector = cv::FeatureDetector::create("GridSIFT");
	_descriptor = cv::DescriptorExtractor::create("SIFT");

	vector<cv::KeyPoint> kp1,kp2;
	_detector->detect(rgb1,kp1);
	_detector->detect(rgb2,kp2);

	cout<<"key point size of rgb1 is "<<kp1.size()<<endl;
	cout<<"key point size of rgb2 is "<<kp2.size()<<endl;

	cv::Mat keypoint_image;
	cv::drawKeypoints(rgb1, kp1, keypoint_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("ketpoint_image", keypoint_image);
	cv::waitKey(0);

	cv::Mat desp1, desp2;
	_descriptor->compute(rgb1, kp1, desp1);
	_descriptor->compute(rgb2, kp2, desp2);

	vector<cv::DMatch> matches;
	cv::FlannBasedMatcher matcher;
	matcher.match(desp1,desp2,matches);
	cout<<"final total match number is "<<matches.size()<<endl;

	cv::Mat matched_image;
	cv::drawMatches(rgb1,kp1,rgb2,kp2,matches,matched_image);
	cv::imshow("matched image", matched_image);
	cv::waitKey(0);

	vector<cv::DMatch> goodMatches;
	double mindis = 10000;
	for(int i=0;i<matches.size();i++)
	{
		if(matches[i].distance < mindis)
			mindis = matches[i].distance;
	}
	for(auto match : matches)
	{
		if(match.distance < 4 * mindis)
			goodMatches.push_back(match);
	}
	cout<<"good match size is "<<goodMatches.size()<<endl;
	cv::drawMatches(rgb1,kp1,rgb2,kp2,goodMatches,matched_image);
	cv::imshow("good match image",matched_image);
	cv::waitKey(0);

	//calc the r and t (pose) for two image based on pnp
	vector<cv::Point3f> pt3d_imageone;
	vector<cv::Point2f> pt2d_imagetwo;
	CAMERA_INTRINSIC_PARAMETERS camera;
	camera.cx = 325.5;
	camera.cy = 253.5;
	camera.fx = 518.0;
	camera.fy = 519.0;
	camera.scale = 1000.0;
	for(int i=0;i<goodMatches.size();i++)
	{
		cv::Point2f imageone_point = kp1[goodMatches[i].queryIdx].pt;
		ushort d = depth1.ptr<ushort>(int(imageone_point.y))[int(imageone_point.x)];
		if(d == 0)
			continue;
		cv::Point3f p(imageone_point.x, imageone_point.y, d);
		pt3d_imageone.push_back(point2dTo3d(p, camera));
		pt2d_imagetwo.push_back(cv::Point2f(kp2[goodMatches[i].trainIdx].pt));
	}
	double camera_matrix_data[3][3]{
			{camera.fx, 0, camera.cx},
			{0, camera.fy, camera.cy},
			{0, 0, 1}
	};
	cv::Mat camera_matrix_mat(3,3,CV_64F,camera_matrix_data);
	cv::Mat rvec, tvec, inliers;
	cv::solvePnPRansac(pt3d_imageone,pt2d_imagetwo,camera_matrix_mat,cv::Mat(),rvec,tvec,false,100,1.0,100,inliers);
	cout<<"inlier's size is "<<inliers.size()<<endl;
	cout<<"R is "<<rvec<<endl;
	cout<<"T is "<<tvec<<endl;

	vector<cv::DMatch> inlier_match;
	for(int i=0;i<inliers.rows;i++)
	{
		inlier_match.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
	}
	cv::drawMatches(rgb1,kp1,rgb2,kp2,inlier_match,matched_image);
	cv::imshow("inlier match",matched_image);
	cv::waitKey(0);
	return 0;

}