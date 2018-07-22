//
// Created by ccfy on 18-7-8.
//

#include <iostream>
#include <string>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

const double camera_depth_factor = 1000; // mm -- m
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

int main()
{
    cv::Mat rgb, depth;
	rgb = cv::imread("../data/rgb.png");
	depth = cv::imread("../data/depth.png",-1);

	PointCloud::Ptr cloud(new PointCloud);
	for(int m=0;m<depth.rows;m++)
	{
		for(int n=0;n<depth.cols;n++)
		{
			ushort d = depth.ptr<ushort>(m)[n];
			if(d == 0)
				continue;
			PointT point;
			point.z = double(d)/camera_depth_factor;
			point.x = (n-camera_cx)/camera_fx*point.z;
			point.y = (m-camera_cy)/camera_fy*point.z;
			point.b = rgb.ptr<uchar>(m)[3*n];
			point.g = rgb.ptr<uchar>(m)[3*n+1];
			point.r = rgb.ptr<uchar>(m)[3*n+2];
			cloud->points.push_back(point);
		}
	}
	cloud->height = 1;
	cloud->width = cloud->points.size();
	cout<<"point cloud size = "<<cloud->points.size()<<endl;
	cloud->is_dense = false;
	pcl::io::savePCDFile("./point_cloud.pcd", *cloud);
	cout<<"point cloud saved"<<endl;
	return 0;
}