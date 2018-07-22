//
// Created by ccfy on 18-7-18.
//
//
// Created by ccfy on 18-7-16.
// add g2o into vo
#include <iostream>
#include <fstream>
#include <pcl/visualization/cloud_viewer.h>
#include <sophus/se3.h>
#include <pangolin/pangolin.h>

using namespace std;

#include "slamBase.h"

//g2o header file , in g2o edge is measurement and vertex is estimation
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>


double normOfTransform(cv::Mat rvec, cv::Mat tvec)
{
	return fabs(min(cv::norm(rvec),2*M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
}

FRAME readFrame(int index, Config config)
{
	FRAME result;
	string rgbdir= config.getConfigPara().at("rgb_dir").get<string>();
	string rgbext= config.getConfigPara().at("rgb_extension").get<string>();
	string depthdir= config.getConfigPara().at("depth_dir").get<string>();
	string depthext= config.getConfigPara().at("depth_extension").get<string>();
	string rgb_path = rgbdir + to_string(index) + rgbext;
	string depth_path = depthdir + to_string(index) + depthext;
	result.rgb = cv::imread(rgb_path);
	result.depth = cv::imread(depth_path, -1);
	return result;
}

void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses) {
	if (poses.empty()) {
		cerr << "Trajectory is empty!" << endl;
		return;
	}

	// create pangolin window and plot the trajectory
	pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	pangolin::OpenGlRenderState s_cam(
			pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
			pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
	);

	pangolin::View &d_cam = pangolin::CreateDisplay()
			.SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
			.SetHandler(new pangolin::Handler3D(s_cam));


	while (pangolin::ShouldQuit() == false) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		d_cam.Activate(s_cam);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		glLineWidth(2);
		for (size_t i = 0; i < poses.size() - 1; i++) {
			glColor3f(1 - (float) i / poses.size(), 0.0f, (float) i / poses.size());
			glBegin(GL_LINES);
			auto p1 = poses[i], p2 = poses[i + 1];
			glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
			glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
			glEnd();
		}
		pangolin::FinishFrame();
		usleep(5000);   // sleep 5 ms
	}

}

int main(int argc, char** argv)
{
	Config config;
	int start_index = config.getConfigPara().at("start_index").get<int>();
	int end_index = config.getConfigPara().at("end_index").get<int>();
	cout<<"Initializing..."<<endl;
	FRAME lastFrame = readFrame(start_index, config);
	string detector = config.getConfigPara().at("detector").get<string>();
	string descriptor = config.getConfigPara().at("descriptor").get<string>();
	CAMERA_INTRINSIC_PARAMETERS camera = get_camera_intrinsic(config);
	computeKeyPointAndDesc(lastFrame,detector,descriptor);
	PointCloud::Ptr Cloud = image2PointCloud(lastFrame.rgb,lastFrame.depth,camera);
	//pcl::visualization::CloudViewer viewer("viewer");
	bool visualize_pointcloud = true?(config.getConfigPara().at("visualize_pointcloud").get<string>() == "yes"):false;
	int min_inlier = config.getConfigPara().at("min_inlier").get<int>();
	double max_norm = config.getConfigPara().at("max_norm").get<double>();
	int curIndex = start_index + 1;

	//g2o optimization
	typedef g2o::BlockSolver_6_3 slamBlockSolver;
	typedef g2o::LinearSolverEigen<slamBlockSolver::PoseMatrixType> slamLinearSolver;
	slamLinearSolver* linearSolver = new slamLinearSolver();
	linearSolver->setBlockOrdering(false);
	slamBlockSolver* blockSolver = new slamBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
	g2o::SparseOptimizer globalOptimizer;
	//set LM solver and do not output debug info
	globalOptimizer.setAlgorithm(solver);
	globalOptimizer.setVerbose(false);

	g2o::VertexSE3* v = new g2o::VertexSE3();
	v->setId(start_index);
	v->setEstimate(Eigen::Isometry3d::Identity());
	v->setFixed(true);
	globalOptimizer.addVertex(v);

	//construct estimated relative poses of total process
	vector<Sophus::SE3,Eigen::aligned_allocator<Sophus::SE3>> poses;

	int lastIndex = start_index;
	for(;curIndex <= end_index;curIndex++)
	{
		cout<<"Reading files "<<curIndex<<endl;
		FRAME curFrame = readFrame(curIndex, config);
		computeKeyPointAndDesc(curFrame,detector,descriptor);
		RESULT_OF_PNP pnp_result = estimateMotion(lastFrame, curFrame, camera);
		if(pnp_result.inliers<min_inlier)
			continue;
		double norm = normOfTransform(pnp_result.rvec, pnp_result.tvec);
		if(norm > max_norm)
			continue;
		cout<<"Frame "<<curIndex<<" is valid"<<endl;
		Eigen::Isometry3d T = cvMat2Eigen(pnp_result.tvec,pnp_result.rvec);
		cout<<"T = "<<T.matrix()<<endl;
		Sophus::SE3 se3_pose(T.rotation(),T.translation());
		poses.push_back(se3_pose);
		cv::imshow("curframe",curFrame.rgb);
		cv::waitKey(5);
		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId(curIndex);
		v->setEstimate(Eigen::Isometry3d::Identity());
		globalOptimizer.addVertex(v);
		g2o::EdgeSE3* edge = new g2o::EdgeSE3();
		edge->vertices()[0] = globalOptimizer.vertex(lastIndex);
		edge->vertices()[1] = globalOptimizer.vertex(curIndex);
		Eigen::Matrix<double,6,6>information = Eigen::Matrix<double,6,6>::Identity();
		edge->setInformation(information);
		edge->setMeasurement(T);
		globalOptimizer.addEdge(edge);

		//Cloud = joinPointCloud(Cloud,curFrame,camera,T);
//		if(visualize_pointcloud)
//			viewer.showCloud(Cloud);
		lastFrame = curFrame;
		lastIndex = curIndex;
	}
	//pcl::io::savePCDFile("../data/result.pcd",*Cloud);
	//cout<<"result.pcd has been saved"<<endl;
	cout<<"optimizing pose graph, vertices "<<globalOptimizer.vertices().size();
	globalOptimizer.save("../data/result_before.g2o");
	globalOptimizer.initializeOptimization();
	globalOptimizer.optimize(100);
	globalOptimizer.save("../data/result_after.g2o");
	cout<<"Optimization done"<<endl;
	globalOptimizer.clear();
	DrawTrajectory(poses);
	return 0;
}
