//
// Created by ccfy on 18-7-22.
// based on slmaEnd, add loop detection

#include <iostream>
#include <fstream>
#include <sophus/se3.h>
#include <pangolin/pangolin.h>

using namespace std;

#include "slamBase.h"
#include <glog/logging.h>

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

//******************algorthiom of slam with loop detection ***********************
//初始化关键帧序列：F，并将第一帧f0放入F。
//对于新来的一帧I，计算F中最后一帧与I的运动，并估计该运动的大小e。有以下几种可能性：
//若e>Eerror，说明运动太大，可能是计算错误，丢弃该帧；
//若没有匹配上（match太少），说明该帧图像质量不高，丢弃；
//若e<Ekey，说明离前一个关键帧很近，同样丢弃；
//剩下的情况，只有是特征匹配成功，运动估计正确，同时又离上一个关键帧有一定距离，则
// 把I作为新的关键帧，进入回环检测程序：
//近距离回环：匹配I与F末尾m个关键帧。匹配成功时，在图里增加一条边。
//随机回环：随机在F里取n个帧，与I进行匹配。若匹配上，在图里增加一条边。
//将I放入F末尾。若有新的数据，则回2； 若无，则进行优化与地图拼接。
//********************************************************************************

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
	result.FrameId = index;
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

typedef g2o::BlockSolver_6_3 slamBlockSolver;
typedef g2o::LinearSolverEigen<slamBlockSolver::PoseMatrixType> slamLinearSolver;

//两帧的匹配结果
enum CHECK_RESULT {
	NOT_MATCHED = 0,TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME
};

CHECK_RESULT check_key_frames(FRAME& f1,FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false);
void checkNearbyLoops(vector<FRAME>&frames, FRAME& curframe,g2o::SparseOptimizer& opti);
void checkRandomLoops(vector<FRAME>&frames,FRAME& curframe,g2o::SparseOptimizer& opti);

int main(int argc, char** argv)
{
	google::InitGoogleLogging(argv[0]);
	FLAGS_log_dir = "../log";
	Config config;
	int start_index = config.getConfigPara().at("start_index").get<int>();
	int end_index = config.getConfigPara().at("end_index").get<int>();
	string check_loop_closure = config.getConfigPara().at("check_loop_closure").get<string>();
	vector<FRAME>keyframes;
	cout<<"Initializing..."<<endl;
	FRAME lastFrame = readFrame(start_index, config);
	string detector = config.getConfigPara().at("detector").get<string>();
	string descriptor = config.getConfigPara().at("descriptor").get<string>();
	CAMERA_INTRINSIC_PARAMETERS camera = get_camera_intrinsic(config);
	computeKeyPointAndDesc(lastFrame,detector,descriptor);
	int curIndex = start_index + 1;

	//g2o optimization
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

	keyframes.push_back(lastFrame);
	//construct estimated relative poses of total process
	vector<Sophus::SE3,Eigen::aligned_allocator<Sophus::SE3>> poses;

	int lastIndex = start_index;
	for(;curIndex <= end_index;curIndex++)
	{
		cout<<"Reading files "<<curIndex<<endl;
		FRAME curFrame = readFrame(curIndex, config);
		cv::imshow("curframe",curFrame.rgb);
		cv::waitKey(5);
		computeKeyPointAndDesc(curFrame,detector,descriptor);
		CHECK_RESULT result = check_key_frames(keyframes.back(),curFrame,globalOptimizer);
		switch (result)
		{
			case NOT_MATCHED:
				cout<<"not enough good match inliers in frame "<<curFrame.FrameId<<endl;
				break;
			case TOO_CLOSE:
				cout<<"frame "<<curFrame.FrameId<<" and frame "<<keyframes.back().FrameId<<" is too close"<<endl;
				break;
			case TOO_FAR_AWAY:
				cout<<"frame "<<curFrame.FrameId<<" and frame "<<keyframes.back().FrameId<<" is too far away"<<endl;
				break;
			case KEYFRAME:
				cout<<"frame "<<curFrame.FrameId<<" is the new keyframe"<<endl;
				//enter into the process of closure loop detection
				if(check_loop_closure == "yes")
				{
					cout<<"start loop detection for "<<curFrame.FrameId<<endl;
					checkNearbyLoops(keyframes,curFrame,globalOptimizer);
					checkRandomLoops(keyframes,curFrame,globalOptimizer);
				}
				keyframes.push_back(curFrame);
				break;
			default:
				break;
		}
	}
	//pcl::io::savePCDFile("../data/result.pcd",*Cloud);
	//cout<<"result.pcd has been saved"<<endl;
	cout<<"optimizing pose graph, vertices "<<globalOptimizer.vertices().size()<<endl;
	cout<<"optimizing pose graph, edges "<<globalOptimizer.edges().size()<<endl;
	globalOptimizer.save("../data/loop_result_before.g2o");
	globalOptimizer.initializeOptimization();
	globalOptimizer.optimize(100);
	globalOptimizer.save("../data/loop_result_after.g2o");
	cout<<"Optimization done"<<endl;

	//contruct the pointcloud map
	cout<<"saving thre point cloud map..."<<endl;
	PointCloud::Ptr output(new PointCloud());
	PointCloud::Ptr tmp(new PointCloud());
	pcl::VoxelGrid<PointT> voxel;
	pcl::PassThrough<PointT> pass;
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0,4.0);
	double gridsize = config.getConfigPara().at("voxel_grid").get<double>();
	voxel.setLeafSize(gridsize,gridsize,gridsize);
	for(int i=0;i<keyframes.size();i++)
	{
		cv::imshow("keyframe",keyframes[i].rgb);
		cv::waitKey(5);
		g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].FrameId ));
		Eigen::Isometry3d pose = vertex->estimate();
		Sophus::SE3 se3_pose(pose.rotation(),pose.translation());
		poses.push_back(se3_pose);
        PointCloud::Ptr newCloud = image2PointCloud(keyframes[i].rgb,keyframes[i].depth,camera);
		voxel.setInputCloud(newCloud);
		voxel.filter(*tmp);
		pass.setInputCloud(tmp);
		pass.filter(*newCloud);
		pcl::transformPointCloud( *newCloud, *tmp, pose.matrix());
		*output += *tmp;
		tmp->clear();
		newCloud->clear();
	}
	voxel.setInputCloud(output);
	voxel.filter(*tmp);
	pcl::io::savePCDFile("../data/result.pcd", *tmp);
	cout<<"final map is saved."<<endl;
	//globalOptimizer.clear();
	DrawTrajectory(poses);
	return 0;
}

CHECK_RESULT check_key_frames(FRAME& f1,FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
	Config config;
	static int min_inliers = config.getConfigPara().at("min_inlier").get<int>();
	static double max_norm = config.getConfigPara().at("max_norm").get<double>();
	static double keyframe_threshold = config.getConfigPara().at("keyframe_threshold").get<double>();
	static double max_norm_lp = config.getConfigPara().at("max_norm_lp").get<double>();
	static CAMERA_INTRINSIC_PARAMETERS camera = get_camera_intrinsic(config);
	static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct("Cauchy");
	RESULT_OF_PNP result = estimateMotion(f1,f2,camera);
	if(result.inliers < min_inliers)
		return NOT_MATCHED;
	double norm = normOfTransform(result.rvec,result.tvec);
	if(is_loops == false)
	{
		if(norm >= max_norm)
			return TOO_FAR_AWAY;
	}
	else{
		if(norm >= max_norm_lp)
			return TOO_FAR_AWAY;
	}
	if(norm <= keyframe_threshold)
		return TOO_CLOSE;
	//only process ketframe
	if(is_loops == false)
	{
		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId(f2.FrameId);
		v->setEstimate(Eigen::Isometry3d::Identity());
		opti.addVertex(v);
	}
	g2o::EdgeSE3* edge = new g2o::EdgeSE3();
	edge->vertices()[0]=opti.vertex(f1.FrameId);
	edge->vertices()[1]=opti.vertex(f2.FrameId);
	edge->setRobustKernel(robustKernel);
	Eigen::Matrix<double,6,6>infomation = Eigen::Matrix<double,6,6>();
	infomation<<100,0,0,0,0,0,
	            0,100,0,0,0,0,
	            0,0,100,0,0,0,
	            0,0,0,100,0,0,
	            0,0,0,0,100,0,
	            0,0,0,0,0,100;
	edge->setInformation(infomation);
	Eigen::Isometry3d T = cvMat2Eigen(result.tvec,result.rvec);
	edge->setMeasurement(T.inverse());
	LOG(INFO) <<"T.INVERSE is used";
	opti.addEdge(edge);
	return KEYFRAME;

}


void checkNearbyLoops(vector<FRAME>&frames, FRAME& curframe,g2o::SparseOptimizer& opti)
{
    Config config;
	static int nearby_loops = config.getConfigPara().at("nearby_loops").get<int>();
	if(frames.size()<nearby_loops)
	{
		for(int i=0;i<frames.size();i++)
		{
			check_key_frames(frames[i],curframe,opti,true);
		}
	} else{
		for(int i=frames.size()-nearby_loops;i<frames.size();i++)
		{
			check_key_frames(frames[i],curframe,opti,true);
		}
	}
}
void checkRandomLoops(vector<FRAME>&frames,FRAME& curframe,g2o::SparseOptimizer& opti)
{
    Config config;
	static int random_loops = config.getConfigPara().at("random_loops").get<int>();
	srand((unsigned int)time(nullptr));
	if(frames.size()<random_loops)
	{
		for(int i=0;i<frames.size();i++)
		{
			check_key_frames(frames[i],curframe,opti,true);
		}
	} else{
		for(int i=0;i<random_loops;i++)
		{
			check_key_frames(frames[rand()%frames.size()],curframe,opti,true);
		}
	}
}
