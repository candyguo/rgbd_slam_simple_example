#include "slamBase.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Sophus;

const int border = 20;
const int width = 640;
const int height = 480;
const double fx = 481.2f;
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 2; // ncc window half width
const int ncc_area = (2*ncc_window_size+1) * (2*ncc_window_size+1);
const double min_cov = 0.1;//收敛判定 收敛方差
const double max_cov = 10; //发散判定 最大方差

bool readDataSetFile(const string& path, vector<string>& color_image_files,vector<Sophus::SE3>& poses);

//深度融合的更新估计
bool update(const cv::Mat& ref,const Mat& cur,const Sophus::SE3& T_C_R,cv::Mat& depth,cv::Mat& depth_cov);

bool epipolarSearch(const cv::Mat& ref,const Mat& cur,const Sophus::SE3& T_C_R,
					const Eigen::Vector2d& pt_ref,const double& depth_mu,
                    const double& depth_cov,Eigen::Vector2d& pt_cur);

bool updateDepthFilter(const Eigen::Vector2d& pt_ref,const Eigen::Vector2d& pt_cur,
                       const Sophus::SE3& T_C_R,const cv::Mat& depth,cv::Mat& depth_conv);

double NCC(const cv::Mat& ref,const cv::Mat& cur,const Eigen::Vector2d& pt_ref,const Eigen::Vector2d& pt_cur);

// 双线性灰度插值
inline double getBilinearInterpolatedValue( const cv::Mat& img, const Eigen::Vector2d& pt ) {
	uchar* d = & img.data[ int(pt(1,0))*img.step+int(pt(0,0)) ];
	double xx = pt(0,0) - floor(pt(0,0));
	double yy = pt(1,0) - floor(pt(1,0));
	return  (( 1-xx ) * ( 1-yy ) * double(d[0]) +
			 xx* ( 1-yy ) * double(d[1]) +
			 ( 1-xx ) *yy* double(d[img.step]) +
			 xx*yy*double(d[img.step+1]))/255.0;
}

void plotDepth( const Mat& depth );

// 像素到相机坐标系
inline Vector3d px2cam ( const Vector2d px ) {
	return Vector3d (
			(px(0,0) - cx)/fx,
			(px(1,0) - cy)/fy,
			1
	);
}

// 相机坐标系到像素
inline Vector2d cam2px ( const Vector3d p_cam ) {
	return Vector2d (
			p_cam(0,0)*fx/p_cam(2,0) + cx,
			p_cam(1,0)*fy/p_cam(2,0) + cy
	);
}

// 检测一个点是否在图像边框内
inline bool inside( const Vector2d& pt ) {
	return pt(0, 0) >= border && pt(1, 0) >= border
		   && pt(0, 0) + border < width && pt(1, 0) + border <= height;
}

// 显示极线匹配
void showEpipolarMatch( const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_curr );

// 显示极线
void showEpipolarLine( const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_min_curr,
					   const Vector2d& px_max_curr );

int main(int argc,char** argv)
{
	//利用第一帧之后的所有帧对第一帧的深度进行融合更新
	vector<string> color_image_files;
	vector<SE3> poses;
	if(argc != 2)
	{
		cout<<"dataset path must be given"<<endl;
		return -1;
	}
	if(!readDataSetFile(argv[1],color_image_files,poses))
		return -1;
	cout<<"read total "<<color_image_files.size()<<" images"<<endl;
	Mat ref = imread(color_image_files[0],0);
	SE3 pose_ref_twc = poses[0];
	double init_depth=3.0;
	double init_cov = 3.0;
	Mat depth = Mat(height,width,CV_64F,init_depth);
	Mat depth_cov = Mat(height,width,CV_64F,init_cov);

	for(int i=1;i<color_image_files.size();i++)
	{
		cout<<"loop "<<"i"<<" image"<<endl;
		Mat cur = imread(color_image_files[i],0);
		if(cur.empty())
		{
			cout<<"current image "<< i << "read failed"<<endl;
			continue;
		}
		Sophus::SE3 T_C_R = poses[i].inverse() * pose_ref_twc;
		update(ref,cur,T_C_R,depth,depth_cov);
		imshow("cur",cur);
		plotDepth(depth);
		waitKey(1);
	}

	return 0;
}


bool readDataSetFile(
		const string& path,
		vector< string >& color_image_files,
		std::vector<Sophus::SE3>& poses
)
{
	//uzh.ifi dataset 单目color & pose
	ifstream fin( path+"/first_200_frames_traj_over_table_input_sequence.txt");
	if ( !fin ) return false;

	while ( !fin.eof() )
	{
		// 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
		string image;
		fin>>image;
		double data[7];
		for ( double& d:data ) fin>>d;
		color_image_files.push_back( path+string("/images/")+image );
		poses.push_back(
				SE3( Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
					 Vector3d(data[0], data[1], data[2]))
		);
		if ( !fin.good() ) break;
	}
	return true;
}

double NCC(const cv::Mat& ref,const cv::Mat& cur,const Eigen::Vector2d& pt_ref,const Eigen::Vector2d& pt_cur)
{
	double mean_ref = 0;
	double mean_cur = 0;
	vector<double> refvalues;
	vector<double> curvalues;
	for(int x = -ncc_window_size;x<=ncc_window_size;x++)
	{
		for(int y = -ncc_window_size;y<=ncc_window_size;y++)
		{
			//()[] 行在前，列在后
			double refvalue = double(ref.ptr<uchar>(int(y+pt_ref(1,0)))[int(x+pt_ref(0,0))]/255);
			double curvalue = getBilinearInterpolatedValue(cur,pt_cur+Vector2d(x,y));
			mean_ref += refvalue;
			mean_cur += curvalue;
			refvalues.push_back(refvalue);
			curvalues.push_back(curvalue);
		}
	}
	mean_ref /= ncc_area;
	mean_ref /= ncc_area;

	double fenzi = 0;
	double demontier1 = 0;
	double demontier2 = 0;
	for(int i=0;i<refvalues.size();i++)
	{
		fenzi += (refvalues[i]-mean_ref)*(curvalues[i]-mean_cur);
		demontier1 += (refvalues[i]-mean_ref)*(refvalues[i]-mean_ref);
		demontier2 += (curvalues[i]-mean_cur)*(curvalues[i]-mean_cur);
	}
	return fenzi/sqrt(demontier1*demontier2+1e-10);
}

bool epipolarSearch(const cv::Mat& ref,const Mat& cur,const Sophus::SE3& T_C_R,
					const Eigen::Vector2d& pt_ref,const double& depth_mu,
					const double& depth_cov,Eigen::Vector2d& pt_cur)
{
	//非特征点的块匹配 极线搜索当前点(在极线上搜索最高ncc点)
	Eigen::Vector3d f_ref = px2cam(pt_ref);
	f_ref.normalize();
	Vector3d P_ref = f_ref * depth_mu;
	Vector2d px_mean_curr = cam2px(T_C_R*P_ref);
	double d_min = max(depth_mu - 3* depth_cov,0.1);
	double d_max = depth_mu + 3* depth_cov;

	Vector2d px_min_curr = cam2px(T_C_R*(f_ref*d_min));
	Vector2d px_max_curr = cam2px(T_C_R*(f_ref*d_max));

	Vector2d epipolar_line = px_max_curr - px_min_curr;
	Vector2d epipolar_dir = epipolar_line;
	epipolar_dir.normalize();

	double half_length = std::min(epipolar_line.norm(),100.0);

	showEpipolarLine(ref, cur, pt_ref, px_min_curr, px_max_curr);

	double best_ncc = -1;
	Vector2d best_pos;
	for(double l = -half_length;l<=half_length;l+=0.7)
	{
		Vector2d curpx = px_mean_curr + l * epipolar_dir;
		if(!inside(curpx))
			continue;
		double tmpncc = NCC(ref,cur,pt_ref,curpx);
		if(tmpncc > best_ncc)
		{
			best_ncc = tmpncc;
			best_pos = curpx;
		}
	}
	if(best_ncc < 0.85)
		return false;
	pt_cur = best_pos;
	return true;
}

void plotDepth(const Mat& depth)
{
	imshow( "depth", depth*0.4 );
	waitKey(1);
}

void showEpipolarMatch(const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_curr)
{
	Mat ref_show, curr_show;
	cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
	cv::cvtColor( curr, curr_show, CV_GRAY2BGR );

	cv::circle( ref_show, cv::Point2f(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0,0,250), 2);
	cv::circle( curr_show, cv::Point2f(px_curr(0,0), px_curr(1,0)), 5, cv::Scalar(0,0,250), 2);

	imshow("ref", ref_show );
	imshow("curr", curr_show );
	waitKey(1);
}

void showEpipolarLine(const Mat& ref, const Mat& curr, const Vector2d& px_ref,
					  const Vector2d& px_min_curr, const Vector2d& px_max_curr)
{

	Mat ref_show, curr_show;
	cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
	cv::cvtColor( curr, curr_show, CV_GRAY2BGR );

	cv::circle( ref_show, cv::Point2f(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0,255,0), 2);
	cv::circle( curr_show, cv::Point2f(px_min_curr(0,0), px_min_curr(1,0)), 5, cv::Scalar(0,255,0), 2);
	cv::circle( curr_show, cv::Point2f(px_max_curr(0,0), px_max_curr(1,0)), 5, cv::Scalar(0,255,0), 2);
	cv::line( curr_show, Point2f(px_min_curr(0,0), px_min_curr(1,0)), Point2f(px_max_curr(0,0),
			  px_max_curr(1,0)), Scalar(0,255,0), 1);

	imshow("ref", ref_show );
	imshow("curr", curr_show );
	waitKey(1);
}

bool updateDepthFilter(const Eigen::Vector2d& pt_ref,const Eigen::Vector2d& pt_cur,
					   const Sophus::SE3& T_C_R,cv::Mat& depth,cv::Mat& depth_conv)
{
	//深度滤波器 高斯融合 参考www.mamicode.com/info-detail-2061030.html
	// T_C_R 代表从参考（世界）到当前的转换（从右到左看）
	SE3 T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
	f_ref.normalize(); //左像特征点的光线指向
	Vector3d f_cur = px2cam(pt_cur);
	f_cur.normalize();
	Vector3d t = T_R_C.translation();
	Matrix3d R = T_R_C.rotation_matrix();
	Matrix2d A;
	A << f_ref.transpose() * f_ref, -f_ref.transpose()*R*f_cur,
			(R*f_cur).transpose()*f_ref, -(R*f_cur).transpose()*(R*f_cur);
	Vector2d b;
	b << f_ref.transpose()*t, (R*f_cur).transpose()*t;
	//cremal法则求解s1,s2 s1 = detB1/detA s2 = detB2/detA
	double detb1 = b(0)*A(1,1) - b(1)*A(0,1);
	double detb2 = A(0,0)*b(1) - b(0)*A(1,0);
	double s1 = detb1/A.determinant();
	double s2 = detb2/A.determinant();
	//s1*x1 = s2*R*x2 + t
	Vector3d xm = s1* f_ref;
	Vector3d xn = t + s2 * R* f_cur;
	Vector3d depthvec = (xm + xn) / 2;
	double depth_estimation = depthvec.norm();

	//计算不确定性 观测误差一个像素引起的深度误差
	double alpha = acos(f_ref.dot(t) / t.norm());
	double beta = acos(((R*f_cur).dot(-t)) / t.norm());
	double new_beta = beta - atan(1/fx);
	double gamma = M_PI - alpha - new_beta;
	double p_cov = depth_estimation - t.norm()*sin(new_beta)/sin(gamma);

	//利用观测的均值和方差进行高斯融合
	double cov2 = depth_conv.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))];
	double u = depth.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))];
	double u_fuse = (p_cov*p_cov*u + cov2*depth_estimation)/(p_cov*p_cov + cov2);
	double cov2_fuse = (p_cov*p_cov*cov2)/(p_cov*p_cov + cov2);

	depth.at<double>(int(pt_ref(1,0)),int(pt_ref(0,0)))= u_fuse;
	depth_conv.at<double>(int(pt_ref(1,0)),int(pt_ref(0,0))) = cov2_fuse;
    return true;
}

//深度融合的更新估计
bool update(const cv::Mat& ref,const Mat& cur,const Sophus::SE3& T_C_R,cv::Mat& depth,cv::Mat& depth_cov)
{
	for(int x = border;x<=width-border;x++)
	{
		for(int y = border;y<=height-border;y++)
		{
			//遍历参考图像的像素，寻找极线匹配点
			if(depth_cov.ptr<double>(y)[x] > max_cov || depth_cov.ptr<double>(y)[x] < min_cov)
				continue;
			Vector2d pt_cur;
			bool  ret = epipolarSearch(ref,cur,T_C_R,Vector2d(x,y),
					depth.ptr<double>(y)[x],depth_cov.ptr<double>(y)[x],pt_cur);
			if(!ret)
				continue;
			showEpipolarMatch(ref,cur,Vector2d(x,y),pt_cur);
			//逐像素深度滤波
			updateDepthFilter(Vector2d(x,y),pt_cur,T_C_R,depth,depth_cov);
		}
	}

}