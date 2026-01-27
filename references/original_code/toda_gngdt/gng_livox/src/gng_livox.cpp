#include <functional>
#include <memory>
#include <time.h>
#include <stdlib.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <tf2/convert.h>

extern "C" {
#include "gng.h"
#include "icp.h"
};
#include "rnd.h"
#include "malloc.h"
#include "parameter.h"

using std::placeholders::_1;

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr g_pub_gng_node;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_gng_edge ;
rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_vel;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_untra_node;


// rclcpp::Node::SharedPtr node;
// rclcpp::Node::SharedPtr edge_node;
// rclcpp::Node::SharedPtr cmd_vel_node;
// rclcpp::Node::SharedPtr untra_node;

double sensor_data[10][3] = { {588.087201, 403.764095, 0.0},
{725.731279,610.876711,0.0},
{795.38176,708.708972,0.0},
{846.259998,795.062993, 0.0},
{964.575821,946.379336,0.0},
{-645.072269,427.077179,0.0},
{-780.072272,608.955419,0.0},
{-845.439412,700.241324,0.0},
{-909.534984,817.973766,0.0},
{-973.709398,950.118438,0.0}};

int time_count = 0;//ë‹è¨òHóp
std::chrono::system_clock::time_point des_time;
int turn_flag = 0;
int init_finish = 0;//óßÇøè„Ç∞èIÇÌÇË
int c_label = 0;
double current_loc_data[3];
double prev_loc_data[3];

void transpose_Matrix(double T1[3][3], double T2[3][3])
{
    int i,j;
    for(i=0;i<3;i++){
        for(j=0;j<3;j++){
            T2[i][j] = T1[j][i];
        }
    }
    
}

void product_matrix(double A[3][3], double B[3][3], double C[3][3])
{
    int i,j,k;
    for(i=0;i<3;i++)
        for(j=0;j<3;j++)
            C[i][j] = 0;
    
    for(i=0;i<3;i++){
        for(j=0;j<3;j++){
            for(k=0;k<3;k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

class pointcloud_subscriber : public rclcpp::Node
{
public:
  // use_tf
  std::string gng_main_frame_id;
  std::string gng_crop_frame_id;
  double gng_crop_min_x;
  double gng_crop_max_x;
  double gng_crop_min_y;
  double gng_crop_max_y;
  double gng_crop_min_z;
  double gng_crop_max_z;
  double gng_crop_distance;

  // 矩形除外領域のパラメータを追加
  bool use_ignore_region;
  double ignore_region_min_x;
  double ignore_region_max_x;
  double ignore_region_min_y;
  double ignore_region_max_y;
  double ignore_region_min_z;
  double ignore_region_max_z;

public:
  pointcloud_subscriber()
  : Node("pointcloud_subscriber")
  {
	double tmp[3][3];
	trobot_param.rot[0][0] = 1.0,trobot_param.rot[0][1] = 0.0,trobot_param.rot[0][2] = 0.0;
	trobot_param.rot[1][0] = 0.0,trobot_param.rot[1][1] = 1.0,trobot_param.rot[1][2] = 0.0;
	trobot_param.rot[2][0] = 0.0,trobot_param.rot[2][1] = 0.0,trobot_param.rot[2][2] = 1.0;
	
	declare_parameter("angle_x", RGBD_CAMERA_ANGLE);
    get_parameter("angle_x", trobot_param.angle_x);
	trobot_param.angle_x = M_PI*trobot_param.angle_x/180.0;
	double tmpx[3][3];
	tmpx[0][0] = 1.0, tmpx[0][1] = 0.0, tmpx[0][2] = 0.0;
	tmpx[1][0] = 0.0, tmpx[1][1] = cos(trobot_param.angle_x), tmpx[1][2] = -sin(trobot_param.angle_x);
	tmpx[2][0] = 0.0, tmpx[2][1] = sin(trobot_param.angle_x), tmpx[2][2] = cos(trobot_param.angle_x);

	declare_parameter("angle_y", RGBD_CAMERA_ANGLE);
    get_parameter("angle_y", trobot_param.angle_y);
	trobot_param.angle_y = M_PI*trobot_param.angle_y/180.0;
	double tmpy[3][3];
	tmpy[0][0] = cos(trobot_param.angle_y), tmpy[0][1] = 0.0, tmpy[0][2] = sin(trobot_param.angle_y);
	tmpy[1][0] = 0.0, tmpy[1][1] = 1.0, tmpy[1][2] = 0.0;
	tmpy[2][0] = -sin(trobot_param.angle_y), tmpy[2][1] = 0.0, tmpy[2][2] = cos(trobot_param.angle_y);

	product_matrix(tmpy, tmpx, tmp);
	
	declare_parameter("angle_z", RGBD_CAMERA_ANGLE);
    get_parameter("angle_z", trobot_param.angle_z);
	trobot_param.angle_z = M_PI*trobot_param.angle_z/180.0;
	double tmpz[3][3];
	tmpz[0][0] = cos(trobot_param.angle_z), tmpz[0][1] = -sin(trobot_param.angle_z), tmpz[0][2] = 0.0;
	tmpz[1][0] = sin(trobot_param.angle_z), tmpz[1][1] = cos(trobot_param.angle_z), tmpz[1][2] = 0.0;
	tmpz[2][0] = 0.0, tmpz[2][1] = 0.0, tmpz[2][2] = 1.0;
	product_matrix(tmp, tmpz, trobot_param.rot);
	transpose_Matrix(trobot_param.rot, trobot_param.rot_t);

	printf("%f, %f, %f\n",trobot_param.angle_x,trobot_param.angle_y,trobot_param.angle_z);
	

	declare_parameter("camera_height", 0.0);
    get_parameter("camera_height", trobot_param.camera_height);
	printf("%f\n",trobot_param.camera_height);
	//declare_parameter("pointCloudTopic", "livox/lidar");
	declare_parameter("pointCloudTopic", "livox/lidar_192_168_1_146");
    get_parameter("pointCloudTopic", pointCloudTopic);

	// declare_parameter("gngViewFrame", "center_livox_frame");
    // get_parameter("gngViewFrame", gngViewFrame);

	// use_tf
	gng_main_frame_id = this->declare_parameter<std::string>("gng_main_frame_id","odom");
	gng_crop_frame_id = this->declare_parameter<std::string>("gng_crop_frame_id","base_footprint");
	gng_crop_min_x = this->declare_parameter<double>("gng_crop_min_x", 0.0);
	gng_crop_max_x = this->declare_parameter<double>("gng_crop_max_x", 999.9);
	gng_crop_min_y = this->declare_parameter<double>("gng_crop_min_y", -2.0);
	gng_crop_max_y = this->declare_parameter<double>("gng_crop_max_y", 2.0);
	gng_crop_min_z = this->declare_parameter<double>("gng_crop_min_z", -999.9);
	gng_crop_max_z = this->declare_parameter<double>("gng_crop_max_z", 2.0);
	gng_crop_distance = this->declare_parameter<double>("gng_crop_distance", 0.0);

	use_ignore_region = this->declare_parameter<bool>("use_ignore_region", false);
	ignore_region_min_x = this->declare_parameter<double>("ignore_region_min_x", 0.0);
	ignore_region_max_x = this->declare_parameter<double>("ignore_region_max_x", 0.0);
	ignore_region_min_y = this->declare_parameter<double>("ignore_region_min_y", 0.0);
	ignore_region_max_y = this->declare_parameter<double>("ignore_region_max_y", 0.0);
	ignore_region_min_z = this->declare_parameter<double>("ignore_region_min_z", 0.0);
	ignore_region_max_z = this->declare_parameter<double>("ignore_region_max_z", 0.0);

	std::cout << gng_main_frame_id << std::endl;
	auto sensor_qos = rclcpp::SensorDataQoS();
	sensor_qos.best_effort();
	sensor_qos.durability_volatile();
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      //"livox/lidar", 10, std::bind(&pointcloud_subscriber::topic_callback, this, _1));
	  pointCloudTopic, sensor_qos, std::bind(&pointcloud_subscriber::topic_callback, this, _1));

	  
	  loc_data_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/loc_data", sensor_qos, std::bind(&pointcloud_subscriber::loc_data_topic_callback, this, _1));


	//camera_Angle = this->declare_parameter<double>("camera_Angle", RGBD_CAMERA_ANGLE);
    // node = rclcpp::Node::make_shared("Node_Pos_to_pointcloud2");
    // edge_node = rclcpp::Node::make_shared("Edge_marker");
    // cmd_vel_node = rclcpp::Node::make_shared("trobot_controller");
	// untra_node = rclcpp::Node::make_shared("untraversability_nodes");

    g_pub_gng_node = this->create_publisher<sensor_msgs::msg::PointCloud2>("gng_node", sensor_qos);
	pub_gng_edge = this->create_publisher<visualization_msgs::msg::Marker>("gng_edge", sensor_qos);
    // pub_cmd_vel = cmd_vel_node->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
	pub_untra_node = this->create_publisher<sensor_msgs::msg::PointCloud2>("untra_node", sensor_qos);

	// use_tf
	tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

	init_genrand((unsigned long)time(NULL));

  }

  std::string pointCloudTopic;
//   std::string gngViewFrame;
  

private:
  bool isPointInIgnoreRegion(double x, double y, double z) const{
    if (!use_ignore_region) return false;
    
    return (x >= ignore_region_min_x && x <= ignore_region_max_x &&
            y >= ignore_region_min_y && y <= ignore_region_max_y &&
            z >= ignore_region_min_z && z <= ignore_region_max_z);
  }
  void loc_data_topic_callback(const std_msgs::msg::Float64MultiArray msg) const
  {
	current_loc_data[0] = msg.data[0];
	current_loc_data[1] = msg.data[1];
	current_loc_data[2] = msg.data[2];
	printf("LOC Data %f, %f, %f\n", current_loc_data[0], current_loc_data[1], current_loc_data[2]);
  }
  void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
  {
    // use_tf
	// gng_main用(odom→LiDAR座標)
	geometry_msgs::msg::TransformStamped gng_main_tf_stamp;
	std::string fromFrameRel= gng_main_frame_id;
	std::string toFrameRel= msg->header.frame_id.c_str();
	try {
		gng_main_tf_stamp = tf_buffer_->lookupTransform(
		fromFrameRel, toFrameRel,
		tf2::TimePointZero);	
	} catch (const tf2::TransformException & ex) {
		RCLCPP_INFO(
		this->get_logger(), "Could not transform %s to %s: %s",
		fromFrameRel.c_str(), toFrameRel.c_str(), ex.what());
		return;
	}
	tf2::Quaternion gng_main_quat(
		gng_main_tf_stamp.transform.rotation.x,
		gng_main_tf_stamp.transform.rotation.y,
		gng_main_tf_stamp.transform.rotation.z,
		gng_main_tf_stamp.transform.rotation.w
	);
	tf2::Matrix3x3 gng_main_mat33(gng_main_quat);
	// デバッグ用(mat33とtrobot_param.rotの3×3の回転行列の値比較)
	// 	std::cout<< "mat33 ="<<std::endl;
	// 	std::cout<< mat33[0][0]<<", "<< mat33[0][1]<<", "<< mat33[0][2]<<", "<<std::endl;
	// 	std::cout<< mat33[1][0]<<", "<< mat33[1][1]<<", "<< mat33[1][2]<<", "<<std::endl;
	// 	std::cout<< mat33[2][0]<<", "<< mat33[2][1]<<", "<< mat33[2][2]<<", "<<std::endl;
	// 	std::cout<< "trobot_param.rot ="<<std::endl;
	// 	std::cout<< trobot_param.rot[0][0]<<", "<< trobot_param.rot[0][1]<<", "<< trobot_param.rot[0][2]<<", "<<std::endl;
	// 	std::cout<< trobot_param.rot[1][0]<<", "<< trobot_param.rot[1][1]<<", "<< trobot_param.rot[1][2]<<", "<<std::endl;
	// 	std::cout<< trobot_param.rot[2][0]<<", "<< trobot_param.rot[2][1]<<", "<< trobot_param.rot[2][2]<<", "<<std::endl;
	// デバッグ用(tfでのx,y,z,roll,pitch,yaw出力)
	// 	double roll, pitch, yaw;
	// 	mat33.getRPY(roll,pitch,yaw);
	// 	std::cout<<fromFrameRel<<" → "<<toFrameRel<<std::endl;
	// 	std::cout<<"x    :"<<t.transform.translation.x<<std::endl;
	// 	std::cout<<"y    :"<<t.transform.translation.y<<std::endl;
	// 	std::cout<<"z    :"<<t.transform.translation.z<<std::endl;
	// 	std::cout<<"roll :"<<roll<<std::endl;
	// 	std::cout<<"pitch:"<<pitch<<std::endl;
	// 	std::cout<<"yaw  :"<<yaw<<std::endl;
	
	// use_tf
	// gng_crop用(base_footprint→LiDAR座標)
	geometry_msgs::msg::TransformStamped gng_crop_tf_stamp;
	fromFrameRel= gng_crop_frame_id;
	toFrameRel= msg->header.frame_id.c_str();
	try {
		gng_crop_tf_stamp = tf_buffer_->lookupTransform(
		fromFrameRel, toFrameRel,
		tf2::TimePointZero);	
	} catch (const tf2::TransformException & ex) {
		RCLCPP_INFO(
		this->get_logger(), "Could not transform %s to %s: %s",
		fromFrameRel.c_str(), toFrameRel.c_str(), ex.what());
		return;
	}
	tf2::Quaternion gng_crop_quat(
		gng_crop_tf_stamp.transform.rotation.x,
		gng_crop_tf_stamp.transform.rotation.y,
		gng_crop_tf_stamp.transform.rotation.z,
		gng_crop_tf_stamp.transform.rotation.w
	);
	tf2::Matrix3x3 gng_crop_mat33(gng_crop_quat);

	
    static int init_flag = 1;
	static struct gng* net = NULL;
    static double **pointcloud_data;
    int data_size = 0;
    	
    if(init_flag == 1){
        net = init_gng(); //GNGの初期化
        net->weight[0] = 1;//x
        net->weight[1] = 1;//y
        net->weight[2] = 1;//z
        net->weight[3] = 0;//r
        net->weight[4] = 0;//g
        net->weight[5] = 0;//b
        init_flag = 0;
        pointcloud_data = malloc2d_double(640 * 576, LDIM);
        std::cout << "Node number = " << net->node_n << std::endl;
    }

    sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*msg, "z");
	sensor_msgs::PointCloud2Iterator<float> iter_i(*msg, "intensity"); 
    
	const double c_r = cos(trobot_param.angle_y);
    const double s_r = sin(trobot_param.angle_y);
	double d_coeff = 0.0;
	if(net->node_n > 100){
		for(int i=0;i<net->cluster_ct[2];i++){
			if(net->flat_property[i] == 1){
				printf("Detect Flat:%f, %f, %f\n", net->cluster_cog[2][i][6], net->cluster_cog[2][i][7], net->cluster_cog[2][i][8]);
				d_coeff = net->cluster_cog[2][i][6]*net->cluster_cog[2][i][0] + net->cluster_cog[2][i][7]*net->cluster_cog[2][i][1] + net->cluster_cog[2][i][8]*net->cluster_cog[2][i][2];
				printf("Coefficient = %f\n",d_coeff);
			}
		}
	}

	// static int fcount = 1;
	// char fname[256];
	// sprintf(fname, "%d.dat", fcount);
	// printf("%s\n",fname);
	// FILE *fp=fopen(fname, "w");
    for(unsigned int i = 0; iter_x != iter_x.end(); ++i,++iter_x, ++iter_y, ++iter_z, ++iter_i)
    {
        if(*iter_x != 0 || *iter_y != 0 || *iter_z != 0){
			//if(*iter_x > 7.0 /*|| *iter_z > 1.0*/) continue;
			//point.xyz[0] = -point_cloud_image_data[3 * i + 0];
			//point.xyz[1] = -(point_cloud_image_data[3 * i + 1]* c_r + point_cloud_image_data[3 * i + 2]*s_r + crawler.camera_height;
			//point.xyz[2] = -point_cloud_image_data[3 * i + 1] * s_r + point_cloud_image_data[3 * i + 2]*c_r;
			float x, y, z;
			x = *iter_x;
			y = *iter_y;
			z = *iter_z;

			// use_tf
			// オリジナル
            // pointcloud_data[data_size][0] = x*trobot_param.rot[0][0]+y*trobot_param.rot[0][1]+z*trobot_param.rot[0][2];
            // pointcloud_data[data_size][1] = x*trobot_param.rot[1][0]+y*trobot_param.rot[1][1]+z*trobot_param.rot[1][2];
            // pointcloud_data[data_size][2] = x*trobot_param.rot[2][0]+y*trobot_param.rot[2][1]+z*trobot_param.rot[2][2]+trobot_param.camera_height;

			// // pointcloud_data[data_size][0] = x*c_r+z*s_r;
            // // pointcloud_data[data_size][1] = y;
            // // pointcloud_data[data_size][2] = -x*s_r+z*c_r+trobot_param.camera_height;

            // pointcloud_data[data_size][3] = (double)*iter_i/255.0;//*invSqrt(x*x+y*y+z*z);
			// // if(fabs(pointcloud_data[data_size][0]) > 7.0) continue;
			// if(pointcloud_data[data_size][2] < 2.0) continue;
			// if(fabs(pointcloud_data[data_size][1]) > 2.0) continue;
			// // fprintf(fp, "%f\t%f\t%f\t%f\n", pointcloud_data[data_size][0], pointcloud_data[data_size][1], pointcloud_data[data_size][2], pointcloud_data[data_size][3]);
            // // pointcloud_data[data_size][4] = (double)*iter_i/255.0;
            // // pointcloud_data[data_size][5] = (double)*iter_i/255.0;

			// 距離を計算
            double distance = sqrt(x*x + y*y + z*z);

            // 距離がしきい値以下の場合はスキップ
            if(distance <= gng_crop_distance) continue;

			// use_tf
			// 有効点群の範囲制限(gng_crop_frame_id座標系から見たとき)
			double crop_frame_pcd_x = x*gng_crop_mat33[0][0]+y*gng_crop_mat33[0][1]+z*gng_crop_mat33[0][2]+gng_crop_tf_stamp.transform.translation.x;
			double crop_frame_pcd_y = x*gng_crop_mat33[1][0]+y*gng_crop_mat33[1][1]+z*gng_crop_mat33[1][2]+gng_crop_tf_stamp.transform.translation.y;
			double crop_frame_pcd_z = x*gng_crop_mat33[2][0]+y*gng_crop_mat33[2][1]+z*gng_crop_mat33[2][2]+gng_crop_tf_stamp.transform.translation.z;
			if(crop_frame_pcd_x < gng_crop_min_x) continue;
			if(crop_frame_pcd_x > gng_crop_max_x) continue;
			if(crop_frame_pcd_y < gng_crop_min_y) continue;
			if(crop_frame_pcd_y > gng_crop_max_y) continue;
			if(crop_frame_pcd_z < gng_crop_min_z) continue;
			if(crop_frame_pcd_z > gng_crop_max_z) continue;
			// 除外領域のチェック
            if(isPointInIgnoreRegion(crop_frame_pcd_x, crop_frame_pcd_y, crop_frame_pcd_z)) continue;

			// 有効点群の座標変換(gng_main_frame_id座標系から見たとき)
			pointcloud_data[data_size][0] = x*gng_main_mat33[0][0]+y*gng_main_mat33[0][1]+z*gng_main_mat33[0][2]+gng_main_tf_stamp.transform.translation.x;
            pointcloud_data[data_size][1] = x*gng_main_mat33[1][0]+y*gng_main_mat33[1][1]+z*gng_main_mat33[1][2]+gng_main_tf_stamp.transform.translation.y;
            pointcloud_data[data_size][2] = x*gng_main_mat33[2][0]+y*gng_main_mat33[2][1]+z*gng_main_mat33[2][2]+gng_main_tf_stamp.transform.translation.z;
			
            data_size++;
        }
    }
	// fclose(fp);

	//GNG知覚システム
    std::chrono::system_clock::time_point gng_proc_start, gng_proc_end;
	gng_proc_start = std::chrono::system_clock::now();
	
    if(data_size != 0){
		// if(net->node_n > 100){
			double rot_ang = 0.0;//prev_yaw_angle - current_yaw_angle;
			double trans_data[3];

			for(int i=0;i<3;i++){
				trans_data[i] = prev_loc_data[i] - current_loc_data[i];
			}

			// double ca = cos(rot_ang);
			// double sa = sin(rot_ang);
			double ca = cos(-trans_data[2]/180.0*M_PI);
			double sa = sin(-trans_data[2]/180.0*M_PI);
			for(int i=0;i<net->node_n;i++){
				// double dx = net->node[i][0];
				// double dy = net->node[i][1];
				// double dz = net->node[i][2];
				double dx = net->node[i][0];
				double dy = net->node[i][1];
				double dz = net->node[i][2];
				

				net->node[i][0] = ca*dx - sa*dy - trans_data[1]/1000.0;
				net->node[i][1] = sa*dx + ca*dy - trans_data[0]/1000.0;
				net->node[i][2] = dz;
			}
		// }

		//prev_yaw_angle = current_yaw_angle;
		for(int i=0;i<3;i++){
			prev_loc_data[i] = current_loc_data[i];
		}

        for (int i = 0; i < 10; i++){
            gng_main(net, pointcloud_data, data_size);
        }
        gng_clustering(net, 5);
		gng_clustering(net, 2);
		gng_clustering(net, 4);
		calc_cluster_features(net, 4);
    }

	gng_proc_end = std::chrono::system_clock::now();
	double gng_proc_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(gng_proc_end - gng_proc_start).count() / 1000.0);
	// FILE *fp = fopen("/home/toda/tmpdata/time_proc.txt", "a");
	// fprintf(fp, "%d\t%f\n",net->node_n, gng_proc_time);
	// fclose(fp);
	
	std::cout << "Processing time = " << gng_proc_time << " [ms]" << std::endl;
    std::cout << "Data size =" << data_size << ", Node number = " << net->node_n << std::endl;

	

	//Trobot制御コマンド
    //pub_cmd_vel->publish(cmd_vel_msg);

	//rvizへの表示
	std::shared_ptr<sensor_msgs::msg::PointCloud2> node2_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();

	//走行不可能ノードのパブリッシュ
	std::shared_ptr<sensor_msgs::msg::PointCloud2> untra_node_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
	untra_node_msg->header.frame_id="base_link";
	 //node2_msg->header.frame_id="base_link";
    untra_node_msg->height=1;
	untra_node_msg->is_dense=false;
	untra_node_msg->is_bigendian=false;
	untra_node_msg->width=net->node_n;
	untra_node_msg->fields.clear();
	untra_node_msg->fields.reserve(1);
	untra_node_msg->data.resize(net->node_n);

	sensor_msgs::PointCloud2Modifier untra_pcd_modifier(*untra_node_msg);
    //pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
    //untra_pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
	untra_pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

	sensor_msgs::PointCloud2Iterator<float> iter_untra_pub_x(*untra_node_msg, "x");
	sensor_msgs::PointCloud2Iterator<float> iter_untra_pub_y(*untra_node_msg, "y");
	sensor_msgs::PointCloud2Iterator<float> iter_untra_pub_z(*untra_node_msg, "z");
	sensor_msgs::PointCloud2Iterator<int8_t> iter_untra_pub_intesity(*untra_node_msg, "r");
	
	
	node2_msg->header.frame_id=gng_main_frame_id;
	 //node2_msg->header.frame_id="base_link";
    node2_msg->height=1;
	node2_msg->is_dense=false;
	node2_msg->is_bigendian=false;
	node2_msg->width=net->node_n;
	node2_msg->fields.clear();
	node2_msg->fields.reserve(1);
	node2_msg->data.resize(net->node_n);

	sensor_msgs::PointCloud2Modifier pcd_modifier(*node2_msg);
    //pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
    pcd_modifier.setPointCloud2FieldsByString(2, "xyz","rgb");

	sensor_msgs::PointCloud2Iterator<float> iter_pub_x(*node2_msg, "x");
	sensor_msgs::PointCloud2Iterator<float> iter_pub_y(*node2_msg, "y");
	sensor_msgs::PointCloud2Iterator<float> iter_pub_z(*node2_msg, "z");
    sensor_msgs::PointCloud2Iterator<int8_t> iter_pub_r(*node2_msg, "r");
    sensor_msgs::PointCloud2Iterator<int8_t> iter_pub_g(*node2_msg, "g");
    sensor_msgs::PointCloud2Iterator<int8_t> iter_pub_b(*node2_msg, "b");

	auto edge_lines = visualization_msgs::msg::Marker();
	edge_lines.header.stamp = this->now();
    edge_lines.header.frame_id = gng_main_frame_id;
	//edge_lines.header.frame_id = "base_link";
    edge_lines.ns = "edge_lines";
    edge_lines.id = 0;
    edge_lines.type = visualization_msgs::msg::Marker::LINE_LIST;
    edge_lines.action =  visualization_msgs::msg::Marker::ADD;
	//edge_lines.pose.orientation.w = 1.0;
	edge_lines.scale.x = 0.005;
	// edge_lines.scale.x = 0.01;
    
    edge_lines.color.a = 1.0;
    edge_lines.color.r = 0.0;
    edge_lines.color.g = 1.0;
    edge_lines.color.b = 0.0;

	static int colorlist[100][3];
	static int color_init = 1;
	if(color_init == 1){
		for(int i=0;i<100;i++){
			colorlist[i][0] = (int)(rnd()*255.0);
			colorlist[i][1] = (int)(rnd()*255.0);
			colorlist[i][2] = (int)(rnd()*255.0);
		}
		color_init = 0;
	}

	// for(int i=0;i<net->cluster_ct[2];i++){
	// 	if(net->cluster_num[2][i] < 50) continue;
	// 	printf("Detect Flat:%f, %f, %f\n", net->cluster_cog[2][i][4], net->cluster_cog[2][i][5], net->cluster_cog[2][i][6]);
	// }

	for(unsigned int i = 0; i < net->node_n; ++i,++iter_pub_x, ++iter_pub_y, ++iter_pub_z,
											++iter_pub_r, ++iter_pub_g, ++iter_pub_b,
											++iter_untra_pub_x, ++iter_untra_pub_y, ++iter_untra_pub_z, ++iter_untra_pub_intesity){
            
            
            if (net->traversability_property[i] == 1) {
				if (net->contour[i] == 1) {
					// 境界ノードのように見えるが、緑を内包してはいない
					// (であるならば、緑にしてしまい、半径で障害点除去したほうが良い？)
					//continue;
                    *iter_pub_r = 0;
                    *iter_pub_g = 255;
                    *iter_pub_b = 0;

				}
				else {
					*iter_pub_r = 0;
                    *iter_pub_g = 255;
                    *iter_pub_b = 0;
                }
				// 黒色は何を表すのか戸田さんに聞く(走行可能か？)
				// この黒色は走行可能であったが、
				*iter_untra_pub_x = 0.0;
				*iter_untra_pub_y = 0.0;
				*iter_untra_pub_z = 0.0;
				*iter_untra_pub_intesity = 255;

            }else{
				// 走行不能クラスタ(平地に出るケースがある)(学習不足？)
				//continue;
				double rate = net->node[i][7]/0.1;
				if(rate > 1.0) rate = 1.0;
                // *iter_pub_r = (int)(255.0*rate);
                *iter_pub_r = 255;
                *iter_pub_g = 0;
                *iter_pub_b = 0;

				*iter_untra_pub_x = (float)net->node[i][0];
				*iter_untra_pub_y = (float)net->node[i][1];
				*iter_untra_pub_z = (float)net->node[i][2];
				*iter_untra_pub_intesity = net->cluster2[5][i];
            }

			// *iter_pub_r = (int)(net->node[i][3]*255.0);
            // *iter_pub_g = 0;//(int)(net->node[i][4]*255.0);
            // *iter_pub_b = 0;//(int)(net->node[i][5]*255.0);

			// *iter_pub_r = (int)(net->node[i][8]*255.0);
            // *iter_pub_g = (int)(net->node[i][9]*255.0);
            // *iter_pub_b = 0;//(int)(net->node[i][10]*255.0);
			
			int c = net->cluster2[5][i];
			// *iter_pub_r = (int)(net->cluster_features[4][c][0]*255.0);
            // *iter_pub_g = (int)(net->cluster_features[4][c][1]*255.0);
            // *iter_pub_b = (int)(net->cluster_features[4][c][2]*255.0);
			
			// *iter_pub_r = colorlist[c][0];
            // *iter_pub_g = colorlist[c][1];
            // *iter_pub_b = colorlist[c][2];//(int)(net->node[i][10]*255.0);
			float x, y, z;
			x = (float)net->node[i][0];
			y = (float)net->node[i][1];
			z = (float)net->node[i][2];//-trobot_param.camera_height;
			// *iter_pub_x = x*c_r-z*s_r;
            // *iter_pub_y = y;
            // *iter_pub_z = x*s_r+z*c_r;

			*iter_pub_x = x;//x*trobot_param.rot_t[0][0]+y*trobot_param.rot_t[0][1]+z*trobot_param.rot_t[0][2];
            *iter_pub_y = y;//x*trobot_param.rot_t[1][0]+y*trobot_param.rot_t[1][1]+z*trobot_param.rot_t[1][2];
            *iter_pub_z = z;//x*trobot_param.rot_t[2][0]+y*trobot_param.rot_t[2][1]+z*trobot_param.rot_t[2][2];

			//Calculate 
			// x = (float)net->node[i][0];
			// y = (float)net->node[i][1];
			// z = (float)net->node[i][2]-trobot_param.camera_height;
			// x = x - net->node[i][7]*net->node[i][4];
			// y = y - net->node[i][7]*net->node[i][5];
			// z = z - net->node[i][7]*net->node[i][6];

			// geometry_msgs::msg::Point p;
			// p.x = x*c_r-z*s_r;
			// p.y = y;
			// p.z = x*s_r+z*c_r;
			// edge_lines.points.push_back(p);

			// x = (float)net->node[i][0];
			// y = (float)net->node[i][1];
			// z = (float)net->node[i][2]-trobot_param.camera_height;
			// x = x + net->node[i][7]*net->node[i][4];
			// y = y + net->node[i][7]*net->node[i][5];
			// z = z + net->node[i][7]*net->node[i][6];
			// p.x = x*c_r-z*s_r;
			// p.y = y;
			// p.z = x*s_r+z*c_r;
			// edge_lines.points.push_back(p);

			if(i != net->node_n -1){
				for(unsigned int j=i+1;j<net->node_n;j++){
					if(net->pedge[i][j] == 1){
						geometry_msgs::msg::Point p;
						p.x = *iter_pub_x;
						p.y = *iter_pub_y;
						p.z = *iter_pub_z;
						edge_lines.points.push_back(p);
						x = (float)net->node[j][0];
						y = (float)net->node[j][1];
						z = (float)net->node[j][2];//-trobot_param.camera_height;
						p.x = x;//x*trobot_param.rot_t[0][0]+y*trobot_param.rot_t[0][1]+z*trobot_param.rot_t[0][2];
						p.y = y;//x*trobot_param.rot_t[1][0]+y*trobot_param.rot_t[1][1]+z*trobot_param.rot_t[1][2];
						p.z = z;//x*trobot_param.rot_t[2][0]+y*trobot_param.rot_t[2][1]+z*trobot_param.rot_t[2][2];
						edge_lines.points.push_back(p);
					}
				}
			}
    }

    g_pub_gng_node->publish(*node2_msg);
	pub_gng_edge->publish(edge_lines);
	pub_untra_node->publish(*untra_node_msg);

  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr loc_data_subscription_;
  robot_parameter trobot_param;
  double camera_Angle;
  
  // use_tf
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

};

int main(int argc, char * argv[])
{

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pointcloud_subscriber>());
  rclcpp::shutdown();
  return 0;
}