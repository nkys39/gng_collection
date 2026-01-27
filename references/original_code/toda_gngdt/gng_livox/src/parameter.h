#pragma once
#include <math.h>
//crawlerópÇÃÉpÉâÉÅÅ[É^

#define RGBD_CAMERA_ANGLE 0.0

class robot_parameter {
public:
	double wheel_dia = 0.15;
	double width = 0.4;//m
	double length = 0.5;
	double height = 0.7;
	//double camera_height = 595;//kt
	double camera_height = 0.0;//rs
	//double camera_depth = 150;//kt
	double camera_depth = 0.1;//rs
	//double angel_x = (M_PI * 45) / 180;//kinect
	double angle_x = RGBD_CAMERA_ANGLE/180.0*M_PI;//RS
	double angle_y = RGBD_CAMERA_ANGLE/180.0*M_PI;//RS
	double angle_z = RGBD_CAMERA_ANGLE/180.0*M_PI;//RS
	double rot[3][3];
	double rot_t[3][3];

	double z_limit = 0.5;//è·äQï®âÒîãóó£ //def500
	int passable_nodenum = 10;//è·äQï®ÉNÉâÉXÉ^ÇÃîªíËÇ»Ç«óp

	int leftsiderun_flag = 0;//0:off 1:on

	int kt_data = 0;
	int rs_data = 0;

};

class sensor_parameter {//rs
public:
	double space = 0.125;
	double space_start = 0.3;
	int space_num = 5;
	double z_limit = 2.5;

};
