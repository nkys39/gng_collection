#pragma once
/*
 *  gng.h
 *  Cluster
 *
 *  Created by Yuichiro Toda on 12/05/18.
 *  Copyright 2012 首都大学東京. All rights reserved.
 *
 */

#define GNGN 1000	//最大ノード数def2000
 //#define ND 428*240		//データ数
#define ND 300000		//データ数
#define DIM 11		//ベクトルの次元数
#define LDIM 4		//学習を行うベクトルの次元数
#define NOP 6 //構築するクラスタの種類の数
#define MAXANGLE 20
#define THV 0.001*0.001  //def 20*20

struct gng {

	double** node;		//ノード用配列
	int** edge;		//エッジ用配列（位置情報）
	int** cedge;		//エッジ用配列（色情報）
	int** nedge;		//エッジ用配列（法線情報）
	int** tedge;        //エッジ用配列（法線情報）
	int** pedge;        //エッジ用配列（走行可能判定情報）
	int** degedge;
	int** dedge;
	//int** edge_s1;     //特徴量エッジ
	//int** edge_s2;
	//int** edge_s3;
	int** age;		//エッジの年齢用配列
	int* edge_ct;        //エッジの個数カウント用
	int epoch2;
	int node_n;					//現在のノード数
	int cluster[NOP][GNGN][GNGN];
	int cluster2[NOP][GNGN];
	int cluster_num[NOP][GNGN];
	double cluster_cog[NOP][GNGN][DIM];
	int cluster_ct[NOP];
	int pre_cluster_ct[NOP];
	double* gng_err;		//積算誤差用配列
	double* gng_u;		//utility valiables
	double cluster_features[NOP][GNGN][DIM];

	int flat_property[GNGN];
	int through_property[GNGN];
	int cur_property[GNGN];
	int dimension_property[GNGN][3];
	int trace_property[GNGN];
	int traversability_property[GNGN];
	int traversability_flag[GNGN]; //クラスタ判定用

	double lean[GNGN]; //cost calculation
	double degree[GNGN];
	double cur[GNGN];

	//double disu[GNGN];
	double dise[GNGN];
	double max_dise;
	double disrate;
	double weight[DIM];

	double cthv;
	double nthv;

	double s1thv;
	double s2thv;
	double s3thv;

	int contour[GNGN];        //輪郭

	//near_n

	int nearnode[50];
	int near_n;
	int pre_near;

	int farnode[50];
	int far_n;
	int pre_far;


};

struct gng* init_gng();
int gng_main(struct gng* net, double** v, int dmax);
void gng_clustering(struct gng* net, int property);
float invSqrt(float x);

void calc_cluster_features(struct gng *net, int property);
