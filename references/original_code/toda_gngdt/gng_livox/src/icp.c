//
//  icp.c
//  ICP
//
//  Created by Yuichiro Toda on 2015/11/30.
//
//

#include "icp.h"
#include "gng.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <f2c.h>
#include <lapacke.h>

double invSqrt_(float x) {
	float halfx = 0.5f * x;
	float y = x;
	long i = *(long*)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float*)&i;
	y = y * (1.5f - (halfx * y * y));
	y = y * (1.5f - (halfx * y * y));
	return (double)y;
}

//重心の計算
int calc_medianpoint(int numberOfPairs, double position[][3], double pairPosition[][3],  double *mup, double *muy, int *checkflag, int sflag)
{
    int i,ct=0;
    
    for(i=0;i<3;i++){
        mup[i]=0.0;
        muy[i]=0.0;
    }
    
    for(i=0;i<numberOfPairs;i++){
        if(checkflag[i] != 1){
            if(sflag == 0 && checkflag[i] == 2)
                continue;
            mup[0]+=position[i][0];
            mup[1]+=position[i][1];
            mup[2]+=position[i][2];
            muy[0]+=pairPosition[i][0];
            muy[1]+=pairPosition[i][1];
            muy[2]+=pairPosition[i][2];
            ct++;
        }
    }
    
    for(i=0;i<3;i++){
        mup[i]/=(double)ct;
        muy[i]/=(double)ct;
    }
    return ct;
}

int calc_medianpoint_weight(int numberOfPairs, double position[][3], double pairPosition[][3],  double *mup, double *muy, double weight[], int *checkflag, int sflag)
{
    int i,ct=0;
	double sum_weight = 0.0;
    
    for(i=0;i<3;i++){
        mup[i]=0.0;
        muy[i]=0.0;
    }
    
    for(i=0;i<numberOfPairs;i++){
        if(checkflag[i] != 1){
            if(sflag == 0 && checkflag[i] == 2)
                continue;
            mup[0]+=weight[i]*position[i][0];
            mup[1]+=weight[i]*position[i][1];
            mup[2]+=weight[i]*position[i][2];
            muy[0]+=weight[i]*pairPosition[i][0];
            muy[1]+=weight[i]*pairPosition[i][1];
            muy[2]+=weight[i]*pairPosition[i][2];
			sum_weight += weight[i];
            ct++;
        }
    }
    
    for(i=0;i<3;i++){
        mup[i]/=(double)sum_weight;
        muy[i]/=(double)sum_weight;
    }
    return ct;
}

//Unit quaternionsを用いたオリエンテーションの計算
//Horn, B. K. (1987). Closed-form solution of absolute orientation using unit quaternions. JOSA A, 4(4), 629-642.
void calc_eigenvalue(int numberOfPairs, double position[][3], double pairPosition[][3], double *mup, double *muy, double *q ,int *checkflag)
{
    int i,j;
    double SIGP[3][3],Q[4][4];
    char jobvl = 'N';
    char jobvr ='V';
    static integer a=4,lda=4,ldvl=1,ldvr=4,lwork=4*4,info;
    double Q2[4*4], wr[4], wi[4], vl[1*4], vr[4*4], work[4*4];
    double max;
    int maxn=0;
    
    memset(SIGP, 0, sizeof(SIGP));
    
    for(i=0;i<numberOfPairs;i++){
        if(checkflag[i] == 0){
            SIGP[0][0]+=(position[i][0]-mup[0])*(pairPosition[i][0]-muy[0]), SIGP[0][1]+=(position[i][0]-mup[0])*(pairPosition[i][1]-muy[1]), SIGP[0][2]+=(position[i][0]-mup[0])*(pairPosition[i][2]-muy[2]);
            SIGP[1][0]+=(position[i][1]-mup[1])*(pairPosition[i][0]-muy[0]), SIGP[1][1]+=(position[i][1]-mup[1])*(pairPosition[i][1]-muy[1]), SIGP[1][2]+=(position[i][1]-mup[1])*(pairPosition[i][2]-muy[2]);
            SIGP[2][0]+=(position[i][2]-mup[2])*(pairPosition[i][0]-muy[0]), SIGP[2][1]+=(position[i][2]-mup[2])*(pairPosition[i][1]-muy[1]), SIGP[2][2]+=(position[i][2]-mup[2])*(pairPosition[i][2]-muy[2]);
        }
    }
    
    Q[0][0] = SIGP[0][0]+SIGP[1][1]+SIGP[2][2],    Q[0][1] = SIGP[1][2]-SIGP[2][1],            Q[0][2] = SIGP[2][0]-SIGP[0][2],             Q[0][3] = SIGP[0][1]-SIGP[1][0];
    Q[1][0] = SIGP[1][2]-SIGP[2][1],               Q[1][1] = SIGP[0][0]-SIGP[1][1]-SIGP[2][2], Q[1][2] = SIGP[0][1]+SIGP[1][0],             Q[1][3] = SIGP[0][2]+SIGP[2][0];
    Q[2][0] = SIGP[2][0]-SIGP[0][2],               Q[2][1] = SIGP[0][1]+SIGP[1][0],            Q[2][2] = -SIGP[0][0]+SIGP[1][1]-SIGP[2][2], Q[2][3] = SIGP[1][2]+SIGP[2][1];
    Q[3][0] = SIGP[0][1]-SIGP[1][0],               Q[3][1] = SIGP[0][2]+SIGP[2][0],            Q[3][2] = SIGP[2][1]+SIGP[1][2],             Q[3][3] = -SIGP[0][0]-SIGP[1][1]+SIGP[2][2];
    
    for(i=0; i < 4; i++){
        for(j=0; j < 4; j++){
            Q2[i*4+j] = Q[j][i];
        }
    }
    
    dgeev_( &jobvl, &jobvr, &a, Q2, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info, 1, 1);
    
    maxn=0;
    max=wr[0];
    for(i=1;i<4;++i){
        if(max < wr[i]){
            max = wr[i];
            maxn = i;
        }
    }
    
    for(i=0;i<4;++i) q[i] = vr[i+maxn*4];
    
}

//unit quaternionsから回転行列と平行移動ベクトルの算出
void calc_transformation(double R[3][3], double *q2, double *mup, double *muy, double *q)
{
    double a[3],b[3],c[3];
    a[0] = q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], b[0] = 2.0*(q[1]*q[2]-q[0]*q[3]),               c[0] = 2.0*(q[1]*q[3]+q[0]*q[2]);
    a[1] = 2.0*(q[1]*q[2]+q[0]*q[3]),               b[1] = q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3], c[1] = 2.0*(q[2]*q[3]-q[0]*q[1]);
    a[2] = 2.0*(q[1]*q[3]-q[0]*q[2]),               b[2] = 2.0*(q[2]*q[3]+q[0]*q[1]),               c[2] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
    
    double dis1 = (double)invSqrt_((double)(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]));
    double tmp = 0.0;
    for(int j=0;j<3;j++){
        a[j] = a[j]*dis1;
        tmp += b[j]*a[j];
    }
    
    for(int j=0;j<3;j++){
        b[j] -= a[j]*tmp;
    }
    
    dis1 = (double)invSqrt_((double)(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]));
    for(int j=0;j<3;j++){
        b[j] = b[j]*dis1;
    }
    
    c[0] = a[1]*b[2] - b[1]*a[2];
    c[1] = -a[0]*b[2] + b[0]*a[2];
    c[2] = a[0]*b[1] - b[0]*a[1];
    dis1 = (double)invSqrt_((double)(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]));
    for(int j=0;j<3;j++){
        c[j] = c[j]*dis1;
    }
    
//    printf("%f\n",dis1);
//    R[0][0] = q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], R[0][1] = 2.0*(q[1]*q[2]-q[0]*q[3]),               R[0][2] = 2.0*(q[1]*q[3]+q[0]*q[2]);
//    R[1][0] = 2.0*(q[1]*q[2]+q[0]*q[3]),               R[1][1] = q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3], R[1][2] = 2.0*(q[2]*q[3]-q[0]*q[1]);
//    R[2][0] = 2.0*(q[1]*q[3]-q[0]*q[2]),               R[2][1] = 2.0*(q[2]*q[3]+q[0]*q[1]),               R[2][2] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];

    R[0][0] = a[0], R[0][1] = b[0], R[0][2] = c[0];
    R[1][0] = a[1], R[1][1] = b[1], R[1][2] = c[1];
    R[2][0] = a[2], R[2][1] = b[2], R[2][2] = c[2];
    
    
    
    q2[0] = muy[0] - R[0][0]*mup[0] - R[0][1]*mup[1] - R[0][2]*mup[2];
    q2[1] = muy[1] - R[1][0]*mup[0] - R[1][1]*mup[1] - R[1][2]*mup[2];
    q2[2] = muy[2] - R[2][0]*mup[0] - R[2][1]*mup[1] - R[2][2]*mup[2];
    
//    for(int i=0;i<3;i++){
//        for(int j=0;j<3;j++){
//            printf("%lf\t",R[i][j]);
//        }
//        printf("\n");
//    }

//    for(int i=0;i<3;i++)
//        printf("%.4lf\n",q2[i]);
}

//点群同士の対応点探索
int search_matchingPoint(double **xyz1, double **xyz2, int mindex[], int cct[], int *checkflag)
{
    int i,j,k;
    double dis, mindis;
    int min_no;
    int ct = 0;
    
    for(i=0;i<cct[0];i++){
        min_no = 0;
        mindis = 0.0;
        for(k=0;k<3;k++){
            mindis += (xyz1[i][k] - xyz2[0][k])*(xyz1[i][k] - xyz2[0][k]);
        }
        
        for(j=1;j<cct[1];j++){
            dis = 0.0;
            for(k=0;k<3;k++){
                dis += (xyz1[i][k] - xyz2[j][k])*(xyz1[i][k] - xyz2[j][k]);
            }
            if(dis < mindis){
                mindis = dis;
                min_no = j;
            }
        }
        if(mindis > 0.2*0.2){
            checkflag[i] = 1;
            ct++;
        }else checkflag[i] = 0;
        //printf("%lf\n",sqrt(mindis));
        mindex[i] = min_no;
    }
    
    return ct;
}

void search_matchingPoint_weight(double **xyz1, double **xyz2, int mindex[], double weight[], int cct[])
{
    int i,j,k;
    double dis, mindis;
    int min_no;
	static double range = 5.0;
    
    for(i=0;i<cct[0];i++){
        min_no = 0;
        mindis = 0.0;
        for(k=0;k<3;k++){
            mindis += (xyz1[i][k] - xyz2[0][k])*(xyz1[i][k] - xyz2[0][k]);
        }
        
        for(j=1;j<cct[1];j++){
            dis = 0.0;
            for(k=0;k<3;k++){
                dis += (xyz1[i][k] - xyz2[j][k])*(xyz1[i][k] - xyz2[j][k]);
            }
            if(dis < mindis){
                mindis = dis;
                min_no = j;
            }
        }
        //printf("%lf\n",sqrt(mindis));
		weight[i] = exp(-mindis/(range*range));
		//printf("%lf\n",weight[i]);
        mindex[i] = min_no;
    }
    
	range *= 0.99;
	if(range < 0.1) range = 0.1;
	//printf("%lf\n",range);
}

void make_index(int mindex[], int cct[])
{
    int i;
    for(i=0;i<cct[0];i++){
        mindex[i] = i;
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


int icp(double **xyz1, double **xyz2, int cct[], double GQ[4], double GT[3], int initflag)
{
    int mct = cct[0];
    int i,j;
    double mup[3],muy[3];//,mudis;
    double q[4];
    double R[3][3],q2[3];
    int pairct=mct;
    static int checkflag[GNGN];
    static int mindex[GNGN];
    double dx,dy,dz;
    double position[GNGN][3],pairPosition[GNGN][3];
    double fmatchp[GNGN][3];
    double weight[GNGN];
    double dis_e=0, dis_e2=0;
    int rct;
    static double dis_e_old;
    int fflag = 0;
    
    if(initflag == 0){
        dis_e_old = 0.0;
    }
    
//    search_matchingPoint_weight(xyz1, xyz2, mindex, weight, cct);
    rct  = search_matchingPoint(xyz1, xyz2, mindex, cct, checkflag);
    
    for(i=0;i<cct[0];i++){
        for(j=0;j<3;j++){
            fmatchp[i][j] = xyz1[i][j];
            position[i][j] = xyz1[i][j];
            pairPosition[i][j] = xyz2[mindex[i]][j];
        }
    }
    
    //座標変換行列の計算
	pairct = calc_medianpoint(mct, position, pairPosition, mup, muy, checkflag, 0);
//    pairct = calc_medianpoint_weight(mct, position, pairPosition, mup, muy, weight, checkflag, 0);
    calc_eigenvalue(mct, position, pairPosition, mup, muy, q, checkflag);
    calc_transformation(R, q2, mup, muy, q);
    
    //座標変換
    dis_e=0;
    int uct = 0;
    for(i=0;i<mct;i++){
        dx = pairPosition[i][0] - R[0][0]*position[i][0] - R[0][1]*position[i][1] - R[0][2]*position[i][2] - q2[0];
        dy = pairPosition[i][1] - R[1][0]*position[i][0] - R[1][1]*position[i][1] - R[1][2]*position[i][2] - q2[1];
        dz = pairPosition[i][2] - R[2][0]*position[i][0] - R[2][1]*position[i][1] - R[2][2]*position[i][2] - q2[2];
        if(checkflag[i] == 0){
            // printf("%f\n",dx*dx+dy*dy+dz*dz);
            dis_e += 1.0/invSqrt_((float)(dx*dx+dy*dy+dz*dz));
            uct++;
        }
        dx = position[i][0];
        dy = position[i][1];
        dz = position[i][2];
        position[i][0] = R[0][0]*dx + R[0][1]*dy + R[0][2]*dz + q2[0];
        position[i][1] = R[1][0]*dx + R[1][1]*dy + R[1][2]*dz + q2[1];
        position[i][2] = R[2][0]*dx + R[2][1]*dy + R[2][2]*dz + q2[2];
    }
    
    dis_e /= (double)uct;
    dis_e2 /= (double)uct;
    
    dx = GT[0];
    dy = GT[1];
    dz = GT[2];

    GT[0] = R[0][0]*dx + R[0][1]*dy + R[0][2]*dz + q2[0];
    GT[1] = R[1][0]*dx + R[1][1]*dy + R[1][2]*dz + q2[1];
    GT[2] = R[2][0]*dx + R[2][1]*dy + R[2][2]*dz + q2[2];
    
    double tmpQ[4];
    for(i=0;i<4;i++) tmpQ[i] = GQ[i];
    GQ[0] = tmpQ[0]*q[0] - tmpQ[1]*q[1] - tmpQ[2]*q[2] - tmpQ[3]*q[3];
    GQ[1] = tmpQ[0]*q[1] + tmpQ[1]*q[0] + tmpQ[2]*q[3] - tmpQ[3]*q[2];
    GQ[2] = tmpQ[0]*q[2] - tmpQ[1]*q[3] + tmpQ[2]*q[0] + tmpQ[3]*q[1];
    GQ[3] = tmpQ[0]*q[3] + tmpQ[1]*q[2] - tmpQ[2]*q[1] + tmpQ[3]*q[0];
    double tmpinv = invSqrt_((float)(GQ[0]*GQ[0]+GQ[1]*GQ[1]+GQ[2]*GQ[2]+GQ[3]*GQ[3]));
    // printf("%f\n",tmpinv);
    for(i=0;i<4;i++) GQ[i] *= tmpinv;
//    dis_e=0;
    for(i=0;i<mct;i++){
        xyz1[i][0] = position[i][0];
        xyz1[i][1] = position[i][1];
        xyz1[i][2] = position[i][2];
    }
    
    //printf("Error:%f, Reject Rate%f\n", dis_e, (double)rct/(double)cct[0]);
    
    //if(fabs(dis_e-dis_e_old) < 0.005)
    //    fflag = 1;
    
    dis_e_old = dis_e;
    
    return fflag;
}
