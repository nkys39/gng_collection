/*
 *  pca.h
 *  Cluster
 *
 *  Created by Yuichiro Toda on 13/06/28.
 *  Copyright 2013 首都大学東京. All rights reserved.
 *
 */
#ifndef PCA_H
#define PCA_H

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <f2c.h>
#include <lapacke.h>

#define PCADIM 3
void calc_covMat(double v1[3], double mat[3][3])
{
    mat[0][0] = v1[0]*v1[0], mat[0][1] = v1[0]*v1[1], mat[0][2] = v1[0]*v1[2];
    mat[1][0] = v1[1]*v1[0], mat[1][1] = v1[1]*v1[1], mat[1][2] = v1[1]*v1[2];
    mat[2][0] = v1[2]*v1[0], mat[2][1] = v1[2]*v1[1], mat[2][2] = v1[2]*v1[2];
}

double pca(double s_ele[][3], double cog[3], int s_ct, double ev[6])
{
    int i,j,k;
    double v1[3];
    double mat1[3][3], mat2[3][3]={0};
    static integer a=3,lda=3,ldvl=1,ldvr=3,lwork=4*3,info;
    double Q2[3*3], wr[3], wi[3], vl[1*3], vr[3*3], work[4*3];
    char jobvl = 'N';
    char jobvr ='V';
    double min;
    double max;
    int minn = 0;
    int maxn = 0;

    
    for(i=0;i<s_ct;i++){
        v1[0] = s_ele[i][0] - cog[0];
        v1[1] = s_ele[i][1] - cog[1];
        v1[2] = s_ele[i][2] - cog[2];
        calc_covMat(v1, mat1);
        for(j=0;j<3;j++)
            for(k=0;k<3;k++)
                mat2[j][k] += mat1[j][k];
    }

    for(j=0;j<3;j++)
        for(k=0;k<3;k++)
            mat2[j][k] /= (double)(s_ct-1);
    
    for(i=0; i < 3; i++){
        for(j=0; j < 3; j++){
            Q2[i*3+j] = mat2[j][i];
            //printf("%d\t",(int)Q2[i*3+j]);
        }
        //printf("\n");
    }
    
    dgeev_( &jobvl, &jobvr, &a, Q2, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info, 1, 1);
    
    for (int i = 0; i < 3; i++)
        wr[i] = fabs(wr[i]);

    minn = 0;
    min = wr[0];
    maxn = 0;
    max = wr[0];
    //printf("%.3lf\t",wr[0]);
    for (i = 1; i < 3; ++i) {
        //printf("%lf\t",wr[i]);
        if (min > wr[i]) {
            min = wr[i];
            minn = i;
        }

        if (max < wr[i]) {
            max = wr[i];
            maxn = i;
        }
    }

    for (i = 3; i < 5; i++) {
        ev[i] = 0;
    }


    for (i = 0; i < 3; ++i) {
        ev[i] = vr[i + minn * 3];
        if (i != maxn && i != minn) {
            ev[4] = (wr[i] - min) / max;
            ev[5] = (max - wr[i]) / max;
        }
    }
    //ev[3] = min / max;
    ev[3] = sqrt(min)*2.0*0.865;

    
    return (wr[minn])/(wr[0]+wr[1]+wr[2]);
}

void calc_covMat_w(double v1[3], double mat[3][3], double weight)
{
    mat[0][0] = v1[0]*v1[0], mat[0][1] = v1[0]*v1[1], mat[0][2] = v1[0]*v1[2];
    mat[1][0] = v1[1]*v1[0], mat[1][1] = v1[1]*v1[1], mat[1][2] = v1[1]*v1[2];
    mat[2][0] = v1[2]*v1[0], mat[2][1] = v1[2]*v1[1], mat[2][2] = v1[2]*v1[2];
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++) mat[i][j] *= weight;
}
double pca_w(double s_ele[][3], double cog[3], int s_ct, double ev[3], double weight[])
{
    int i,j,k;
    double v1[3];
    double mat1[3][3], mat2[3][3]={0};
    static integer a=3,lda=3,ldvl=1,ldvr=3,lwork=4*3,info;
    double Q2[3*3], wr[3], wi[3], vl[1*3], vr[3*3], work[4*3];
    char jobvl = 'N';
    char jobvr ='V';
    double min;
    int minn=0;
    
    //共分散行列の計算
    for(i=0;i<s_ct;i++){
        v1[0] = s_ele[i][0] - cog[0];
        v1[1] = s_ele[i][1] - cog[1];
        v1[2] = s_ele[i][2] - cog[2];
        calc_covMat_w(v1, mat1, weight[i]);
        for(j=0;j<3;j++)
            for(k=0;k<3;k++)
                mat2[j][k] += mat1[j][k];
    }
    
    for(i=0; i < 3; i++){
        for(j=0; j < 3; j++){
            Q2[i*3+j] = mat2[j][i];
        }
    }
    
    //共分散行列から固有値を計算
    dgeev_( &jobvl, &jobvr, &a, Q2, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info, 1, 1);
    
    //法線ベクトルの決定（最小の固有値を持つベクトルの選択）
    minn=0;
    min=wr[0];
    //printf("%.3lf\t",wr[0]);
    for(i=1;i<3;++i){
        //printf("%lf\t",wr[i]);
        if(min > wr[i]){
            min = wr[i];
            minn = i;
        }
    }
    
    //法線ベクトルの代入
    for(i=0;i<3;++i) ev[i] = vr[i+minn*3];
    
    //曲率の計算
    return (wr[minn])/(wr[0]+wr[1]+wr[2]);
}
#endif // PCA_H