/*
 *  malloc.h
 *  Cluster
 *
 *  Created by Naoyuki Kubota on 14/07/03.
 *  Copyright 2014 首都大学東京. All rights reserved.
 *
 */
#ifndef MALLOC_H
#define MALLOC_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//2次元配列の開放
void free2d_double(double ** a)
{
	free(a[0]);
	free(a);
}
//2次元配列の初期化
double **malloc2d_double(int x, int y)
{
	double **a;
	int i;
	a = (double **)malloc(sizeof(double *)*(x+1));
	a[0] = (double *)malloc(sizeof(double)*(y+1)*(x+1));
	for(i=1;i<(x+1);i++) a[i] = a[0] + i*(y+1);
	memset(a[0], 0,sizeof(*a[0]));
	return a;
}

//2次元配列の開放
void free2d_int(int ** a)
{
	free(a[0]);
	free(a);
}
//2次元配列の初期化
int **malloc2d_int(int x, int y)
{
	int **a;
	int i;
	a = (int **)malloc(sizeof(int *)*(x+1));
	a[0] = (int *)malloc(sizeof(int)*(y+1)*(x+1));
	for(i=1;i<(x+1);i++) a[i] = a[0] + i*(y+1);
	memset(a[0], 0,sizeof(*a[0]));
	return a;
}

#endif // MALLOC_H