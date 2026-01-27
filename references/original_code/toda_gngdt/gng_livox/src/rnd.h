/*
 *  rnd.h
 *  Claster
 *
 *  Created by Naoyuki Kubota on 12/05/18.
 *  Copyright 2012 首都大学東京. All rights reserved.
 *
 */
#ifndef RND_H
#define RND_H
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "mt.h"

// uniform random number
double rnd()
{
	// return((double)(rand()%30001)/30000.0);
	// return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
	return genrand_real3();

}
// normal random number
double rndn()
{
	return (rnd()+rnd()+rnd()+rnd()+rnd()+rnd()+
			rnd()+rnd()+rnd()+rnd()+rnd()+rnd()-6.0);
}
#endif // RND_H