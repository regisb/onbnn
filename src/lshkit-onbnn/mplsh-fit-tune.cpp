/***************************************************************************
 *   Copyright (C) 2008 by Regis Behmo,,,   *
 *   regis.behmo@ecp.fr   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/



#include <cstdlib>
#include <gsl/gsl_multifit.h>
#include "lshkit.h"

using namespace std;
using namespace lshkit;
using namespace tr1;

#include "lshkit/tune.h"

static const int MIN_L = 1;
static const int MAX_L = 20;

static const int MIN_T = 1;
static const int MAX_T = 100;

static const int MIN_M = 1;
static const int MAX_M = 20;

static const double MIN_W = 0.01;
static const double MAX_W = 10;
static const double NUM_W = 400;
static const double DELTA_W = (MAX_W - MIN_W) / NUM_W;

double target_recall;

MultiProbeLshDataModel *model;
			 
bool is_good_value (double v)
{
	return ((v > -std::numeric_limits<double>::max()) && (v < std::numeric_limits<double>::max()));
}

double recall (const lshkit::tune::Input &x) {
	model->setL(x[0]);
	model->setT(x[1]);
	model->setM(MAX_M - x[2]);
	model->setW(MIN_W + DELTA_W * x[3]);
	return model->avgRecall();
}

bool constraint (const tune::Input &x) {
	return recall(x) > target_recall;
}

double cost (const tune::Input &x) {
	model->setL(x[0]);
	model->setT(x[1]);
	model->setM(MAX_M - x[2]);
	model->setW(MIN_W + DELTA_W * x[3]);
	return model->cost();
}

double mplsh_tune(const double& param_M, const double& param_G, 
                  const double& param_a_M, const double& param_b_M, const double& param_c_M, 
                  const double& param_a_G, const double& param_b_G, const double& param_c_G, 
                  int N, int K, int L, int T, int& M, float& W, float R )
{
	M = -1;
	W = -1;
	target_recall = R;
	tune::Interval intervals[]= {{MIN_L, MAX_L + 1}, {MIN_T, MAX_T + 1}, {0, MAX_M - MIN_M + 1}, {0, NUM_W + 1}};
	if (L <= 0 || T <= 0) {
		cout << "mplsh_tune: You need to specify L and T." << endl;
		return 0;
	}

	if (L > 0) {
		intervals[0].begin = L;
		intervals[0].end = L + 1;
	}

	if (T > 0) {
		intervals[1].begin = T;
		intervals[1].end = T + 1;
	}

	if (M > 0) {
		intervals[2].begin = MAX_M - M;
		intervals[2].end = MAX_M - M + 1;
	}

	if (W > 0) {
		intervals[3].begin = (W - MIN_W) / DELTA_W;
		intervals[3].end = intervals[3].begin + 1;
	}

	DataParam param( param_M, param_G, param_a_M, param_b_M, param_c_M, param_a_G, param_b_G, param_c_G );
	double scale = param.scale();

	MultiProbeLshDataModel local_model(param, N, K);
	model = &local_model;

	tune::Range range(intervals, intervals + sizeof intervals /sizeof intervals[0]);
	tune::Input input;
	bool ok = tune::Tune(range, constraint, &input);

	if (ok) {
		M = MAX_M - input[2];
		W = (MIN_W + DELTA_W * input[3])*sqrt(scale);
	} else {
		cout << "mplsh_tune: Failed." << endl;
	}
	
	double tune_cost = cost( input );
	return tune_cost;
}
