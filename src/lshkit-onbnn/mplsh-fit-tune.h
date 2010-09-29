/***************************************************************************
 *   Copyright (C) 2008 by Regis Behmo,,,   *
 *   regis.behmo@ecp.fr   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/

#ifndef _MPLSH_FIT_TUNE_H_INCLUDED_
#define _MPLSH_FIT_TUNE_H_INCLUDED_

#include "lshkit.h"
#include "lshkit/matrix.h"
#include "lshkit/tune.h"

#include <cstdlib>
#include <gsl/gsl_multifit.h>
// #include <boost/program_options.hpp>
// #include <boost/progress.hpp>

bool is_good_value (double v);
bool constraint (const lshkit::tune::Input &x);

double mplsh_tune(const double& param_M, const double& param_G, 
                  const double& param_a_M, const double& param_b_M, const double& param_c_M, 
                  const double& param_a_G, const double& param_b_G, const double& param_c_G, 
                  int N, int K, int L, int T, int& M, float& W, float R );

template<class DATA, class METRIC> 
void mplsh_fit( const DATA& data, const METRIC& metric, int NN, double& param_M, double& param_G, 
                double& param_a_M, double& param_b_M, double& param_c_M, 
                double& param_a_G, double& param_b_G, double& param_c_G )
{
	// Parameters
	unsigned N, P, Q, K, F;
	N = 0;// # points
	P = 50000;// number of pairs to sample
	Q = 1000;// number of queries to sample
	K = NN;// search for K nearest neighbors
	F = 10;// divide the sample to F folds
	
	std::vector<unsigned> idx(data.size());
	for (unsigned i = 0; i < idx.size(); ++i) idx[i] = i;
	random_shuffle(idx.begin(), idx.end());

	if (N > 0 && N < data.size()) idx.resize(N);

	lshkit::DefaultRng rng;
	rng.seed(0);//plant the same seed, always
	boost::variate_generator<lshkit::DefaultRng &, lshkit::UniformUnsigned> gen(rng, lshkit::UniformUnsigned(0, idx.size()-1));

	double gM = 0.0;
	double gG = 0.0;
	{
		// sample P pairs of points
		for (unsigned k = 0; k < P; ++k)
		{
			double dist, logdist;
			for (;;)
			{
				unsigned i = gen();
				unsigned j = gen();
				if (i == j) continue;
				dist = metric( data[idx[i]], data[idx[j]] );
				logdist = log(dist);
				if (is_good_value(logdist)) break;
			}
			gM += dist;
			gG += logdist;
		}
		gM /= P;
		gG /= P;
		gG = exp(gG);
	}

  // TODO check that
	// Custom
	Q = (idx.size() > 1000)? 1000 : idx.size();// Added
	
	// TODO check that 
  // Supprime
  //if (Q > idx.size()) Q = idx.size();
  //if (K > idx.size() - Q) K = idx.size() - Q;
	
	/* sample query */
	std::vector<unsigned> qry(Q);

	lshkit::SampleQueries(&qry, idx.size(), rng);

	/* do the queries */
	std::vector<lshkit::Topk<unsigned> > topks(Q);
	for (unsigned i = 0; i < Q; ++i) topks[i].reset(K);

	/* ... */
	gsl_matrix *X = gsl_matrix_alloc(F * K, 3);
	gsl_vector *yM = gsl_vector_alloc(F * K);
	gsl_vector *yG = gsl_vector_alloc(F * K);
	gsl_vector *pM = gsl_vector_alloc(3);
	gsl_vector *pG = gsl_vector_alloc(3);
	gsl_matrix *cov = gsl_matrix_alloc(3,3);

	std::vector<double> M(K);
	std::vector<double> G(K);

	// boost::progress_display progress(F, cerr);
	unsigned m = 0;
	for (unsigned l = 0; l < F; l++)
	{
		// Scan
		for (unsigned i = l; i< idx.size(); i += F)
		{
			for (unsigned j = 0; j < Q; j++) 
			{
				int id = qry[j];
				if (i != id) 
				{
					float d = metric( data[idx[id]], data[idx[i]] );
					if (is_good_value(log(double(d)))) topks[j] << lshkit::Topk<unsigned>::Element(i, d);
				}
			}
		}

		std::fill(M.begin(), M.end(), 0.0);
		std::fill(G.begin(), G.end(), 0.0);

		for (unsigned i = 0; i < Q; i++)
		{
			for (unsigned k = 0; k < K; k++)
			{
				M[k] += topks[i][k].dist;
				G[k] += log(topks[i][k].dist);
			}
		}

		for (unsigned k = 0; k < K; k++)
		{
			M[k] = log(M[k]/Q);
			G[k] /= Q;
			gsl_matrix_set(X, m, 0, 1.0);
			gsl_matrix_set(X, m, 1, log(double(data.size() * (l + 1)) / double(F)));
			gsl_matrix_set(X, m, 2, log(double(k + 1)));
			gsl_vector_set(yM, m, M[k]);
			gsl_vector_set(yG, m, G[k]);
			++m;
		}
	}

	gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(F * K, 3);

	double chisq;

	gsl_multifit_linear(X, yM, pM, cov, &chisq, work);
	gsl_multifit_linear(X, yG, pG, cov, &chisq, work);
	
	param_M = gM;
	param_G = gG;
	param_a_M = gsl_vector_get(pM, 0);
	param_b_M = gsl_vector_get(pM, 1);
	param_c_M = gsl_vector_get(pM, 2);
	param_a_G = gsl_vector_get(pG, 0);
	param_b_G = gsl_vector_get(pG, 1);
	param_c_G = gsl_vector_get(pG, 2);
	
	gsl_matrix_free(X);
	gsl_matrix_free(cov);
	gsl_vector_free(pM);
	gsl_vector_free(pG);
	gsl_vector_free(yM);
	gsl_vector_free(yG);
}

template<class DATA, class METRIC> 
double mplsh_fit_tune( const DATA& data, const METRIC& metric, const int& L, const int& T, int& M, float& W, float R = 0.8, int K = 1 )
{
	
	/** ***************************/
	/** ********** FIT  ***********/
	/** ***************************/
  //std::cout << "Fitting..." << std::endl;
	double param_M, param_G, param_a_M, param_b_M, param_c_M, param_a_G, param_b_G, param_c_G;
	mplsh_fit( data, metric, K, param_M, param_G, param_a_M, param_b_M, param_c_M, param_a_G, param_b_G, param_c_G );
	
  //std::cout << param_M << "    " << param_G << std::endl;
  //std::cout << param_a_M << "    " << param_b_M << "    " << param_c_M << std::endl;
  //std::cout << param_a_G << "    " << param_b_G << "    " << param_c_G << std::endl;
	
	/** ***************************/
	/** ********** TUNE ***********/
	/** ***************************/
  //std::cout << "Tuning..." << std::endl;
	return mplsh_tune(param_M, param_G, param_a_M, param_b_M, param_c_M, param_a_G, param_b_G, param_c_G, 
                    data.size(), K, L, T, M, W, R );
  //std::cout << "L = " << L;
  //std::cout << "    " << "T = " << T;
  //std::cout << "    " << "M = " << M;
  //std::cout << "    " << "W = " << W;
  //std::cout << "    " << "R = " << R;
  //std::cout << "    " << "M = " << M;
  //std::cout << std::endl;
}

#endif

