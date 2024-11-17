// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "Rcpp.h"
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <string>
#include <armadillo>
#include <vector>

#include "Utilities.hpp"
#include "Distributions.hpp"
#include "InputReader.hpp"

#include "NormCFA.hpp"
#include "StudentTCFA.hpp"
#include "ImportanceSamplingEstimator.hpp"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//


// [[Rcpp::export]]
void NormalCFA(std::string inputFilePath, std::string modelString, std::string mcmcOutFilePath, std::string mcmcLatentOutFilePath,
               double alpha_prior = 3.0, double beta_prior = 2.0, double sigma_prior=10.0, 
               double sigma_cross_loading_prior = 0.0,  double wishart_degree_freedom_prior = 4, uint32_t burnin = 1000, uint32_t samples = 1000,
               int seed=0)
  
{
  //set_seed_r(seed);
  
  distributions::RandomState* rnd = distributions::RandomState::GetInstance(seed);
  
  NormalCfa::Parameters params;
  params.m_alpha_prior = alpha_prior;
  params.m_beta_prior = beta_prior;
  params.m_sigma = sigma_prior;
  params.m_sigma_cross_loadings = sigma_cross_loading_prior;
  params.m_v  = wishart_degree_freedom_prior;
  params.m_A =  arma::mat(1,1);;
  params.m_N = samples;
  params.m_burnin = burnin;
  params.mcmcOutFilePath = mcmcOutFilePath;
  params.mcmcLatentOutFilePath = mcmcLatentOutFilePath;
  
  inputreader::InputReader inreader( modelString, inputFilePath );
  
  NormalCfa::NormCFA cfa(inreader, params);
  cfa.DoGibbs();
  
} 

// [[Rcpp::export]]
void StudentCFA(std::string inputFilePath, std::string modelString, std::string mcmcOutFilePath, std::string mcmcLatentOutFilePath,
                double alpha_prior = 3.0, double beta_prior = 2.0, double sigma_prior=10.0, 
                double sigma_cross_loading_prior = 0.0, 
                double wishart_degree_freedom_prior = 4, double max_t_degree_freedom_prior = 300,
                uint32_t burnin = 1000, uint32_t samples = 1000, int seed=0)
{
  
  //set_seed_r(std::floor(std::fabs(seed)));
  distributions::RandomState* rnd = distributions::RandomState::GetInstance(seed);
  
  StudentCfa::Parameters params;
  params.m_alpha_prior = alpha_prior;
  params.m_beta_prior = beta_prior;
  params.m_sigma = sigma_prior;
  params.m_sigma_cross_loadings = sigma_cross_loading_prior;
  params.m_v  = wishart_degree_freedom_prior;
  params.m_A = arma::mat(1,1);
  params.m_N = samples;
  params.m_max_nu = max_t_degree_freedom_prior;
  params.m_burnin = burnin;
  params.mcmcOutFilePath = mcmcOutFilePath;
  params.mcmcLatentOutFilePath = mcmcLatentOutFilePath;
  
  inputreader::InputReader inreader( modelString, inputFilePath );
  
  StudentCfa::StudentTCFA cfa(inreader, params);
  cfa.DoGibbs();
  
}

// [[Rcpp::export]]
std::vector<double> NormalCalculateMargLikl(std::string inputFilePath, std::string modelString, 
                             std::string mcmcOutFilePath, std::string mcmcLatentOutFilePath, 
                             double alpha_prior = 3.0, double beta_prior = 2.0, double sigma_prior=10.0, 
                             double sigma_cross_loading_prior = 0.0, 
                             double wishart_degree_freedom_prior = 4, 
                             uint32_t samplesz = 1000,
                             uint32_t replicationnumber = 10)
{
  NormalCfa::Parameters params;
  params.m_alpha_prior = alpha_prior;
  params.m_beta_prior = beta_prior;
  params.m_sigma = sigma_prior;
  params.m_sigma_cross_loadings = sigma_cross_loading_prior;
  params.m_v  = 4.0;
  IS::ImportanceSamplingEstimator is;
  
  double estimator,  RE, time;
  
  is.NormalMargLikl(inputFilePath, modelString, params, mcmcOutFilePath, mcmcLatentOutFilePath, 
                    estimator, RE, time, 
                    samplesz, replicationnumber);
  
  std::vector<double> res{estimator, RE, time};
  return res;
  
}


// [[Rcpp::export]]
std::vector<double> StudentCalculateMargLikl(std::string inputFilePath, std::string modelString, 
                              std::string mcmcOutFilePath, std::string mcmcLatentOutFilePath,
                              double alpha_prior = 3.0, double beta_prior = 2.0, double sigma_prior=10.0, 
                              double sigma_cross_loading_prior = 0.0, 
                              double wishart_degree_freedom_prior = 4, 
                              double nu_max = 300, 
                              uint32_t samplesz = 1000,
                              uint32_t replicationnumber = 10)
{
  StudentCfa::Parameters params;
  params.m_alpha_prior = alpha_prior;
  params.m_beta_prior = beta_prior;
  params.m_sigma = sigma_prior;
  params.m_sigma_cross_loadings = sigma_cross_loading_prior;
  params.m_v  = 4.0;
  params.m_max_nu = nu_max;
  
  IS::ImportanceSamplingEstimator is;
  
  double estimator,  RE, time;
  
  is.StudentMargLikl(inputFilePath, modelString, params, mcmcOutFilePath, mcmcLatentOutFilePath, 
                     estimator, RE, time,  
                     samplesz, replicationnumber);
  
  std::vector<double> res{estimator, RE, time};
  return res;
}