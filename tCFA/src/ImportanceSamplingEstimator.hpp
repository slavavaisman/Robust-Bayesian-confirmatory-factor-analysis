#ifndef IMPORTANCESAMPLINGESTIMATOR_HPP
#define IMPORTANCESAMPLINGESTIMATOR_HPP
#include <iostream>
#include <string>
#include <armadillo>
#include <vector>
#include "NormCFA.hpp"
#include "StudentTCFA.hpp"
#include "Utilities.hpp"
#include "Distributions.hpp"
#include "InputReader.hpp"
namespace IS
{

class ImportanceSamplingEstimator
{
public:
    ImportanceSamplingEstimator()
    {
        
    }
    ~ImportanceSamplingEstimator()
    {
        
    }
    
  
   

    void NormalMargLikl(std::string inputFilePath, std::string modelString, NormalCfa::Parameters& params,
                std::string mcmcOutFilePath, std::string mcmcLatentOutFilePath, 
                double& estimator, double& RE, double& time,
                uint32_t samplesz, uint32_t replicationnumber)
    {
         std::cout<<"\n"<<"--------------------------------------------------------------------"<<"\n";
         std::cout<<"Normal model Marginal Likelihood estimation"<<"\n";
         arma::wall_clock timer;
         timer.tic(); 

         inputreader::InputReader inreader( modelString, inputFilePath );
         arma::mat Theta, W; Theta.load(mcmcOutFilePath); W.load(mcmcLatentOutFilePath);
         arma::mat Y  = inreader.GetData();
         // the first row is zeros, so we remove it
         Theta.shed_row(0); W.shed_row(0);
         
         uint32_t p = inreader.GetNumVariablesP(); uint32_t q = inreader.GetNumFactorsQ(); uint32_t n = inreader.GetNumOfPeopleN();
         std::vector<uint32_t> psi_ids, L_ids, Phi_ids, W_ids;
         NormalGetStructureIDs(p, q, n, psi_ids, L_ids,  Phi_ids,  W_ids);
         
         
         arma::vec psi_alpha_is, psi_beta_is, W_mean_is, W_std_is;
         arma::mat L_is_mean, L_is_std, Phi_A;
         double Phi_v_is;
         
         // Get Importance Densities
         std::vector<distributions::Gamma1> psi_is_dist;
         std::vector<std::vector<distributions::Normal>> L_is_dist;
         std::vector<distributions::Normal> W_is_dist;
         distributions::InverseWishart phi_is_dist;
         
        
         GetImportanceDensitiesNormal(Theta, W, p,q,n, psi_ids,  L_ids, Phi_ids,  W_ids,
                                psi_alpha_is, psi_beta_is, L_is_mean, L_is_std, 
                                Phi_A, Phi_v_is, W_mean_is, W_std_is,
                                psi_is_dist, L_is_dist, phi_is_dist, W_is_dist);
         
         
         // Get Prior Densities
         
         distributions::InverseGamma2 psi_prior_dist;
         distributions::MVNormal w_prior_dist;
         distributions::InverseWishart Phi_prior_dist; 
         distributions::Normal L_prior_dist;
         distributions::Normal L_prior_dist_cross_load;
         std::vector<std::vector<std::string>> Lpattern ;
         std::vector<std::pair<uint32_t, uint32_t>> lFixedOneIndices;
         std::vector<std::pair<uint32_t, uint32_t>> lFreeIndices;
         std::vector<std::pair<uint32_t, uint32_t>> lCrossLoadingIndices;
         GetPriorDensitiesNormal(n,  p, q, params, inreader, psi_prior_dist, w_prior_dist, L_prior_dist, 
                 L_prior_dist_cross_load, Phi_prior_dist,  Lpattern,
                 lFixedOneIndices,  lFreeIndices,  lCrossLoadingIndices);
         
         
         // IMPORTANCE SAMPLING MONTE CARLO
         // sample from IS -- preallocate
         arma::vec psi_is = arma::zeros(p);
         arma::mat L_is = arma::mat(p,q);
         arma::mat Phi_is = arma::mat(q,q);
         arma::vec W_is_vec = arma::zeros(n*q);
         arma::mat W_is = arma::mat(n,q);
                
         arma::vec repEll = arma::zeros(replicationnumber);   
         std::vector<double> logEll(samplesz,-std::numeric_limits<double>::infinity());         
         
         for(uint32_t r=0; r< replicationnumber; r++)
         {
             repEll(r) = -std::numeric_limits<double>::infinity(); 
             for(uint32_t i=0; i< samplesz; i++)
             {
                  if(i % 10 == 0)
                  {
                    double perc = (double)(r*samplesz) + (i+1.0);
                    perc = (perc+1.0)/(double)(replicationnumber*samplesz);
                    utilities::PrintProgressBar(perc); 
                  }
                  double W_importance = 0;
                  SampleFromISNormalModel(n,p,q, psi_is, L_is, Phi_is, W_is_vec, W_is, W_importance,
                          psi_is_dist, L_is_dist, W_is_dist, phi_is_dist);
                 
                 
                  double logPrior = GetLogPriorNormal(psi_is, L_is, Phi_is,  W_is, psi_prior_dist,
                     w_prior_dist,  L_prior_dist,  L_prior_dist_cross_load,  Phi_prior_dist,
                     Lpattern, lFixedOneIndices, lFreeIndices, lCrossLoadingIndices);
                  
                  // get log liklihood
                  double logLikl = LogPLiklihoodNormal(psi_is, W_is , L_is, Y);   
                  logEll[i] = logLikl + logPrior - W_importance;
             }
             
             // log calculation
             double c =  *std::max_element(logEll.begin(),logEll.end());
             double logL = 0;
             double logM2 = 0;
             for(uint32_t i=0; i<logEll.size(); i++)
             {
                logL += exp(logEll[i]-c);
                logM2 += exp(2.0*logEll[i]-2.0*c);
             }
             
             double logLmean  = log(logL) + c - log((double)logEll.size());
             //double REsq = (exp(log((double)logEll.size()) + log(logM2) - 2.0*log(logL) ) - 1.0)/((double)logEll.size());
             //std::cout<<"logLmean: "<< logLmean<< " RE^2: "<<2*log(REsq)<<std::endl; 
             repEll(r) =  logLmean;
         }
         utilities::PrintProgressBar(1.0);
         //std::cout<<repEll<<std::endl;
         
         std::cout<<"\n";
         RE = abs(arma::stddev(repEll)/arma::mean(repEll)/std::sqrt((double)repEll.size()));
         estimator = arma::mean(repEll);
         std::cout<<"mean: "<<estimator<< " RE: "<<RE <<std::endl;
         time = timer.toc();
         std::cout <<"execution time: " << time << " seconds \n"; 
         std::cout<<"--------------------------------------------------------------------"<<"\n";
    }

    
    
    void StudentMargLikl(std::string inputFilePath, std::string modelString, StudentCfa::Parameters& params,
                std::string mcmcOutFilePath, std::string mcmcLatentOutFilePath, 
                double& estimator, double& RE, double& time,
                uint32_t samplesz,
                uint32_t replicationnumber)
    {
         std::cout<<"\n"<<"--------------------------------------------------------------------"<<"\n";
         std::cout<<"Student model Marginal Likelihood estimation"<<"\n";
         arma::wall_clock timer;
         timer.tic(); 
         inputreader::InputReader inreader( modelString, inputFilePath );
         arma::mat Theta, W;
         Theta.load(mcmcOutFilePath);   
         W.load(mcmcLatentOutFilePath);
         // the first row is zeros, so we remove it
         Theta.shed_row(0); W.shed_row(0);
         arma::mat Y  = inreader.GetData();
         uint32_t p = inreader.GetNumVariablesP();
         uint32_t q = inreader.GetNumFactorsQ();
         uint32_t n = inreader.GetNumOfPeopleN();

         std::vector<uint32_t> psi_ids;
         std::vector<uint32_t> L_ids;
         std::vector<uint32_t> Phi_ids;
         std::vector<uint32_t> Nu_ids;
         std::vector<uint32_t> W_ids;
         
         StudentGetStructureIDs(p, q, n, psi_ids, L_ids,  Phi_ids, Nu_ids, W_ids);
         //utilities::PrintStdVector(Nu_ids);
         
         arma::vec psi_alpha_is;
         arma::vec psi_beta_is;
         arma::mat L_is_mean;
         arma::mat L_is_std;   
         arma::mat Phi_A;
         double Phi_v_is;
         arma::vec nu_alpha_is;
         arma::vec nu_beta_is;
         arma::vec W_mean_is;
         arma::vec W_std_is;
         
          // Get Importance Densities
         std::vector<distributions::Gamma1> psi_is_dist;
         std::vector<distributions::Gamma1> nu_is_dist;
         std::vector<std::vector<distributions::Normal>> L_is_dist;
         std::vector<distributions::Normal> W_is_dist;
         distributions::InverseWishart phi_is_dist;
         
         GetImportanceDensitiesStudent(Theta, W, p,q,n, psi_ids,  L_ids, Phi_ids,  Nu_ids, W_ids,
                                psi_alpha_is, psi_beta_is, nu_alpha_is, nu_beta_is,
                                L_is_mean, L_is_std, 
                                Phi_A, Phi_v_is, W_mean_is, W_std_is,
                                psi_is_dist, L_is_dist, phi_is_dist, nu_is_dist, W_is_dist);  
   

              // Get Prior Densities
         
         distributions::InverseGamma2 psi_prior_dist;
         distributions::MVNormal w_prior_dist;
         distributions::InverseWishart Phi_prior_dist; 
         distributions::Normal L_prior_dist;
         distributions::Normal L_prior_dist_cross_load;
         std::vector<std::vector<std::string>> Lpattern ;
         std::vector<std::pair<uint32_t, uint32_t>> lFixedOneIndices;
         std::vector<std::pair<uint32_t, uint32_t>> lFreeIndices;
         std::vector<std::pair<uint32_t, uint32_t>> lCrossLoadingIndices;
         GetPriorDensitiesStudent(n,  p, q, params, inreader, psi_prior_dist, w_prior_dist, L_prior_dist, 
                 L_prior_dist_cross_load, Phi_prior_dist,  Lpattern,
                 lFixedOneIndices,  lFreeIndices,  lCrossLoadingIndices);
         
         // IMPORTANCE SAMPLING MONTE CARLO
         // sample from IS -- preallocate
        
         arma::vec psi_is = arma::zeros(p);
         arma::vec nu_is = arma::zeros(p);
         arma::mat L_is = arma::mat(p,q);
         arma::mat Phi_is = arma::mat(q,q);
         arma::mat W_is = arma::mat(n,q);
         arma::vec W_is_vec = arma::zeros(n*q);   
         
         arma::vec repEll = arma::zeros(replicationnumber);   
         std::vector<double> logEll(samplesz,-std::numeric_limits<double>::infinity());         
         
         for(uint32_t r=0; r< replicationnumber; r++)
         {
             repEll(r) = -std::numeric_limits<double>::infinity(); 
             for(uint32_t i=0; i< samplesz; i++)
             {
                  if(i % 10 == 0)
                  {
                    double perc = (double)(r*samplesz) + (i+1.0);
                    perc = (perc+1.0)/(double)(replicationnumber*samplesz);
                    utilities::PrintProgressBar(perc); 
                  }
                  double W_importance = 0;
                  SampleFromISStudentModel(n, p, q, psi_is, nu_is,  L_is,  Phi_is, W_is_vec,  W_is,  W_importance,
                      psi_is_dist, nu_is_dist, L_is_dist, W_is_dist, phi_is_dist,
                      lFixedOneIndices, lFreeIndices, lCrossLoadingIndices);
                 
                  double logPrior = GetLogPriorStudent(psi_is, L_is, Phi_is,  W_is, psi_prior_dist,
                      w_prior_dist,  L_prior_dist,  L_prior_dist_cross_load,  Phi_prior_dist,
                      Lpattern, lFixedOneIndices, lFreeIndices, lCrossLoadingIndices, params);
                  
                  // get log liklihood
                  double logLikl = LogPLiklihoodStudent(psi_is, W_is , L_is, Y, nu_is);  
                  logEll[i] = logLikl + logPrior - W_importance;
             }
             
             // log calculation
             double c =  *std::max_element(logEll.begin(),logEll.end());
             double logL = 0;
             double logM2 = 0;
             for(uint32_t i=0; i<logEll.size(); i++)
             {
                logL += exp(logEll[i]-c);
                logM2 += exp(2.0*logEll[i]-2.0*c);
             }
             
             double logLmean  = log(logL) + c - log((double)logEll.size());
             //double REsq = (exp(log((double)logEll.size()) + log(logM2) - 2.0*log(logL) ) - 1.0)/((double)logEll.size());
             //std::cout<<"logLmean: "<< logLmean<< " RE^2: "<<2*log(REsq)<<std::endl; 
             repEll(r) =  logLmean;
         }
         //std::cout<<repEll<<std::endl;
         utilities::PrintProgressBar(1.0); 
         std::cout<<"\n";
         RE = abs(arma::stddev(repEll)/arma::mean(repEll)/std::sqrt((double)repEll.size()));
         estimator = arma::mean(repEll);
         std::cout<<"mean: "<<estimator<< " RE: "<<RE <<std::endl;
         time = timer.toc();
         std::cout <<"execution time: " << time << " seconds \n"; 
         std::cout<<"--------------------------------------------------------------------"<<"\n";
         
         
    }
    
    
    private:
    
    
        double LogPLiklihoodStudent(arma::vec& psi_diag, arma::mat& W, arma::mat& L, arma::mat& y, arma::vec& nu)
            {
                double llikl = 0;
                for(uint32_t i=0; i<W.n_rows; i++)
                {   
                    arma::vec mu = L*W.row(i).t();
                    for(uint32_t k=0; k<psi_diag.size(); k++)
                    {
                        distributions::Studentst dist(nu[k], mu(k), std::sqrt(psi_diag[k]));
                        llikl += dist.LogProbDist(y(i,k));
                    }
                }
                return llikl;
            }

    
         double LogPLiklihoodNormal(arma::vec& psi_diag, arma::mat& W , arma::mat& L, arma::mat& y)
         {
                double llikl = 0;
                for(uint32_t i=0; i<W.n_rows; i++)
                {   
                    arma::vec mu = L*W.row(i).t();
                    for(uint32_t k=0; k<psi_diag.size(); k++)
                    {
                        distributions::Normal dist(mu(k), std::sqrt(psi_diag(k)));
                        llikl += dist.LogProbDist(y(i,k));
                    }
                }
                return llikl;
          }
    
    
       double GetLogPriorNormal(arma::vec& psi_is, arma::mat& L_is, arma::mat& Phi_is,  arma::mat& W_is,
                       distributions::InverseGamma2& psi_prior_dist,
                distributions::MVNormal& w_prior_dist, distributions::Normal& L_prior_dist, 
                distributions::Normal& L_prior_dist_cross_load, distributions::InverseWishart& Phi_prior_dist,
                std::vector<std::vector<std::string>>& Lpattern,
                std::vector<std::pair<uint32_t, uint32_t>>& lFixedOneIndices,
                std::vector<std::pair<uint32_t, uint32_t>>& lFreeIndices,
                std::vector<std::pair<uint32_t, uint32_t>>& lCrossLoadingIndices)
                {
                    double logPrior = 0;
                    logPrior+=LogPriorPsi(psi_is,psi_prior_dist);
                    logPrior+=LogPriorPhi(Phi_is,Phi_prior_dist);
                    logPrior+=LogPriorL(L_is,L_prior_dist,L_prior_dist_cross_load,lFixedOneIndices, lFreeIndices, lCrossLoadingIndices);
                    logPrior+= LogPriorW(W_is,w_prior_dist);
                    
                    return logPrior;
                }
                
       double GetLogPriorStudent(arma::vec& psi_is, arma::mat& L_is, arma::mat& Phi_is,  arma::mat& W_is,
                       distributions::InverseGamma2& psi_prior_dist,
                distributions::MVNormal& w_prior_dist, distributions::Normal& L_prior_dist, 
                distributions::Normal& L_prior_dist_cross_load, distributions::InverseWishart& Phi_prior_dist,
                std::vector<std::vector<std::string>>& Lpattern,
                std::vector<std::pair<uint32_t, uint32_t>>& lFixedOneIndices,
                std::vector<std::pair<uint32_t, uint32_t>>& lFreeIndices,
                std::vector<std::pair<uint32_t, uint32_t>>& lCrossLoadingIndices, StudentCfa::Parameters& params)
                {
                    double logPrior = 0;
                    logPrior+=LogPriorPsi(psi_is,psi_prior_dist);
                    logPrior+=LogPriorPhi(Phi_is,Phi_prior_dist);
                    logPrior+=LogPriorL(L_is,L_prior_dist,L_prior_dist_cross_load,lFixedOneIndices, lFreeIndices, lCrossLoadingIndices);
                    logPrior+= LogPriorW(W_is,w_prior_dist);
                    logPrior+=  ((double)psi_is.size()) * log(1.0/(double)params.m_max_nu);
                    return logPrior;
                }
    
        void GetPriorDensitiesStudent(uint32_t n, uint32_t  p, uint32_t q, StudentCfa::Parameters params, 
                inputreader::InputReader& inreader, distributions::InverseGamma2& psi_prior_dist,
                distributions::MVNormal& w_prior_dist, distributions::Normal& L_prior_dist, 
                distributions::Normal& L_prior_dist_cross_load, distributions::InverseWishart& Phi_prior_dist,
                std::vector<std::vector<std::string>>& Lpattern,
                std::vector<std::pair<uint32_t, uint32_t>>& lFixedOneIndices,
                std::vector<std::pair<uint32_t, uint32_t>>& lFreeIndices,
                std::vector<std::pair<uint32_t, uint32_t>>& lCrossLoadingIndices)
                {
                     arma::mat I = arma::eye(q,q);   
                     psi_prior_dist = distributions::InverseGamma2(params.m_alpha_prior, params.m_beta_prior);
                     w_prior_dist = distributions::MVNormal(arma::zeros(q),I);
                     Phi_prior_dist = distributions::InverseWishart(I, params.m_v);
                     L_prior_dist = distributions::Normal(0, params.m_sigma);
                     L_prior_dist_cross_load = distributions::Normal(0, params.m_sigma_cross_loadings);
                     Lpattern =  inreader.GetFactorLoadingMatrixPattern();
             
                     for(uint32_t i=0; i<p; i++)
                     {
                        for(uint32_t j=0; j<q; j++)
                        {
                            if(Lpattern[i][j].compare("1") == 0 )
                            {
                                lFixedOneIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                            }
                                        
                            if(Lpattern[i][j].compare("?") == 0 )
                            {
                                lFreeIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                            }
                            if(params.m_sigma_cross_loadings>0 &&  Lpattern[i][j].compare("cl") == 0 )
                            {
                                lCrossLoadingIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                            }
                        }
                    }
                }
    
        void  GetPriorDensitiesNormal(uint32_t n, uint32_t  p, uint32_t q, NormalCfa::Parameters params, 
                inputreader::InputReader& inreader, distributions::InverseGamma2& psi_prior_dist,
                distributions::MVNormal& w_prior_dist, distributions::Normal& L_prior_dist, 
                distributions::Normal& L_prior_dist_cross_load, distributions::InverseWishart& Phi_prior_dist,
                std::vector<std::vector<std::string>>& Lpattern,
                std::vector<std::pair<uint32_t, uint32_t>>& lFixedOneIndices,
                std::vector<std::pair<uint32_t, uint32_t>>& lFreeIndices,
                std::vector<std::pair<uint32_t, uint32_t>>& lCrossLoadingIndices)
         {
             arma::mat I = arma::eye(q,q);   
             psi_prior_dist = distributions::InverseGamma2(params.m_alpha_prior, params.m_beta_prior);
             w_prior_dist = distributions::MVNormal(arma::zeros(q),I);
             Phi_prior_dist = distributions::InverseWishart(I, params.m_v);
             L_prior_dist = distributions::Normal(0, params.m_sigma);
             L_prior_dist_cross_load = distributions::Normal(0, params.m_sigma_cross_loadings);
             Lpattern =  inreader.GetFactorLoadingMatrixPattern();
             
             for(uint32_t i=0; i<p; i++)
             {
                for(uint32_t j=0; j<q; j++)
                {
                    if(Lpattern[i][j].compare("1") == 0 )
                    {
                        lFixedOneIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                    }
                                
                    if(Lpattern[i][j].compare("?") == 0 )
                    {
                        lFreeIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                    }
                    if(params.m_sigma_cross_loadings>0 &&  Lpattern[i][j].compare("cl") == 0 )
                    {
                        lCrossLoadingIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                    }
                }
            }
        }
            
            double LogPriorPsi(arma::vec& psi_diag, distributions::InverseGamma2& psi_prior_dist)
            {
                double llkl = 0;
                for(uint32_t k=0; k<psi_diag.size(); k++)
                {
                    llkl+=psi_prior_dist.LogProbDist(psi_diag(k));
                }
                return llkl;
            }
        
            double LogPriorPhi(arma::mat& Phi, distributions::InverseWishart& iw_prior_dist)
            {
                return iw_prior_dist.LogProbDist(Phi);
            }
            
            double LogPriorW(arma::mat& W, distributions::MVNormal& w_prior_dist)
            {                
                double llkl = 0;
                for(uint32_t i=0; i<W.n_rows; i++)
                {
                     arma::vec v = W.row(i).t();
                     llkl+=w_prior_dist.LogProbDist(v);
                }
               
                return llkl;
            }
            
            double LogPriorL(arma::mat& L,  distributions::Normal& L_prior_dist, distributions::Normal& L_prior_dist_cross_load,
                    std::vector<std::pair<uint32_t, uint32_t>>& lFixedOneIndices,
                    std::vector<std::pair<uint32_t, uint32_t>>& lFreeIndices,
                    std::vector<std::pair<uint32_t, uint32_t>>& lCrossLoadingIndices)
            {
                
                 double llkl = 0;
                 
                 for(uint32_t id=0; id<lFreeIndices.size(); id++)
                 {
                      uint32_t k = lFreeIndices[id].first;
                      uint32_t j = lFreeIndices[id].second;
                      llkl+=L_prior_dist.LogProbDist(L(k,j));
                 }
                 
                 for(uint32_t id=0; id<lCrossLoadingIndices.size(); id++)
                 {
                      uint32_t k = lCrossLoadingIndices[id].first;
                      uint32_t j = lCrossLoadingIndices[id].second;
                      llkl+=L_prior_dist_cross_load.LogProbDist(L(k,j));
                 }
                
                return llkl;
            }

        















  
         void SampleFromISNormalModel(uint32_t n, uint32_t p, uint32_t q, 
                      arma::vec& psi_is, arma::mat& L_is, arma::mat& Phi_is, arma::vec& W_is_vec, arma::mat& W_is, 
                      double& W_importance,
                      std::vector<distributions::Gamma1>& psi_is_dist,
                      std::vector<std::vector<distributions::Normal>>& L_is_dist,
                      std::vector<distributions::Normal>& W_is_dist,
                      distributions::InverseWishart& phi_is_dist)
        {
             W_importance = 0;
             for(uint32_t i=0; i<p; i++)
             {
                 psi_is(i) =  psi_is_dist[i].Rvs();
                 W_importance+=psi_is_dist[i].LogProbDist(psi_is(i));
             }
             
             
             for(uint32_t i=0; i<p; i++)
             {
                 for(uint32_t j=0; j<q; j++)
                 {
                    L_is(i,j) =  L_is_dist[i][j].Rvs();
                    W_importance+=L_is_dist[i][j].LogProbDist(L_is(i,j));
                 }
             }
             
 
             Phi_is = phi_is_dist.Rvs();
             W_importance+=phi_is_dist.LogProbDist(Phi_is);
             for(uint32_t i=0; i<n*q; i++)
             {
                 W_is_vec(i) =  W_is_dist[i].Rvs();
                 W_importance+=W_is_dist[i].LogProbDist( W_is_vec(i));
             }
             W_is = arma::reshape(W_is_vec,q,n).t();
        }
        
        
         void SampleFromISStudentModel(uint32_t n, uint32_t p, uint32_t q, 
                      arma::vec& psi_is, arma::vec& nu_is, arma::mat& L_is, arma::mat& Phi_is, arma::vec& W_is_vec, arma::mat& W_is, 
                      double& W_importance,
                      std::vector<distributions::Gamma1>& psi_is_dist,
                      std::vector<distributions::Gamma1>& nu_is_dist,
                      std::vector<std::vector<distributions::Normal>>& L_is_dist,
                      std::vector<distributions::Normal>& W_is_dist,
                      distributions::InverseWishart& phi_is_dist,
                      std::vector<std::pair<uint32_t, uint32_t>>& lFixedOneIndices,
                    std::vector<std::pair<uint32_t, uint32_t>>& lFreeIndices,
                    std::vector<std::pair<uint32_t, uint32_t>>& lCrossLoadingIndices)
        {
             W_importance = 0;
             for(uint32_t i=0; i<p; i++)
             {
                 psi_is(i) =  psi_is_dist[i].Rvs();
                 W_importance+=psi_is_dist[i].LogProbDist( psi_is(i) );
             }
             for(uint32_t i=0; i<p; i++)
             {
                 nu_is(i) =  nu_is_dist[i].Rvs();
                 W_importance+=nu_is_dist[i].LogProbDist( nu_is(i) );
             }
             for(uint32_t i=0; i<p; i++)
             {
                 for(uint32_t j=0; j<q; j++)
                 {
                    L_is(i,j) =  L_is_dist[i][j].Rvs();
                    W_importance+=L_is_dist[i][j].LogProbDist( L_is(i,j) );
                 }
             }
             Phi_is = phi_is_dist.Rvs();
             W_importance+=phi_is_dist.LogProbDist( Phi_is );
             for(uint32_t i=0; i<n*q; i++)
             {
                 W_is_vec(i) =  W_is_dist[i].Rvs();
                 W_importance+=W_is_dist[i].LogProbDist( W_is_vec(i));
             }
             W_is = arma::reshape(W_is_vec,q,n).t();
        }
        
        void GetImportanceDensitiesStudent(const arma::mat& Theta, const  arma::mat& W,uint32_t p, uint32_t q, uint32_t n, 
               const std::vector<uint32_t>& psi_ids, const std::vector<uint32_t>& L_ids,
               const std::vector<uint32_t>& Phi_ids, const std::vector<uint32_t>& Nu_ids, std::vector<uint32_t>& W_ids, 
               arma::vec& psi_alpha_is, arma::vec& psi_beta_is, 
               arma::vec& nu_alpha_is, arma::vec& nu_beta_is,
               arma::mat& L_is_mean, arma::mat& L_is_std, 
               arma::mat& Phi_A, double& Phi_v_is, arma::vec& W_mean_is, arma::vec& W_std_is,
               std::vector<distributions::Gamma1>& psi_is_dist,
               std::vector<std::vector<distributions::Normal>>& L_is_dist,
               distributions::InverseWishart& phi_is_dist,
               std::vector<distributions::Gamma1>& nu_is_dist,
               std::vector<distributions::Normal>& W_is_dist)
        {
        arma::vec theta_hat = arma::mean(Theta).t();
            arma::vec theta_stds = arma::stddev(Theta).t();
            
            // psi
            psi_alpha_is = arma::zeros(p);
            psi_beta_is = arma::zeros(p);
            for(uint32_t i=0; i<p; i++)
            {
                psi_beta_is(i) =  pow(theta_stds(psi_ids[i]),2.0)/theta_hat(psi_ids[i]);
                psi_alpha_is(i) = theta_hat(psi_ids[i])/psi_beta_is(i);
            }
            // L
            arma::vec L_is_meanVec = arma::zeros(p*q);
            arma::vec L_is_stdVec = arma::zeros(p*q);
            for(uint32_t i=0; i<p*q; i++)
            {
                L_is_meanVec(i) = theta_hat(L_ids[i]);
                L_is_stdVec(i) = theta_stds(L_ids[i]);
            }
            L_is_mean = arma::reshape(L_is_meanVec,p,q);
            L_is_std = arma::reshape(L_is_stdVec,p,q);
            // Phi
            arma::vec meanA_is = theta_hat.subvec(Phi_ids[0],Phi_ids[Phi_ids.size()-1]);
            double Phi11Var = pow(theta_stds(Phi_ids[0]),2.0);
            arma::mat meanA_isMat = arma::reshape(meanA_is,q,q);
            
            Phi_v_is = 3+q+2.0*(meanA_isMat(1,1)*meanA_isMat(1,1) / Phi11Var);
            Phi_A = meanA_isMat*(Phi_v_is-q-1.0);
  
            // W
            W_mean_is = arma::mean(W).t();
            W_std_is = arma::stddev(W).t();
            // nu
            nu_alpha_is = arma::zeros(p);
            nu_beta_is = arma::zeros(p);
            for(uint32_t i=0; i<p; i++)
            {
                nu_beta_is(i) =  pow(theta_stds(Nu_ids[i]),2.0)/theta_hat(Nu_ids[i]);
                nu_alpha_is(i) = theta_hat(Nu_ids[i])/psi_beta_is(i);
            }
            
            phi_is_dist = distributions::InverseWishart(Phi_A, Phi_v_is);
     
             for(uint32_t i=0; i<psi_alpha_is.size(); i++)
             {
                 psi_is_dist.push_back(distributions::Gamma1(psi_alpha_is[i],psi_beta_is[i]));
             }  
             for(uint32_t i=0; i<nu_alpha_is.size(); i++)
             {
                 nu_is_dist.push_back(distributions::Gamma1(nu_alpha_is[i],nu_beta_is[i]));
             }           
             for(uint32_t i=0; i<W_mean_is.size(); i++)
             {
                 W_is_dist.push_back(distributions::Normal(W_mean_is[i],W_std_is[i]));
             }
             for(uint32_t i=0; i<L_is_mean.n_rows; i++)
             {
                 std::vector<distributions::Normal> row;
                 for(uint32_t j=0; j<L_is_mean.n_cols; j++)
                 {
                     distributions::Normal dist(L_is_mean(i,j), L_is_std(i,j));
                     row.push_back(dist);
                 }
                 L_is_dist.push_back(row);
             }
        }
        
           
        void GetImportanceDensitiesNormal(const arma::mat& Theta, const  arma::mat& W,uint32_t p, uint32_t q, uint32_t n, 
               const std::vector<uint32_t>& psi_ids, const std::vector<uint32_t>& L_ids,
               const std::vector<uint32_t>& Phi_ids, std::vector<uint32_t>& W_ids, 
               arma::vec& psi_alpha_is, arma::vec& psi_beta_is, 
               arma::mat& L_is_mean, arma::mat& L_is_std, 
               arma::mat& Phi_A, double& Phi_v_is, arma::vec& W_mean_is, arma::vec& W_std_is,
               std::vector<distributions::Gamma1>& psi_is_dist,
               std::vector<std::vector<distributions::Normal>>& L_is_dist,
               distributions::InverseWishart& phi_is_dist,
               std::vector<distributions::Normal>& W_is_dist
               )
        {
            arma::vec theta_hat = arma::mean(Theta).t();
            arma::vec theta_stds = arma::stddev(Theta).t();
            
            // psi
            psi_alpha_is = arma::zeros(p);
            psi_beta_is = arma::zeros(p);
            for(uint32_t i=0; i<p; i++)
            {
                psi_beta_is(i) =  pow(theta_stds(psi_ids[i]),2.0)/theta_hat(psi_ids[i]);
                psi_alpha_is(i) = theta_hat(psi_ids[i])/psi_beta_is(i);
                psi_is_dist.push_back(distributions::Gamma1(psi_alpha_is[i],psi_beta_is[i]));
            }
                     
            // L
            arma::vec L_is_meanVec = arma::zeros(p*q);
            arma::vec L_is_stdVec = arma::zeros(p*q);
            for(uint32_t i=0; i<p*q; i++)
            {
                L_is_meanVec(i) = theta_hat(L_ids[i]);
                L_is_stdVec(i) = theta_stds(L_ids[i]);
            }
            L_is_mean = arma::reshape(L_is_meanVec,p,q);
            L_is_std = arma::reshape(L_is_stdVec,p,q);
            for(uint32_t i=0; i<L_is_mean.n_rows; i++)
            {
                std::vector<distributions::Normal> row;
                for(uint32_t j=0; j<L_is_mean.n_cols; j++)
                {
                     distributions::Normal dist(L_is_mean(i,j), L_is_std(i,j));
                     row.push_back(dist);
                }
                L_is_dist.push_back(row);
            }
            // Phi
            arma::vec meanA_is = theta_hat.subvec(Phi_ids[0],Phi_ids[Phi_ids.size()-1.0]);
            double Phi11Var = pow(theta_stds(Phi_ids[0]),2.0);
            arma::mat meanA_isMat = arma::reshape(meanA_is,q,q);
            
            Phi_v_is = 3.0 + q + 2.0*(meanA_isMat(1,1)*meanA_isMat(1,1) / Phi11Var);
            Phi_A = meanA_isMat*(Phi_v_is-q-1.0);
            phi_is_dist = distributions::InverseWishart(Phi_A, Phi_v_is);
            
            
            // W
            W_mean_is = arma::mean(W).t();
            W_std_is = arma::stddev(W).t();

            for(uint32_t i=0; i<W_mean_is.size(); i++)
            {
                W_is_dist.push_back(distributions::Normal(W_mean_is[i],W_std_is[i]));
            }     
        }
        
        void GetImportanceDensitiesNormalTest(const arma::mat& Theta, const  arma::mat& W,uint32_t p, uint32_t q, uint32_t n, 
               const std::vector<uint32_t>& psi_ids, const std::vector<uint32_t>& L_ids,
               const std::vector<uint32_t>& Phi_ids, std::vector<uint32_t>& W_ids, 
               arma::vec& psi_alpha_is, arma::vec& psi_beta_is, 
               arma::mat& L_is_mean, arma::mat& L_is_std, 
               arma::mat& Phi_A, double& Phi_v_is, arma::vec& W_mean_is, arma::vec& W_std_is,
               std::vector<distributions::Gamma1>& psi_is_dist,
               std::vector<std::vector<distributions::Normal>>& L_is_dist,
               distributions::InverseWishart& phi_is_dist,
               std::vector<distributions::Normal>& W_is_dist
               )
        {
            arma::vec theta_hat = Theta.row(0).t();
            arma::vec theta_stds = Theta.row(1).t();
            
            // psi
            psi_alpha_is = arma::zeros(p);
            psi_beta_is = arma::zeros(p);
            for(uint32_t i=0; i<p; i++)
            {
                psi_beta_is(i) =  pow(theta_stds(psi_ids[i]),2.0)/theta_hat(psi_ids[i]);
                psi_alpha_is(i) = theta_hat(psi_ids[i])/psi_beta_is(i);
                psi_is_dist.push_back(distributions::Gamma1(psi_alpha_is[i],psi_beta_is[i]));
            }
                     
            // L
            arma::vec L_is_meanVec = arma::zeros(p*q);
            arma::vec L_is_stdVec = arma::zeros(p*q);
            for(uint32_t i=0; i<p*q; i++)
            {
                L_is_meanVec(i) = theta_hat(L_ids[i]);
                L_is_stdVec(i) = theta_stds(L_ids[i]);
            }
            L_is_mean = arma::reshape(L_is_meanVec,p,q);
            L_is_std = arma::reshape(L_is_stdVec,p,q);
            for(uint32_t i=0; i<L_is_mean.n_rows; i++)
            {
                std::vector<distributions::Normal> row;
                for(uint32_t j=0; j<L_is_mean.n_cols; j++)
                {
                     distributions::Normal dist(L_is_mean(i,j), L_is_std(i,j));
                     row.push_back(dist);
                }
                L_is_dist.push_back(row);
            }
            // Phi
            arma::vec meanA_is = theta_hat.subvec(Phi_ids[0],Phi_ids[Phi_ids.size()-1.0]);
            double Phi11Var = pow(theta_stds(Phi_ids[0]),2.0);
            arma::mat meanA_isMat = arma::reshape(meanA_is,q,q);
            
            Phi_v_is = 3.0 + q + 2.0*(meanA_isMat(1,1)*meanA_isMat(1,1) / Phi11Var);
            Phi_A = meanA_isMat*(Phi_v_is-q-1.0);
        
            
            phi_is_dist = distributions::InverseWishart(Phi_A, Phi_v_is);
            
            
            // W
            W_mean_is = W.row(0).t();
            W_std_is = W.row(1).t();

            for(uint32_t i=0; i<W_mean_is.size(); i++)
            {
                W_is_dist.push_back(distributions::Normal(W_mean_is[i],W_std_is[i]));
            }     
        }
        
        void GetStructureFromVectorNormal(arma::vec& theta, arma::vec& Wrow, uint32_t p, uint32_t q, uint32_t n, 
            arma::vec& psi, arma::mat& L, arma::mat& Phi, arma::mat& W, 
            const std::vector<uint32_t>& psi_ids, const std::vector<uint32_t>& L_ids,
            const std::vector<uint32_t>& Phi_ids, std::vector<uint32_t>& W_ids)
        {
            psi = theta.subvec(psi_ids[0],psi_ids[psi_ids.size()-1]);
            L = arma::reshape( theta.subvec(L_ids[0],L_ids[L_ids.size()-1]),p,q );
            Phi = arma::reshape( theta.subvec(Phi_ids[0],Phi_ids[Phi_ids.size()-1]),q,q );
            W = arma::reshape(Wrow,q,n ).t();
        }
        
        void GetStructureFromVectorStudent(arma::vec& theta, arma::vec& Wrow, uint32_t p, uint32_t q, uint32_t n, 
            arma::vec& psi, arma::mat& L, arma::mat& Phi, arma::vec& nu, arma::mat& W, 
            const std::vector<uint32_t>& psi_ids, const std::vector<uint32_t>& L_ids,
            const std::vector<uint32_t>& Phi_ids, const std::vector<uint32_t>& Nu_ids, const std::vector<uint32_t>& W_ids)
        {
             psi = theta.subvec(psi_ids[0],psi_ids[psi_ids.size()-1]);
             L = arma::reshape( theta.subvec(L_ids[0],L_ids[L_ids.size()-1]),p,q );
             Phi = arma::reshape( theta.subvec(Phi_ids[0],Phi_ids[Phi_ids.size()-1]),q,q );
             nu = theta.subvec(Nu_ids[0],Nu_ids[Nu_ids.size()-1]);
             W = arma::reshape(Wrow,q,n ).t();
        }

    
        void StudentGetStructureIDs(uint32_t p, uint32_t q, uint32_t n,  
            std::vector<uint32_t>& psi_ids, std::vector<uint32_t>& L_ids, std::vector<uint32_t>& Phi_ids, 
            std::vector<uint32_t>& Nu_ids, std::vector<uint32_t>& W_ids)
        {
            
            
            uint32_t last_id = 0;
            
            for(uint32_t i=0; i<p; i++)
            {
                psi_ids.push_back(last_id);
                last_id++;
            }
            
            for(uint32_t i=0; i<p*q; i++)
            {
                L_ids.push_back(last_id);
                last_id++;
            }
            
            for(uint32_t i=0; i<q*q; i++)
            {
                Phi_ids.push_back(last_id);
                last_id++;
            }
            for(uint32_t i=0; i<q*q; i++)
            {
                Nu_ids.push_back(last_id);
                last_id++;
            }
            
            for(uint32_t i=0; i<n*q; i++)
            {
                W_ids.push_back(i);
            }
            
        }

    
    
         void NormalGetStructureIDs(uint32_t p, uint32_t q, uint32_t n,  
            std::vector<uint32_t>& psi_ids, std::vector<uint32_t>& L_ids, std::vector<uint32_t>& Phi_ids, 
            std::vector<uint32_t>& W_ids)
        {
            
            
            uint32_t last_id = 0;
            
            for(uint32_t i=0; i<p; i++)
            {
                psi_ids.push_back(last_id);
                last_id++;
            }
            
            for(uint32_t i=0; i<p*q; i++)
            {
                L_ids.push_back(last_id);
                last_id++;
            }
            
            for(uint32_t i=0; i<q*q; i++)
            {
                Phi_ids.push_back(last_id);
                last_id++;
            }
            
            for(uint32_t i=0; i<n*q; i++)
            {
                W_ids.push_back(i);
            }
            
        }
};

}

#endif // IMPORTANCESAMPLINGESTIMATOR_HPP
