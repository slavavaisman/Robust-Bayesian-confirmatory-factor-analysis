#ifndef NORMCFA_HPP
#define NORMCFA_HPP

#include <iostream>
#include <armadillo>
#include <vector>
#include <string>
#include <armadillo>
#include "InputReader.hpp"
#include <limits>

#include "Utilities.hpp"
#include "Distributions.hpp"
#include "InputReader.hpp"



namespace NormalCfa
{
    struct Parameters
    {
        std::string mcmcOutFilePath;
        std::string mcmcLatentOutFilePath;
        // priors for inverse gamma distributions
        double m_alpha_prior;
        double m_beta_prior;
        
        // prior for Phi (Wishart)
        arma::mat m_A;
        uint32_t m_v;  
        
        // prior for factor loading
        double m_sigma;
        double m_sigma_cross_loadings;
        
        // sample size
        uint32_t m_burnin;
        uint32_t m_N;
    } ;

    class NormCFA
    {
        private:
            inputreader::InputReader& m_input;
            Parameters m_params;
            uint32_t m_p;
            uint32_t m_q;
            uint32_t m_n; 
    
            std::vector<std::string> m_posterior_mcmc_names;
            std::vector<std::string> m_posterior_mcmc_names_full;
            std::vector<std::string> m_posterior_mcmc_latent;
            
            std::vector<std::pair<uint32_t, uint32_t>> m_lFixedOneIndices;
            std::vector<std::pair<uint32_t, uint32_t>> m_lFreeIndices;
            std::vector<std::pair<uint32_t, uint32_t>> m_lCrossLoadingIndices;
            arma::mat m_y;
    
    public:
            NormCFA(inputreader::InputReader& input, Parameters params) :  m_input(input)
            {
                m_p = input.GetNumVariablesP();
                m_q = input.GetNumFactorsQ();
                m_n = input.GetNumOfPeopleN();
                // set default prior parameters
                m_params = params;
                
                arma::vec v = arma::ones(m_q);
                m_params.m_A = arma::diagmat(v);
                                
                CreatePosteriorVarNames();
            }
            
            ~NormCFA()
            {
                
            }
            
            void DoGibbs()
            {
                arma::wall_clock timer;
                timer.tic(); 
            
                std::vector<double> psi_diag;
                arma::mat L;
                
                arma::mat Phi;
                arma::mat W;

                InitGibbs(psi_diag, L, Phi, W);
                  
                m_y = m_input.GetData();  
                
                arma::mat Theta(m_params.m_N, m_p + m_p * m_q + m_q * m_q);
                arma::mat ThetaLatent(m_params.m_N,m_n*m_q);
                
                double perc = 0;
                for(uint32_t i=0; i<m_params.m_burnin+m_params.m_N; i++)
                {
                    
                    if(0 ==i)
                    {
                        std::cout<<"\n"<<"--------------------------------------------------------------------"<<"\n";
                        std::cout<<"Normal CFA"<<"\n";
                        std::cout<<"Burnin step"<<"\n";
                    }
                    
                    if(i<m_params.m_burnin)
                    {
                        perc = (double)i;
                        perc = (perc+1.0)/(double)m_params.m_burnin;
                        utilities::PrintProgressBar(perc); 
                    }
                    
                    // sample psi
                    SamplePsi(psi_diag, m_y, L, W);
                    
                    // sample Phi
                    SamplePhi(Phi, W);
                    
                    // sample W
                    SampleW( W,  Phi,  m_y,   L,   psi_diag);
                    
                    // sample L
                    SampleL( W, Phi, m_y,  L,  psi_diag, m_lFreeIndices, m_lCrossLoadingIndices);
                    
                    if(i>=m_params.m_burnin)
                    {
                        if(i == m_params.m_burnin)
                        {
                            std::cout<<"\n"<<"Sampling step"<<"\n";
                        }
                        
                        perc = (double)i - (double)m_params.m_burnin;
                        perc = (perc+1.0)/(double)m_params.m_N;
                        utilities::PrintProgressBar(perc); 
                        
                        arma::vec tmpV;
                        tmpV = join_cols(tmpV, utilities::StdToVec(psi_diag));
                        tmpV = join_cols(tmpV, vectorise(L));
                        tmpV = join_cols(tmpV, vectorise(Phi));  
                        
                        Theta.row(i-m_params.m_burnin) = tmpV.t();
                      
                        arma::mat Wt = W.t();
                        //tmpV = join_cols(tmpV, vectorise(Wt));
                        ThetaLatent.row(i-m_params.m_burnin) = vectorise(Wt).t();                    
                    }
                }
                Theta.save(arma::csv_name(m_params.mcmcOutFilePath,m_posterior_mcmc_names));
                ThetaLatent.save(arma::csv_name(m_params.mcmcLatentOutFilePath, m_posterior_mcmc_latent));
                
                double sec = timer.toc();
                std::cout <<"\n" <<"execution time: " << sec << " seconds";    
                std::cout<<"\n"<<"--------------------------------------------------------------------"<<"\n";
            }
                
            protected:
        
                    void  SampleLType(arma::mat& W, arma::mat& Phi, arma::mat& y,  arma::mat& L, std::vector<double>&  psi_diag,
                            std::vector<std::pair<uint32_t, uint32_t>> lIndices, double sig)
                {
                    for(uint32_t id=0; id<lIndices.size(); id++)
                    {
                        std::pair<uint32_t,uint32_t> tmp = lIndices[id];
                        uint32_t k = tmp.first;
                        uint32_t j = tmp.second;
                        
                        arma::vec wsum = W.col(j).t() * W.col(j);
                        double sigmaSq = 1.0/( (1.0/(sig*sig))  + wsum(0)/psi_diag[k]);
                        
                        double mu = 0;
                        for(uint32_t i=0; i<m_n; i++)
                        {
                            double tmp2 = 0;
                            for(uint32_t l=0; l<m_q; l++)
                            {
                                if(l != j)
                                    tmp2+=L(k,l)*W(i,l);
                            }
                            mu = mu +  W(i,j)*(y(i,k) - tmp2);
                        }
                        mu = sigmaSq*mu/psi_diag[k];
                        distributions::Normal lkj_dist(mu,std::sqrt(sigmaSq));
                        L(k,j) = lkj_dist.Rvs();
                    }                   
                }
                                
                void  SampleL(arma::mat& W, arma::mat& Phi, arma::mat& y,  arma::mat& L, std::vector<double>&  psi_diag,
                                std::vector<std::pair<uint32_t, uint32_t>> lFreeIndices, std::vector<std::pair<uint32_t, uint32_t>> lCrossLoadingIndices)
                {
                    SampleLType( W, Phi, y,  L,  psi_diag, lFreeIndices, m_params.m_sigma);
                    SampleLType( W, Phi, y,  L,  psi_diag, lCrossLoadingIndices, m_params.m_sigma_cross_loadings);
                }
                
                void SampleW(arma::mat& W, arma::mat&  Phi,  arma::mat& y,  arma::mat& L,  std::vector<double>& psi_diag)
                {
                    arma::mat PhiInv = arma::inv(Phi);
                    arma::mat PsiInv = arma::eye(m_p,m_p);
                    for(uint32_t k=0; k<m_p; k++)
                    {
                        PsiInv(k,k) = 1.0/psi_diag[k];
                    }
                    arma::mat tmp = L.t()*PsiInv;
                    arma::mat Sigma = arma::inv(PhiInv + tmp*L);
                    

                    for(uint32_t i=0; i<m_n; i++)
                    {
                        arma::vec mu_i = Sigma * tmp*y.row(i).t();
                        distributions::MVNormal w_dist_i(mu_i, Sigma);
                        W.row(i) = w_dist_i.Rvs().t();
                    }
                }
                
                 void SamplePhi(arma::mat& Phi, arma::mat&  W)
                 {
                    arma::mat tmp = m_params.m_A + W.t()*W;
                    distributions::InverseWishart phi_dist(tmp, m_params.m_v  + m_n);
                    Phi = phi_dist.Rvs();
                 }
                 
                 void SamplePsi(std::vector<double>& psi_diag,  arma::mat& y, arma::mat& L, arma::mat& W)
                {
                    double alpha = 0.5*((double)m_n) + m_params.m_alpha_prior;
                    for(uint32_t k=0; k<m_p; k++)
                    {
                        double sum = 0;
                        for(uint32_t i=0; i<m_n; i++)
                        {
                            arma::vec tmp = (L.row(k)*W.row(i).t());
                            sum  = sum + (y(i,k) - tmp(0))*(y(i,k) - tmp(0));
                        }
                        
                        double beta = 0.5*sum + m_params.m_beta_prior;
                        distributions::InverseGamma2 g_dist(alpha,beta);
                        psi_diag[k] = g_dist.Rvs();
                    }
                }
            
            void InitGibbs(std::vector<double>& psi_diag, arma::mat& L, arma::mat& Phi, arma::mat& W)
            {
              
              distributions::Uniform distU(-1,1);
              
              distributions::InverseWishart iw_dist(m_params.m_A,m_params.m_v);
              Phi = arma::eye(m_q,m_q);
              psi_diag = std::vector<double>(m_p,1.0);
              distributions::InverseGamma2 ig_dist(m_params.m_alpha_prior, m_params.m_beta_prior);
              for(uint32_t i=0; i<m_p; i++)
              {
                  //psi_diag[i] = ig_dist.Rvs();
                  psi_diag[i] = 1.0;
              }
              
              
              W.zeros(m_n,m_q);
              distributions::MVNormal mv_dist(arma::zeros(m_q),Phi);
              for(uint32_t i=0; i<m_n; i++)
              {
                 //arma::vec row = mv_dist.Rvs();
                  //W.row(i) = row.t();
                  for(uint32_t j=0; j<m_q; j++)
                  {
                    W(i,j) = distU.Rvs();
                  }
              }
              
              std::vector<std::vector<std::string>> Lpattern =  m_input.GetFactorLoadingMatrixPattern();
              distributions::Normal normSigma(0,m_params.m_sigma);
              distributions::Normal normCross;
              if(m_params.m_sigma_cross_loadings>0)
              {
                normCross = distributions::Normal(0,m_params.m_sigma_cross_loadings);
              }
              
              L.zeros(m_p,m_q);
              
              for(uint32_t i=0; i<m_p; i++)
              {
                for(uint32_t j=0; j<m_q; j++)
                {
                  if(Lpattern[i][j].compare("1") == 0 )
                  {
                    L(i,j) = 1.0;
                    m_lFixedOneIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                  }
                  
                  if(Lpattern[i][j].compare("?") == 0 )
                  {
                    //L(i,j) = (normSigma.Rvs());
                    L(i,j) = 0;
                    m_lFreeIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                  }
                  if(m_params.m_sigma_cross_loadings>0 &&  Lpattern[i][j].compare("cl") == 0 )
                  {
                    //L(i,j) = (normCross.Rvs());
                    L(i,j) = 0;
                    m_lCrossLoadingIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                  }
                }
              }
            }   
            
            
                void InitGibbs0(std::vector<double>& psi_diag, arma::mat& L, arma::mat& Phi, arma::mat& W)
                {
                    distributions::InverseWishart iw_dist(m_params.m_A,m_params.m_v);
                    Phi = iw_dist.Rvs();
                    psi_diag = std::vector<double>(m_p,1.0);
                    distributions::InverseGamma2 ig_dist(m_params.m_alpha_prior, m_params.m_beta_prior);
                    for(uint32_t i=0; i<m_p; i++)
                    {
                        psi_diag[i] = ig_dist.Rvs();
                    }
                    
                   
                    W.zeros(m_n,m_q);
                    distributions::MVNormal mv_dist(arma::zeros(m_q),Phi);
                    for(uint32_t i=0; i<m_n; i++)
                    {
                        arma::vec row = mv_dist.Rvs();
                        W.row(i) = row.t();
                    }
                    
                    std::vector<std::vector<std::string>> Lpattern =  m_input.GetFactorLoadingMatrixPattern();
                    distributions::Normal normSigma(0,m_params.m_sigma);
                    distributions::Normal normCross;
                    if(m_params.m_sigma_cross_loadings>0)
                    {
                        normCross = distributions::Normal(0,m_params.m_sigma_cross_loadings);
                    }
                    
                    L.zeros(m_p,m_q);
                    
                    for(uint32_t i=0; i<m_p; i++)
                    {
                        for(uint32_t j=0; j<m_q; j++)
                        {
                            if(Lpattern[i][j].compare("1") == 0 )
                            {
                                L(i,j) = 1.0;
                                m_lFixedOneIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                            }
                            
                            if(Lpattern[i][j].compare("?") == 0 )
                            {
                                L(i,j) = (normSigma.Rvs());
                                m_lFreeIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                            }
                            if(m_params.m_sigma_cross_loadings>0 &&  Lpattern[i][j].compare("cl") == 0 )
                            {
                                L(i,j) = (normCross.Rvs());
                                m_lCrossLoadingIndices.push_back(std::pair<uint32_t, uint32_t>(i,j));
                            }
                        }
                    }
                }

                void CreatePosteriorVarNames()
                {
                    
                    // m_p for std::vector<double> psi_diag;
                    // m_p * m_q for L
                    // m_q * m_q for Phi
                    // m_n * m_q for W
                    
                    for(uint32_t i=1; i<=m_p; i++)
                    {
                        std::string psi_i = "psi["+ std::to_string(i) +"]";
                        m_posterior_mcmc_names.push_back(psi_i);
                    }
                    for(uint32_t j=1; j<=m_q; j++)
                    {
                        for(uint32_t i=1; i<=m_p; i++)
                        {
                            std::string L_ij = "L["+ std::to_string(i) +"][" +  std::to_string(j) +"]";
                            m_posterior_mcmc_names.push_back(L_ij);
                        }
                    }
                    
                    for(uint32_t j=1; j<=m_q; j++)
                    {
                        for(uint32_t i=1; i<=m_q; i++)
                        {
                            std::string Phi_ij = "Phi["+ std::to_string(i) +"][" +  std::to_string(j) +"]";
                            m_posterior_mcmc_names.push_back(Phi_ij);
                        }
                    }

                    m_posterior_mcmc_names_full = m_posterior_mcmc_names;
                    m_posterior_mcmc_latent.clear();
                    for(uint32_t i=1; i<=m_n; i++)
                    {
                        for(uint32_t j=1; j<=m_q; j++)
                        {
                           
                            std::string W_ij = "W["+ std::to_string(i) +"][" +  std::to_string(j) +"]";
                            m_posterior_mcmc_names_full.push_back(W_ij);
                            m_posterior_mcmc_latent.push_back(W_ij);
                            //std::cout<<W_ij<<std::endl;
                        }
                    }
                }
                
            
            double LogPrior(std::vector<double>& psi_diag, arma::mat& Phi, arma::mat& W,
                              arma::mat& L, std::vector<std::pair<uint32_t, uint32_t>>& lFreeIndices,
                                std::vector<std::pair<uint32_t, uint32_t>>& lCrossLoadingIndices)
            {
                double llkl = 0;
                llkl += LogPriorPsi( psi_diag);
                
                llkl += LogPriorPhi(Phi);
                
                llkl += LogPriorW(W);
                
                llkl += LogPriorL(L, lFreeIndices, m_params.m_sigma);
                if(lCrossLoadingIndices.size()>0)
                {
                    llkl += LogPriorL(L, lCrossLoadingIndices, m_params.m_sigma_cross_loadings);
                }
                return llkl;
            }
    
            double LogPriorPsi(const std::vector<double>& psi_diag)
            {
                double llkl = 0;
                for(uint32_t k=0; k<m_p; k++)
                {
                    distributions::InverseGamma2 dist(m_params.m_alpha_prior, m_params.m_beta_prior);
                    llkl+=dist.LogProbDist(psi_diag[k]);
                }
                return llkl;
            }
        
            double LogPriorPhi(arma::mat& Phi)
            {
                distributions::InverseWishart iw_dist(m_params.m_A,m_params.m_v);
                return iw_dist.LogProbDist(Phi);
            }
            
            double LogPriorW(arma::mat& W)
            {
                arma::mat I = arma::eye(m_q,m_q);
                distributions::MVNormal mv_dist(arma::zeros(m_q),I);
                
                double llkl = 0;
                for(uint32_t i=0; i<m_n; i++)
                {
                     arma::vec v = W.row(i).t();
                     std::vector<double> row = utilities::VecToStd(v);
                     llkl+=mv_dist.LogProbDist(row);
                }
               
                return llkl;
            }
            
            double LogPriorL(arma::mat& L, std::vector<std::pair<uint32_t, uint32_t>>& ids, double sigma)
            {
                 double llkl = 0;
                 distributions::Normal normSigma(0,sigma);
                 for(uint32_t id=0; id<ids.size(); id++)
                 {
                    std::pair<uint32_t,uint32_t> tmp = ids[id];
                    uint32_t k = tmp.first;
                    uint32_t j = tmp.second;
                    llkl+=normSigma.LogProbDist(L(k,j));
                 }
                
                return llkl;
            }
            
            double LogPLikl(std::vector<double>&psi_diag, arma::mat& W,arma::mat& L, arma::mat& y)
            {
                double llikl = 0;
                for(uint32_t i=0; i<m_n; i++)
                {   
                    arma::vec mu = L*W.row(i).t();
                    for(uint32_t k=0; k<m_p; k++)
                    {
                        distributions::Normal dist(mu(k), std::sqrt(psi_diag[k]));
                        llikl += dist.LogProbDist(y(i,k));
                    }
                }
                return llikl;
            }
            
              double LogLiklFromRow(arma::vec row)
            {
                arma::vec psi;
                arma::mat L, Phi,  W;
                VectorToDataStructure(row,  psi,  L,  Phi,  W);
                
               // cout<<row<<endl;
                std::vector<double> psi_diag = utilities::VecToStd(psi);
                double logPrior = LogPrior(psi_diag, Phi, W, L, m_lFreeIndices, m_lCrossLoadingIndices);
                double logLikl = LogPLikl(psi_diag, W,L, m_y); 
                
                return logPrior+logLikl;
            }
            
            void VectorToDataStructure(arma::vec& row, arma::vec& psi, arma::mat& L, arma::mat& Phi, arma::mat& W)
            {
                psi =  arma::zeros(m_p);
                for(uint32_t id=0;id<m_p; id++)
                {
                    psi(id) = row(id);
                }
                L = arma::zeros(m_p,m_q);
                
                uint32_t index = m_p;
                for(uint32_t id=0;id<m_lFreeIndices.size(); id++)
                {
                    L(m_lFreeIndices[id].first,m_lFreeIndices[id].second) = row(index);
                    index++;
                }
                for(uint32_t id=0;id<m_lCrossLoadingIndices.size(); id++)
                {
                    L(m_lCrossLoadingIndices[id].first,m_lCrossLoadingIndices[id].second) = row(index);
                    index++;
                }
                for(uint32_t id=0;id<m_lFixedOneIndices.size(); id++)
                {
                    L(m_lFixedOneIndices[id].first,m_lFixedOneIndices[id].second) = 1.0;
                }
                
                arma::vec PhiVec = row.subvec(index,index + m_q*m_q-1);
                Phi = arma::reshape(PhiVec,m_q,m_q);
                
                arma::vec WVec = row.subvec(index + m_q*m_q,row.size()-1);
                W = arma::reshape(WVec,m_n,m_q);                
            }           
           

    };
}
#endif // NORMCFA_HPP
