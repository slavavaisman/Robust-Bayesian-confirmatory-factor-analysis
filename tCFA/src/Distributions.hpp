#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include <iostream>
#include <mutex>
#include <random>
#include <vector>
#include <stdexcept>
#include <armadillo>

namespace distributions
{
     double LogMultivariateGamma(uint32_t p,double nu)
     {
        double pp = (double)p;
        double logMV_g = log(M_PI)*(pp*(pp-1.0))/4.0;
        for(uint32_t j=1; j<=p; j++)
        {
            double jj = (double)j;
            logMV_g = logMV_g  + lgamma( nu + ((1.0-jj)/2.0) );
        }    
        return logMV_g;
     }
                
    /************************************************************************/
    //typedef std::mt19937_64 generator;
    typedef std::mt19937_64 generator;
    
    class RandomState
    {
    private:
        static RandomState* pinstance_;
        static std::mutex mutex_;
        
        
      
    protected:
        ~RandomState()
        {}
      
        RandomState(uint64_t seed)
        {
            if(seed<0)
            {
                throw std::invalid_argument("random seed should be non-negative");
            }
            
            if(0==seed){  
                m_seed = std::random_device{}();
            }
            else{
                m_seed = seed;
            }
            m_e.seed(m_seed);
            
            //arma::arma_rng::set_seed(m_seed);
        }
        
        
        uint64_t m_seed;
        generator m_e;
        
        public:

             RandomState(RandomState& other) = delete;
             
             void operator=(const RandomState&) = delete;
             
             static RandomState* GetInstance(uint64_t seed=0);

             generator& GetGenerator()
             {
                return m_e;
             }
    };  

    distributions::RandomState* distributions::RandomState::pinstance_{ nullptr };
    std::mutex distributions::RandomState::mutex_;
        
    distributions::RandomState* distributions::RandomState::GetInstance(uint64_t seed)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pinstance_ == nullptr)
        {
            pinstance_ = new RandomState(seed);
        }
        return pinstance_;
    }


    /************************************************************************/
    class BaseDistribution
{
    protected:
       
        BaseDistribution(uint64_t seed = 0)
        {
            m_rnd_state = RandomState::GetInstance(seed);
            m_rnd = &m_rnd_state->GetGenerator();
        }
        ~BaseDistribution(){}
        
        long double LogBeta(double a, double b)
        {
            return lgamma(a) + lgamma(b) - lgamma(a+b);
        }
        
        long double  LogBinomial(double n, double k)
        {
            return lgamma(n+1.0) - (lgamma(k+1.0) + lgamma(n-k+1.0));
        }
        
        long double  LogFactorial(uint32_t n)
        {
            long double res = 0;
            if(n<=1)
                return 0.0;
                
            for(uint32_t i=2; i<=n; i++)
                res+=log(i);
            return res;
        }
        
        RandomState* m_rnd_state;
        generator* m_rnd;
    };

    class UVContinuousDistribution: public distributions::BaseDistribution
    {
        public:
            virtual double LogProbDist(double x) = 0;
            virtual double Rvs() = 0;
            virtual double GetExpectation() = 0;
            virtual double GetVariance() = 0;
    };

    class UVDiscreteDistribution: public distributions::BaseDistribution
    {
        public:
            virtual double LogProbDist(int x) = 0;
            virtual int Rvs() = 0;
            virtual double GetExpectation() = 0;
            virtual double GetVariance() = 0;
    };

    template <class T>
    class MVDistribution: public distributions::BaseDistribution {
      public:
         virtual double LogProbDist(T x) = 0;
         virtual T Rvs() = 0;
    };
    
    /************************************************************************/
    
    //////////////////////////////////////////////////////////////////////////////////////v
    // continuous Uniform distribution
    //////////////////////////////////////////////////////////////////////////////////////v
    class Uniform: public UVContinuousDistribution 
    {
        private:
            std::uniform_real_distribution<> m_dis;
            double m_low;
            double m_high; 
           
        public:
            Uniform(double low=0, double high=1)
            {
                if(high<=low)
                {
                    throw std::invalid_argument("invalid distribution argument");
                }
                m_low = low;
                m_high = high; 
                m_dis = std::uniform_real_distribution<>(low,high);
            }

            ~Uniform()
            {
            }
            
            double GetExpectation()
            {
                return (m_high + m_low) /2;
        
            }
            double GetVariance()
            {
                return (1.0/12)*pow(m_high - m_low,2.0);
            }
            
            double LogProbDist(double x)
            {
                 if(x<m_low || x>m_high)
                     return -INFINITY;
                 return -log(m_high-m_low);
            }
            double Rvs()
            {
                return (m_dis)(*m_rnd);
            }
        };
        
         //////////////////////////////////////////////////////////////////////////////////////
        // Studentst distribution: mu, sigma (sigma is scale)
        //////////////////////////////////////////////////////////////////////////////////////
        class Studentst: public UVContinuousDistribution 
        {
            private:
                std::student_t_distribution<> m_dis;
                double m_mu;
                double m_sigma;
                double m_nu;
           
            public:
                Studentst(double nu, double mu, double sigma)
                {
                    if(sigma<=0 || nu<=0)
                    {
                        throw std::invalid_argument("invalid distribution argument");
                    }
                    m_mu = mu;
                    m_sigma = sigma;
                    m_nu = nu;
                    m_dis = std::student_t_distribution<>(m_nu);
                }
                
                ~Studentst()
                {
                }
                
                double GetExpectation()
                {
                    return m_mu;
            
                }
                double GetVariance()
                {
                    return pow(m_sigma,2.0)*(m_nu/(m_nu-2.0));
                }
                
                double LogProbDist(double x)
                {
                     return lgamma((m_nu+1.0)/2.0) -  lgamma(m_nu/2.0) -0.5*log(m_nu*M_PI) 
                            - log(m_sigma) - ((m_nu+1.0)/2.0)*log(1.0 + (1.0/m_nu)*pow(((x-m_mu)/m_sigma),2.0) );   
                }
                double Rvs()
                {
                    return m_mu + m_sigma*(m_dis)(*m_rnd);
                }
        };
        
        //////////////////////////////////////////////////////////////////////////////////////
        // Gamma_1: alpha, beta (exp(-x/beta))
        //////////////////////////////////////////////////////////////////////////////////////
        class Gamma1: public UVContinuousDistribution 
        {
            private:
                std::gamma_distribution<> m_dis;
                double m_alpha;
                double m_beta;
            
            public:
                Gamma1(double alpha, double beta)
                {
                    if(alpha<=0 || beta<=0)
                    {
                        throw std::invalid_argument("invalid distribution argument");
                    }
                    m_alpha = alpha;
                    m_beta = beta;
                    m_dis = std::gamma_distribution<>(m_alpha, m_beta);
                }
                ~Gamma1()
                {
                }
                double GetExpectation()
                {
                    return m_alpha*m_beta;
            
                }
                double GetVariance()
                {
                    return  m_alpha * pow(m_beta,2.0);
                }
                
                double LogProbDist(double x)
                {
                     if(x<0)
                        return -INFINITY;
                     return - lgamma(m_alpha) - m_alpha*log( m_beta ) 
                            + ( m_alpha-1)*log( x ) - x/m_beta;
                }
                double Rvs()
                {
                    return m_dis(*m_rnd);
                }
                
        };
        
        //////////////////////////////////////////////////////////////////////////////////////
        // Gamma_2: alpha, beta (exp(-x*beta))
        //////////////////////////////////////////////////////////////////////////////////////
        class Gamma2: public UVContinuousDistribution 
        {
            private:
                std::gamma_distribution<> m_dis;
                double m_alpha;
                double m_beta;
            
            public:
                Gamma2(){}
                Gamma2(double alpha, double beta)
                {
                    if(alpha<=0 || beta<=0)
                    {
                        throw std::invalid_argument("invalid distribution argument");
                    }
                    m_alpha = alpha;
                    m_beta = beta;
                    m_dis = std::gamma_distribution<>(m_alpha, 1.0/m_beta);
                }
                ~Gamma2()
                {
                }
                double GetExpectation()
                {
                    return m_alpha/m_beta;
            
                }
                double GetVariance()
                {
                    return  m_alpha / pow(m_beta,2.0);
                }
                
                double LogProbDist(double x)
                {
                     if(x<0)
                        return -INFINITY;   
                     return - lgamma(m_alpha) + m_alpha*log( m_beta ) 
                            + ( m_alpha-1)*log( x ) - x*m_beta;
                }
                double Rvs()
                {
                    return m_dis(*m_rnd);
                } 
        };
        
          //////////////////////////////////////////////////////////////////////////////////////
        // InverseGamma1: alpha, beta (exp(-beta/x))
        //////////////////////////////////////////////////////////////////////////////////////
        class InverseGamma2: public UVContinuousDistribution 
        {
            private:
                std::gamma_distribution<> m_dis;
                double m_alpha;
                double m_beta;
            
    public:
                InverseGamma2(){}
                InverseGamma2(double alpha, double beta)
                {
                    if(alpha<=0 || beta<=0)
                    {
                        throw std::invalid_argument("invalid distribution argument");
                    }
                    m_alpha = alpha;
                    m_beta = beta;
                    m_dis = std::gamma_distribution<>(m_alpha, 1.0/m_beta);
                }
                ~InverseGamma2()
                {
                 }
                double GetExpectation()
                {
                    return  m_beta/(m_alpha-1);
            
                }
                double GetVariance()
                {
                    return  pow(m_beta,2.0)/(  pow((m_alpha-1.0),2.0) * (m_alpha-2) );  
                }
                
                double LogProbDist(double x)
                {
                     if(x<0)
                        return -INFINITY;
                     return - lgamma(m_alpha) + m_alpha*log( m_beta ) 
                            - ( m_alpha+1)*log( x ) - m_beta/x;
                }
                double Rvs()
                {
                    return 1.0/(m_dis)(*m_rnd);
                }    
        };
        
         //////////////////////////////////////////////////////////////////////////////////////
        // Normal distribution: mu, sigma (sigma is standard deviation)
        //////////////////////////////////////////////////////////////////////////////////////
        class Normal: public UVContinuousDistribution 
        {
            private:
                std::normal_distribution<> m_dis;
                double m_mu;
                double m_sigma;
           
            public:
                Normal()
                {                    
                    m_mu = 0;
                    m_sigma = 1;
                    m_dis = std::normal_distribution<>(0, 1);
                }
                
                Normal(double mu, double sigma)
                {
                    if(sigma<0)
                    {
                        throw std::invalid_argument("invalid distribution argument");
                    }
                    m_mu = mu;
                    m_sigma = sigma;
                    m_dis = std::normal_distribution<>(m_mu, m_sigma);
                }
                
                ~Normal()
                {
                }
                
                double GetExpectation()
                {
                    return m_mu;
            
                }
                double GetVariance()
                {
                    return pow(m_sigma,2.0);
                }
                
                double LogProbDist(double x)
                {
                     if(0 == m_sigma)
                     {
                         if(m_mu == x)
                            return 0;
                         else
                            return -INFINITY;
                     }
                     return -0.5*log(2.0*M_PI*pow(m_sigma,2.0))  -0.5*pow(((x-m_mu)/m_sigma),2.0);
                }
                double Rvs()
                {
                    if(0 == m_sigma)
                        return m_mu;
                        
                    return m_dis(*m_rnd);
                }
        };
        
    class ChiSquared: public UVContinuousDistribution 
        {
            private:
                std::chi_squared_distribution<> m_dis;;
                uint32_t m_nu;
            
            public:
                ChiSquared(){}
                ChiSquared(uint32_t nu)
                {
                    if(nu<=0 )
                    {
                        throw std::invalid_argument("invalid distribution argument");
                    }
                    m_nu = nu;
                    m_dis = std::chi_squared_distribution<>(m_nu);
                }
                
                ~ChiSquared()
                {
                }
                
                double GetExpectation()
                {
                    return m_nu;
            
                }
                double GetVariance()
                {
                    return   2*m_nu;
                }
                
                double LogProbDist(double x)
                {
                     if(x<0)
                        return -INFINITY;
                     return -(m_nu/2.0) *log(2) - lgamma(m_nu/2.0) + (-1 + m_nu/2.0 )*log(x) - x/2.0;
                }
                
                double Rvs()
                {
                    return (m_dis)(*m_rnd);;
                }
        };
            
    //////////////////////////////////////////////////////////////////////////////////////v
    // MV Normal distribution
    //////////////////////////////////////////////////////////////////////////////////////v
    class MVNormal : public MVDistribution< arma::vec >
    {
        private:
            uint32_t m_k;
            arma::colvec m_mu_arm;
            arma::mat m_Sigma_arm;
            arma::mat m_Chol;
            Normal m_norm;
            
            
        public:
        
            MVNormal()
            {
                m_mu_arm= {0};
                m_Sigma_arm = arma::ones(1,1);
                m_k = m_mu_arm.size();
                m_Chol = chol( m_Sigma_arm,"lower" );
                m_norm = Normal(0, 1);
            }
            
                      
            MVNormal(arma::vec mu, arma::mat Sigma)
            {
                
                m_mu_arm = mu;
                m_Sigma_arm = Sigma;
                m_k = m_mu_arm.size();
                m_Chol = chol( m_Sigma_arm,"lower" );
                m_norm = Normal(0, 1);
            }
            
            ~MVNormal()
            {
            }
            
            double LogProbDist(arma::vec X)
            {
                arma::vec tmp = X-m_mu_arm;
                arma::vec tmp2 = tmp.t() * inv(m_Sigma_arm) * tmp;
  
                return -(m_k/2.0)*log(2.0*M_PI) -0.5*log(det(m_Sigma_arm)) - 0.5*tmp2(0,0);
            }
            
                    
            arma::vec Rvs() 
            {
                 arma::vec Z(m_k);
                 for(uint32_t i=0; i<m_k; i++)
                     Z(i) = m_norm.Rvs();
                 
                 Z = m_mu_arm + m_Chol*Z;
                 return Z;   
            }
    };
    
    
    //////////////////////////////////////////////////////////////////////////////////////v
    // Wishart distribution
    //////////////////////////////////////////////////////////////////////////////////////v
    class Wishart : public MVDistribution< arma::mat >
    {
        private:
      
            arma::mat m_A_arm;
            arma::mat m_Chol;
            
            uint32_t m_p;
            double m_nu;
            Normal m_norm;
            ChiSquared m_chi;
            
                
        public:
            Wishart() {}
            Wishart(arma::mat& A, double nu)
            {
                m_p = A.n_cols;
                m_nu = nu;
                
                if(m_nu <= m_p-1)
                {
                    throw std::invalid_argument("invalid distribution argument");
                }
                
                m_A_arm = A;
                m_Chol = chol( m_A_arm,"lower" );
                
                m_norm = Normal(0,1);
            }
            ~Wishart()
            {
            }
            
            double LogProbDist(arma::mat X)
            {   
                return ((m_nu - m_p - 1.0)/2.0)*log(det(X)) - ((m_nu*m_p)/2.0)*log(2.0)
                        - ((m_nu)/2.0)*log(det(m_A_arm)) - LogMultivariateGamma(m_p,m_nu/2.0)  -0.5*arma::trace(inv(m_A_arm)*X);
            }
            
            arma::mat Rvs1() 
            {
              arma::mat res = arma::wishrnd(m_A_arm,m_nu);
              return res;   
            }
          
            arma::mat Rvs() 
            {
                 arma::mat C = m_Chol;
                 arma::mat A;
                 A.zeros(m_p,m_p);
                
                 for(uint32_t i=0; i<m_p;i++)
                 {
                    for(uint32_t j=0; j<=i;j++)
                    { 
                        if(i == j)
                        {
                            m_chi = ChiSquared(m_nu - (i+1) +1 );
                            A(i,i) = pow(m_chi.Rvs(),0.5);
                        }
                        else
                        {
                            A(i,j) = m_norm.Rvs();
                        }
                    }
                 }
                 arma::mat Res=C*A*A.t()*C.t();
                 return Res;   
            }             
    };
    
    
    /////////////////////////////////////////////////////////////////////////////////////v
    // InverseWishart distribution
    //////////////////////////////////////////////////////////////////////////////////////v
    class InverseWishart : public MVDistribution< arma::mat >
    {
        private:
              std::vector<std::vector<double>> m_A;
              arma::mat m_A_arm;
 
                
              uint32_t m_p;
              double m_nu;
              Wishart m_wish;
              
              
              
    public:
            InverseWishart(){}
            InverseWishart(arma::mat A, double nu)
            {
                m_p = A.n_rows;
                m_nu = nu;
                m_A_arm = A;
                if(m_nu <= m_p-1)
                {
                     throw std::invalid_argument("invalid distribution argument");
                }
                arma::mat A_inv = inv(m_A_arm);  
                m_wish = Wishart(A_inv,m_nu);
                 
            }
          
            ~InverseWishart()
            {
            }
            
            double LogProbDist(arma::mat X)
            {
                 double pp = (double)m_p;
                 return -((m_nu + pp + 1.0)/2.0)*log(det(X)) - ((m_nu*pp)/2.0)*log(2.0)
                        + (m_nu/2.0)*log(det(m_A_arm)) - LogMultivariateGamma(m_p,m_nu/2.0) - 0.5*trace(m_A_arm*inv(X));
            }
            
            arma::mat Rvs1() 
            {
              arma::mat res = arma::iwishrnd(m_A_arm,m_nu);
              return res;   
            }
      
            arma::mat Rvs() 
            {
                 arma::mat res = m_wish.Rvs();
                 res = inv(res);
                 return res;   
            }
            
            
    };
}

#endif // DISTRIBUTIONS_HPP
