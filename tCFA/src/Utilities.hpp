#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include  <cstdio>
#include  <vector>
#include  <string>
#include  <armadillo>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60



namespace utilities
{
    
    void PrintProgressBar(double percentage) 
    {
        int val = (int) (percentage * 100);
        int lpad = (int) (percentage * PBWIDTH);
        int rpad = PBWIDTH - lpad;
        printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
        fflush(stdout);
    }

    ////////////////////////////////////////////////////////////////////
    // using typedefs for flexibility
    ////////////////////////////////////////////////////////////////////
    typedef arma::wall_clock Timer; 
  

    //////////////////////////////////////////////////////////////////////
    std::vector<double> VecToStd(arma::vec& vect)
    {
        std::vector<double> res = arma::conv_to< std::vector<double> >::from(vect);
        return res;
    }

    arma::vec  StdToVec(std::vector<double>& vect)
    {
        arma::vec res = arma::conv_to< arma::vec >::from(vect);
        return res;
    }

    std::vector<std::vector<double>> MatToStd(arma::mat& matrix)
    {
        std::vector<std::vector<double>> res;
        for(uint32_t i=0; i<matrix.n_rows; i++)
        {
            std::vector<double> row = arma::conv_to< std::vector<double> >::from(matrix.row(i));
            res.push_back(row);
        }
        return res;
    }

    arma::mat  StdToMat(std::vector<std::vector<double>>& matrix)
    {
        uint32_t n_rows = matrix.size();
        uint32_t n_cols = matrix[0].size();
        
        arma::mat res(n_rows,n_cols);
        for(uint32_t i=0; i<n_rows; i++)
        {
            res.row(i) = arma::conv_to< arma::rowvec >::from(matrix[i]);
        }
        return res;
    }
    
     void PrintStdVector(std::vector<uint32_t>& vect)
    {
        std::cout<<"[";
        for(uint32_t i=0; i<vect.size(); i++)
            std::cout<<vect[i]<<" ";
        std::cout<<"]"<<std::endl;
    }

    void PrintStdVector(std::vector<double>& vect)
    {
        std::cout<<"[";
        for(uint32_t i=0; i<vect.size(); i++)
            std::cout<<vect[i]<<" ";
        std::cout<<"]"<<std::endl;
    }
    
    void PrintStdVector(std::vector<std::string>& vect)
    {
        std::cout<<"[";
        for(uint32_t i=0; i<vect.size(); i++)
            std::cout<<vect[i]<<" ";
        std::cout<<"]"<<std::endl;
    }
    
    
    void PrintStdMatrix(std::vector<std::vector<double>>& matrx)
    {
        for(uint32_t i=0; i<matrx.size(); i++)
        {
            PrintStdVector(matrx[i]);
        }
    }
    
    void PrintStdMatrix(std::vector<std::vector<std::string>>& matrx)
    {
        for(uint32_t i=0; i<matrx.size(); i++)
        {
            PrintStdVector(matrx[i]);
        }
    }

}

#endif // UTILITIES_HPP
