#ifndef INPUTREADER_HPP
#define INPUTREADER_HPP

#include <iostream>
#include <vector>
#include <string>
#include "Utilities.hpp"
#include <map>
#include <armadillo>
#include<boost/tokenizer.hpp>

namespace inputreader
{
    
    class InputReader
    {
        private:
            uint32_t m_q = 0; //# of factors
            uint32_t m_p = 0; //# of unique variables
            uint32_t m_n = 0; //# of unique variables
                
            std::vector<std::string> m_var_order;
            std::vector<std::string> m_factor_order;
            std::vector<std::vector<std::string>> m_L_pattern;
            
            std::map<std::string, uint32_t> m_name_to_column_map;
            std::vector<std::string> m_col_names;
            std::vector<std::vector<double>> m_csv_data;
            arma::mat m_data;
            
         
        public:
            InputReader(std::string modelStr, std::string dataFName )
            {
                SyntaxRead( modelStr, m_q, m_p, m_L_pattern, m_var_order,  m_factor_order );
                DataReadFromCsv(dataFName, m_col_names, m_name_to_column_map, m_csv_data);
                
                // fix the columns, elliminate redundant
                arma::mat dataTmp = utilities::StdToMat(m_csv_data);
                
                
                m_data = arma::zeros(m_n, m_p);
                
                for(uint32_t i=0; i<m_var_order.size(); i++)
                {
                    uint32_t id = m_name_to_column_map[m_var_order[i]];
                    m_data.col(i) = dataTmp.col(id);
                }
                /*
                for(uint32_t i=0; i<m_data.n_cols; i++)
                {
                    double mean = arma::mean(m_data.col(i));
                    m_data.col(i) = m_data.col(i)-mean;
                }
                 */
            }
            
            arma::mat& GetData()
            {
                return m_data;
            } 
            
            std::vector<std::string>& GetColumnNames()
            {
                return m_var_order;
            }
      
   
            std::vector<std::vector<std::string>>& GetFactorLoadingMatrixPattern()
            {
                return m_L_pattern;
            }   

            uint32_t GetNumFactorsQ()
            {
                return m_q;
            }
    
            uint32_t GetNumOfPeopleN()
            {
                return m_n;
            }
    
            uint32_t GetNumVariablesP()
            {
                return m_p;
            }

    
        private:
            void DataReadFromCsv(std::string fName, std::vector<std::string>& col_names, 
                    std::map<std::string, uint32_t>& name_to_column_map , std::vector<std::vector<double>>& csv_data)
            {
                 std::string line;
                 std::ifstream myfile;
                 myfile.open (fName);
                
                 boost::char_separator<char> separators(";, ");
                
                 int linenum = 0;
                 if(myfile.is_open())
                 {
                    while(std::getline(myfile, line))
                    {	
                        boost::tokenizer< boost::char_separator<char> > tok(line,separators);
                        if(0==linenum)
                        { 
                            uint32_t varnum = 0;
                            for(boost::tokenizer<boost::char_separator<char>>::iterator beg=tok.begin(); beg!=tok.end();++beg)
                            {
                                col_names.push_back(*beg);
                                name_to_column_map[*beg] = varnum;
                                varnum++;
                            }
                        }
                        else
                        {
                            std::vector<double> row(col_names.size(),0);
                            uint32_t ind = 0;
                            for(boost::tokenizer<boost::char_separator<char>>::iterator beg=tok.begin(); beg!=tok.end();++beg)
                            {
                                row[ind] = std::stod((*beg));
                                ind++;
                            }
                            csv_data.push_back(row);
                        }
                        linenum++;
                    } 
                
                    myfile.close();
                    
                    m_n = csv_data.size();
                }
                else
                {
                    std::cout<<"bad file name"<<std::endl;   
                }   
            }
            void SyntaxRead(std::string modelStr, uint32_t& q, uint32_t& p, std::vector<std::vector<std::string>>& L_pattern,
                   std::vector<std::string>& var_order,  std::vector<std::string>& factor_order)
            {
                q = 0; //# of factors
                p = 0; //# of unique variables
            
                std::map<std::string, int> unique_vars;
                std::map<std::string, int> unique_factors;
                std::map<std::string, std::vector<std::string>> L_pattern_arr; 
            
                boost::char_separator<char> separatorsnewline("'\n'");
                boost::char_separator<char> separators("'=~','\n','+' ");
            
                boost::tokenizer< boost::char_separator<char> > tokln(modelStr,separatorsnewline);
                for(boost::tokenizer<boost::char_separator<char>>::iterator beg=tokln.begin(); beg!=tokln.end();++beg)
                {
                    std::string line = (*beg);
                    q++;
                    boost::tokenizer< boost::char_separator<char> > tok(line,separators);
                    uint32_t counter = 0;
                    std::string factorName = "";
                    for(boost::tokenizer<boost::char_separator<char>>::iterator beg=tok.begin(); beg!=tok.end();++beg)
                    {
                        if(counter>0){
                            if (unique_vars.count((*beg)) == 0){
                                unique_vars.insert(std::pair<std::string, int>((*beg), 1));
                                p++;
                                var_order.push_back((*beg));
                                L_pattern_arr[factorName].push_back((*beg));
                            }
                        }
                        else{
                            unique_factors.insert(std::pair<std::string, int>((*beg), 1));
                            factor_order.push_back(*beg);
                            L_pattern_arr.insert(std::pair<std::string, std::vector<std::string>>((*beg), std::vector<std::string>() ));
                            factorName = *beg;
                        }
                        
                        //cout<< counter<<" - "<<(*beg)<<" - "<<endl;
                        counter++;
                    }
                    
                }
                
                for(uint32_t i =0; i<factor_order.size(); i++)
                {
                    std::vector<std::string> fVars = L_pattern_arr[factor_order[i]];
                    for(uint32_t j =0; j<fVars.size(); j++)
                    {
                        std::vector<std::string> row(q,"?");
                        if(0==j)
                        {
                            row[i] = "1";
                        }
                        for(uint32_t k=0; k<i; k++)
                             row[k] = "cl";
                        for(uint32_t k=i+1; k<factor_order.size(); k++)
                             row[k] = "cl";
                        L_pattern.push_back(row);
                    }
                }
            }
        
    };
}

#endif // INPUTREADER_HPP
