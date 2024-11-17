library(tCFA)

cfa_model <- 'visual  =~ x1 + x2 + x3
                textual =~ x4 + x5 + x6
                speed   =~ x7 + x8 + x9'

# sample size
burnsz <- 1000
samplesz <- 5000

# prior parameters
alpha_prior = 1
beta_prior = 1
sigma_prior = 10
max_t_degree_freedom_prior = 100

inputFile = "HolzingerSwineford1939data.csv"

# mcmc sampling step
tCFA::NormalCFA(inputFile,modelString = cfa_model,
                mcmcOutFilePath = "1N.csv", mcmcLatentOutFilePath = "W1N.csv",
                alpha_prior = alpha_prior,beta_prior = beta_prior, 
                sigma_prior = sigma_prior, burnin = burnsz, samples = samplesz, 
                seed = 42)

# marginal likelihood estimation
tCFA::NormalCalculateMargLikl(inputFile,modelString = cfa_model,
                              mcmcOutFilePath = "1N.csv", mcmcLatentOutFilePath = "W1N.csv",
                              alpha_prior = alpha_prior,beta_prior = beta_prior,
                              sigma_prior = sigma_prior)

# remove temporary files
file.remove("1N.csv")
file.remove("W1N.csv")

# mcmc sampling step
tCFA::StudentCFA(inputFile,modelString = cfa_model,
                 mcmcOutFilePath = "1T.csv", mcmcLatentOutFilePath = "W1T.csv",
                 alpha_prior = alpha_prior,beta_prior = beta_prior,
                 sigma_prior = sigma_prior, burnin = burnsz, samples = samplesz,
                 max_t_degree_freedom_prior = max_t_degree_freedom_prior)

# marginal likelihood estimation
tCFA::StudentCalculateMargLikl(inputFile,modelString = cfa_model,
                               mcmcOutFilePath = "1T.csv", mcmcLatentOutFilePath = "W1T.csv",
                               alpha_prior = alpha_prior,beta_prior = beta_prior,
                               sigma_prior = sigma_prior, nu_max = max_t_degree_freedom_prior)

# remove temporary files
file.remove("1T.csv")
file.remove("W1T.csv")
