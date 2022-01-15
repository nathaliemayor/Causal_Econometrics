"""
Data Analytics II: Self Study.

Spring Semester 2021.

@author: nathaliemayor

University of St. Gallen.
"""
# Data Analytics II: Self Study
# Functions file


# import modules
import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm


# 3 Data Generating Processes
def dgp1(b,mean,cov,u_sd,n):
    """ Creates one draw of a DGP with multivariate normal covariates
        and normal noise
    Input:
        - b: vector of true betas values including the intercept
        - mean: vector of mean 
        - cov: covariance matrix 
        - u_sd: standard deviation of the noise
        - n: the sample size
    Output:
        - treat: treatment vector     
        - x: covariate matrix
        - y: outcome vector
    """
    # define the matrix of exogenous variables
    x = multivariate_normal(mean, cov, n)
    # define the vector of the treatment vector
    treat = np.random.randint(2, size=n)
    # define the vecore of outcomes
    y = b[0] + x @ b[1:3]+treat * b[3] + np.random.normal(0,u_sd,n)
    # gives the tretment, covariates and outcomes as a result
    return(treat,x,y)
    
    
def dgp2(b,u_sd,n):
    """ Creates one draw of a DGP with multivariate normal covariates
        and normal noise
    Input:
        - b: vector of true betas values including the intercept
        - u_sd: standard deviation the noise
        - n: the sample size
    Output:
        - treat: treatement vector
        - x: covariate vector
        - y: outcome vector
    """
    # set m as a ratio of the sample size 
    m=int(n/100)
    # define the vector of exogenous variables
    x=pd.Series([3]*m*30+[2]*m*70)
    # define the vector of the treatment vector
    treat=pd.Series([1]*25*m+[0]*5*m+[1]*10*m+[0]*60*m)
    # define the vecore of outcomes
    y=b[0] + x * b[2]+ treat * b[3] + np.random.normal(0,u_sd,n)
    # gives the tretment, covariates and outcomes as a result
    return(treat,x,y)
    

    
def dgp3(b,u_sd,n):
    """ Creates one draw of a DGP with multivariate normal covariates
        and normal noise
    Input:
        - b: vector of true betas values including the intercept
        - u_sd: standard deviation the noise
         - n: the sample size
    Output:
        - treat: treatement vector
        - x: covariate vector
        - y: outcome vector
        - z: the omitted variable
    """
    # define the vector of exogenous variables
    x = np.random.random(n)*300
    # define the vector of the omitted variable z
    z = np.random.random(n)*500
    # define the vecore of treatments
    treat = np.random.randint(2, size=n)
    # define the vecore of outcomes
    y = b[0] + x * b[2]+treat * b[3] + np.random.normal(0,u_sd,n)+60*z
    return(treat,x,y,z)
    
    

# 2 Estimators OLS and IPW
    
# Function of the OLS estimator
def ols(x,y):
    """ OLS estimator
    Input: 
        x: the matrix of covariates
        y: the vector of outcomes
    Output: 
        betas coefficents estimates
    """
    # get the sample size
    n = y.shape[0]          
    x_c = np.c_[np.ones(n),x] 
    # estimating the beta coefficients
    b = np.linalg.inv(x_c.T @ x_c) @ x_c.T @ y 
    # produce the result of estimations
    return(b)


### Function for the Inverse Probability Weiting Estimator
def IPW(exog, outcome, treat, replications):
    ''' IPW estimator.
    Input:
        covariates: independant variables (x) : pd.Series or pd.DataFrame
        outcome: outcome (y): pd.Series
        treat:  treatment (D): pd.Series
        replications: number of replications
    Output:
        Returns a table with the ATE estimate, SE and t-value
    '''
    # getting the propensity scores
    propscores = sm.Logit(endog=treat, exog=sm.add_constant(exog)).fit(disp=0).predict()
    # getting the average treatment effect using the propensity scores
    ate = np.mean((treat * outcome) / propscores - ((1 - treat) * outcome) / (1 - propscores))
    # getting the estimator SE via bootstrap
    # creating an empty array to store the results 
    atebts = {}
    # using the loop to replicate the computations
    for rep in range(replications):
    # random samples selections, with replacement, using random choice
        samplebts = np.random.choice(exog.index, size=exog.shape[0],
                                       replace=True)
    # bootstrap samples for the oucome and treatment  
        treatbts = treat.loc[samplebts]
        outcome_boot = outcome.loc[samplebts]
    # ate estimation via ipw passing in the bootstrap sample
    # saving the new ate
        atebts[rep] = np. mean((treatbts * outcome_boot) / propscores - 
                                   ((1 - treatbts) * outcome_boot) / (1 - propscores))
    # getting the ATE standard error via the bootstrap values obtained
    ate_se = np.std(list(atebts.values()))
    final = pd.DataFrame([ate, ate_se],
                          index=['coef', 'se'],
                          columns=[treat.name]).transpose()
    return final


## Simulations and measures of performance (BIAS, VARIANCE, MSE)
# Simulations
    
def simulation1(n_sim,b,mean,cov,u_sd,n):
    """ Runs a simulation and returns coverage rate for beta1
    with homogeneous and robust SE
    Input:
        - n_sim: number of draws
        - b: vector  of true beta values 
        - mean: vector (1D-array) of mean of multivariate normal
        - cov: covariance matrix 
        - u_sd: standard deviation of the noise
    Output:
        - Histograms of OLS and IPW estimates
        - Bias, Variance and MSE of OLS and IPW estimates
    """
    all_results = np.empty( (n_sim, 2) )
    # looping for each simulations
    for i in range(n_sim):
    # define the treatments, x and outcome variable with the DGP
        treat,x,y = dgp1(b,mean,cov,u_sd,n)
        # define the covariates (x and treatment for OLS)
        covariates=pd.concat([pd.Series(treat), pd.DataFrame(x)], axis=1)
        # get the ols results    
        ols1 = ols(covariates,pd.Series(y))
        # put the OLS results for the ATE into a vector
        all_results[i,0] = ols1[1]
        # get the IPW estimate of the ATE 
        IPW1 = IPW(pd.DataFrame(x),pd.Series(y),pd.Series(treat),100).coef[0]
        # put the resluts of the IPW estimate of the ATE into a vector
        all_results[i,1] = IPW1
    # compute the OLS bias
    bias_OLS = np.mean(all_results[:,0]) - b[3]
    # compute the OLS variance
    variance_OLS = np.mean( (all_results[:,0] - np.mean(all_results[:,0]))**2 )
    # compute the OLS MSE
    mse_OLS = bias_OLS**2 + variance_OLS
    # compute the IPW bias
    bias_IPW = np.mean(all_results[:,1]) - b[3]
    # compute the IPW variance
    variance_IPW = np.mean( (all_results[:,1] - np.mean(all_results[:,1]))**2 )
    # compute the IPW MSE
    mse_IPW = bias_IPW**2 + variance_IPW
    # plot the histogram for the OLS estimates 
    plt.figure()
    plt.hist(x=all_results[:,0], bins=100, color='purple',alpha=0.5,label="ATE OLS estimates")
    plt.axvline(x=b[3],label="ATE")
    plt.legend(loc='upper right')
    plt.show()
    # plot the histogram for the IPW estimates
    plt.figure()
    plt.hist(x=all_results[:,1], bins=100, color='purple',alpha=0.5,label="ATE IPW estimates")
    plt.axvline(x=b[3],label="ATE")
    plt.legend(loc='upper right')
    plt.show()
    # returns the Bias, Variance and MSE for both OLS and IPW
    return([print("Bias OLS: " + str( round(bias_OLS,3) )),print("Variance OLS: " + str( round(variance_OLS,3) )),print("MSE OLS: " + str( round(mse_OLS,3) ))
    ,print("Bias IPW: " + str( round(bias_IPW,3) )),print("Variance IPW: " + str( round(variance_IPW,3) )),print("MSE IPW: " + str( round(mse_IPW,3) ))])
    
    


    

def simulation2(n_sim,b,u_sd,n):
    """ Runs a simulation and returns coverage rate for beta1
    with homogeneous and robust SE
    Input:
        - n_sim: number of draws
        - b: vector  of true beta values  
        - u_sd: standard deviation of the noise
    Output:
        - Histograms of OLS and IPW estimates
        - Bias, Variance and MSE of OLS and IPW estimates
    """
    all_results = np.empty( (n_sim, 2) )
    # looping for each simulations
    for i in range(n_sim):
        # define the treatments, x and outcome variable with the DGP
        treat,x,y = dgp2(b,u_sd,n)
        # define the covariates (x and treatment for OLS)
        covariates=pd.concat([pd.Series(treat), pd.DataFrame(x)], axis=1)
        # get the ols results   
        ols1 = ols(covariates,pd.Series(y))
        # put the OLS results for the ATE into a vector
        all_results[i,0] = ols1[1]
        # get the IPW estimate of the ATE 
        IPW1 = IPW(pd.DataFrame(x),pd.Series(y),pd.Series(treat),100).coef[0]
        # put the resluts of the IPW estimate of the ATE into a vector
        all_results[i,1] = IPW1
    # compute the OLS bias
    bias_OLS = np.mean(all_results[:,0]) - b[3]
    # compute the OLS variance
    variance_OLS = np.mean( (all_results[:,0] - np.mean(all_results[:,0]))**2 )
    # compute the OLS MSE
    mse_OLS = bias_OLS**2 + variance_OLS
    # compute the IPW bias
    bias_IPW = np.mean(all_results[:,1]) - b[3]
    # compute the IPW variance
    variance_IPW = np.mean( (all_results[:,1] - np.mean(all_results[:,1]))**2 )
    # compute the IPW MSE
    mse_IPW = bias_IPW**2 + variance_IPW
    # plot the histogram for the OLS estimates
    plt.figure()
    plt.hist(x=all_results[:,0], bins=100, color='purple',alpha=0.5,label="ATE OLS estimates")
    plt.axvline(x=b[3],label="ATE")
    plt.legend(loc='upper right')
    plt.show()
    # plot the histogram for the IPW estimates
    plt.figure()
    plt.hist(x=all_results[:,1], bins=100, color='purple',alpha=0.5,label="ATE IPW estimates")
    plt.axvline(x=b[3],label="ATE")
    plt.legend(loc='upper right')
    plt.show()
    # returns the Bias, Variance and MSE for both OLS and IPW
    return([print("Bias OLS: " + str( round(bias_OLS,3) )),print("Variance OLS: " + str( round(variance_OLS,3) )),print("MSE OLS: " + str( round(mse_OLS,3) ))
    ,print("Bias IPW: " + str( round(bias_IPW,3) )),print("Variance IPW: " + str( round(variance_IPW,3) )),print("MSE IPW: " + str( round(mse_IPW,3) ))])
    


def simulation3(n_sim,b,u_sd,n):
    """ Runs a simulation and returns coverage rate for beta1
    with homogeneous and robust SE
    Input:
        - n_sim: number of draws
        - b: vector  of true beta values  
        - u_sd: standard deviation of the noise
    Output:
        - Histograms of OLS and IPW estimates
        - Bias, Variance and MSE of OLS and IPW estimates
    """
    all_results = np.empty( (n_sim, 2) )
    for i in range(n_sim):
        # define the treatments, x and outcome variable and z with the DGP
        treat,x,y,z = dgp3(b,u_sd,n)
        # define the covariates (x and treatment for OLS)
        covariates=pd.concat([pd.Series(treat), pd.DataFrame(x)], axis=1)
        # get the ols results 
        ols1 = ols(covariates,pd.Series(y))
        # put the OLS results for the ATE into a vector
        all_results[i,0] = ols1[1]
        # get the IPW estimate of the ATE 
        IPW1 = IPW(pd.DataFrame(x),pd.Series(y),pd.Series(treat),100).coef[0]
        # put the resluts of the IPW estimate of the ATE into a vector
        all_results[i,1] = IPW1
    # compute the OLS bias
    bias_OLS = np.mean(all_results[:,0]) - b[3]
    # compute the OLS variance
    variance_OLS = np.mean( (all_results[:,0] - np.mean(all_results[:,0]))**2 )
    # compute the OLS MSE
    mse_OLS = bias_OLS**2 + variance_OLS
    # compute the IPW bias
    bias_IPW = np.mean(all_results[:,1]) - b[3]
    # compute the IPW variance
    variance_IPW = np.mean( (all_results[:,1] - np.mean(all_results[:,1]))**2 )
    # compute the IPW MSE
    mse_IPW = bias_IPW**2 + variance_IPW
    # plot the histogram for the OLS estimates
    plt.figure()
    plt.hist(x=all_results[:,0], bins=100, color='purple',alpha=0.5,label="ATE OLS estimates")
    plt.axvline(x=b[3],label="ATE")
    plt.legend(loc='upper right')
    plt.show()
    # plot the histogram for the IPW estimates
    plt.figure()
    plt.hist(x=all_results[:,1], bins=100, color='purple',alpha=0.5,label="ATE IPW estimates")
    plt.axvline(x=b[3],label="ATE")
    plt.legend(loc='upper right')
    plt.show()
    # returns the Bias, Variance and MSE for both OLS and IPW
    return([print("Bias OLS: " + str( round(bias_OLS,3) )),print("Variance OLS: " + str( round(variance_OLS,3) )),print("MSE OLS: " + str( round(mse_OLS,3) ))
    ,print("Bias IPW: " + str( round(bias_IPW,3) )),print("Variance IPW: " + str( round(variance_IPW,3) )),print("MSE IPW: " + str( round(mse_IPW,3) ))])





















