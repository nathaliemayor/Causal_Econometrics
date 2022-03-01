"""
Data Analytics II: Self Study.

Spring Semester 2021.

@author: nathaliemayor

University of St. Gallen.
"""

# Data Analytics II: Self Study
# Main code

# import modules 
import sys
import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import statsmodels.api as sm
import os 

# set working directory
PATH = os.getcwd()
sys.path.append(PATH)

# Set Seed
seed(1234)

# load the functions
import SS_functions as SS


# setting the parameters for the 1st and 3rd DGP
mean = [10, 0]                          # mean vector   
cov = [[15, 0], [0, 5]]                 # covariance matrix 
betas = np.array([200,4400,1000,300])   # vecotr of betas
u_sd = 100                              # sd of the noise
num_simulations = 1000                  # number of simulations
sample_size=10000                       # sample size


# simulation for the 1st DGP
SS.simulation1(num_simulations,betas,mean,cov,u_sd,sample_size)

# simulation for the 3rd DGP
SS.simulation3(num_simulations,betas,u_sd,sample_size)

# setting the parameters for the 2nd DGP
betas = np.array([5000,100,1000,300])
u_sd = 2000
num_simulations = 1000
sample_size=10000


seed(1234)
# simulation for the 2nd DGP
SS.simulation2(num_simulations,betas,u_sd,sample_size)
