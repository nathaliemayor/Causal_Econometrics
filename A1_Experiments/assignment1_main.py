"""
CE: PC1.

Spring Semester 2021.

University of St. Gallen.
"""

# Data Analytics II: PC Project 1

# import modules here
import sys
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import os 

# set working directory
PATH = os.getcwd()
sys.path.append(PATH)

print(PATH)

# load own functions
import pc1_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc1_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc1.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #
## 1.b

print(data.mean())
print(data.median())
print(data.describe())

# 1.c
    
StatisticsSummary1(data)

# 1.d 
data.drop("age2",inplace=True,axis=1)

# 1.e
histogram(data)
    
    
# 1.f logtransformation of earnings
logdata=data.copy()
logdata['logre78']=np.log(logdata['re78']+1)
logdata.drop("re78",inplace=True,axis=1)

logdata.to_csv(PATH+'inaldata_pc1.csv',index=False)


# 1.g 

bc=balance_check(data,'treat',('age','ed','black','hisp','married','nodeg','re74','re75','re78'))
print(bc)

# 2.a
ate_md(data['re78'],data['treat'])


# 2.b 

model1 = sm.OLS(data['re78'], data[['treat','age','ed','black','hisp','married','nodeg','re74','re75']])
results1 = model1.fit()
print(results1.summary())

model2 = sm.OLS(data['re78'], data[['treat']])
results2 = model2.fit()
print(results2.summary())


# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# End of the PC 1 Session #

























































