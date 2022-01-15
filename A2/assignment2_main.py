"""
CE - assignment 2
University of St. Gallen.
"""


# import modules here
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats
import statsmodels.api as sm

# set working directory
import os 

# set working directory
PATH = os.getcwd()
sys.path.append(PATH)

# load own functions
import pc2_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc2_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc2.csv'
DATANAME_CLEAN = 'data_pc2_clean.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #

# define variable names first
Y_NAME = 'bweight'
D_NAME = 'mbsmoke'
x_name = ('mhisp', 'alcohol', 'deadkids', 'mage', 'medu',
          'nprenatal', 'mrace', 'order_1.0', 'prenatal_1.0')
WIDTH = 80
# calling the function my_summary_stats from the function file

pc.my_summary_stats(data)

# making a for loop to generate the histogramm of the non binary variables
# using the my_hist function from the file

for varname in data.columns:
    # check if var has more than 2 values (no histograms needed for dummies)
    if len(data[varname].unique()) > 2:
        # if True, plot a histogram for this variable
        pc.my_hist(data, varname, path=PATH, nbins=50)

# 1.c
# Dropping "msmoke" (multiple treatments) and "monthslb" (many missung values)
data = data.drop(['msmoke', 'monthslb'], axis=1)

# delete the observations that have missing values in "order" and "prenatal"
# We can use the function on the whole data since there are only in "order"
# and in "prenatal" missing values after dropping "monthslb"
data = data.dropna()

# Recode "order" and "prenatal" into dummies

data_w_dummies = pd.get_dummies(data, prefix='order', columns=['order'])
data_w_dummies = data_w_dummies.drop(data_w_dummies.columns[11:21], axis=1)

data_w_dummies = pd.get_dummies(data_w_dummies, prefix='prenatal',
                                columns=['prenatal'])
data_w_dummies = data_w_dummies.drop(data_w_dummies.columns[11:12], axis=1)

# 1.e

pc.my_summary_stats(data_w_dummies)

data_w_dummies.to_csv(PATH+DATANAME_CLEAN, index=False)

corr=round(data_w_dummies.corr(),2)

# 1.f
round(pc.balance_check(data_w_dummies, D_NAME, x_name), 3)

# 2.a

ols = sm.OLS(endog=data_w_dummies[Y_NAME],
             exog=sm.add_constant(data_w_dummies[D_NAME]))
ols_result = ols.fit()
print('OLS estimation using Statsmodels:', '-' * WIDTH,
      ols_result.summary(), '-' * WIDTH, '\n\n', sep='\n')

# 2.b
# all variables

ols_smx = sm.OLS(endog=data_w_dummies[Y_NAME],
                 exog=sm.add_constant(data_w_dummies.loc[:, (D_NAME, ) + x_name]))
ols_result = ols_smx.fit()
print('OLS estimation using Statsmodels:', '-' * WIDTH,
      ols_result.summary(), '-' * WIDTH, '\n\n', sep='\n')

# selected variables

ols_smx1 = sm.OLS(endog=data_w_dummies[Y_NAME],
                 exog=sm.add_constant(data_w_dummies.loc[:, (D_NAME, ) + ('alcohol','nprenatal', 'mrace', 'order_1.0', 'prenatal_1.0')]))
ols_result1 = ols_smx1.fit()
print('OLS estimation using Statsmodels:', '-' * WIDTH,
      ols_result1.summary(), '-' * WIDTH, '\n\n', sep='\n')



# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 2 Session #
