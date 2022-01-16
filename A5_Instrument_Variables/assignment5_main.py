"""
CE: PC5.


University of St. Gallen.
"""

# Data Analytics II: PC Project 5

# import modules
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
# set working directory

import os 

# set working directory
PATH = os.getcwd()
sys.path.append(PATH)

# load own functions
import pc5_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc5_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc5.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #

# 1.b descriptives and histograms

pc.my_summary_stats(data)

pc.my_hist(data, "kidcount", PATH)
pc.my_hist(data, "weeks_work", PATH)

# 1.c

kids = range(2, 13)
meantable = pd.DataFrame(index=range(0, len(kids)), 
                         columns= ['kidcount', 'employed', 'Obs'])
idx = 0
for values in kids:
    meantable.loc[idx, 'kidcount'] = values
    meantable.loc[idx, 'employed'] = np.mean(data.loc[
        (data['kidcount'] == values), 'employed'])
    meantable.loc[idx, 'Obs'] = len(data.loc[
        (data['kidcount'] == values), :])

    idx = idx + 1
print(meantable)

# 1.d
CrossTable(data,'morekids','multi2nd')

# 2.a
my_ols(data[['morekids', 'age_mother','black','hisp','hsgrad','colgrad']],data['weeks_work'])


# 2.b

my_ols(data[['multi2nd','age_mother','black','hisp','hsgrad','colgrad']],data['morekids'])

Y = pd.DataFrame(data['weeks_work'])
Z = pd.DataFrame(data['multi2nd'])
D = pd.DataFrame(data['morekids'])
X = data[['age_mother', 'black', 'hisp', 'hsgrad', 'colgrad']]


sls(data, Y, Z, D, X)


# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 5 Session #
