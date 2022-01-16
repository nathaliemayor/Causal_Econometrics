"""
CE: PC3.

Spring Semester 2021.

University of St. Gallen.
"""

# Data Analytics II: PC Project 3

# import modules here
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import statistics as st

import os 

# set working directory
PATH = os.getcwd()
sys.path.append(PATH)

sys.path.append(PATH)

# load own functions
import pc3_functions_nat as pc

# define the name for the output file
OUTPUT_NAME = 'pc3_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc3.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

xdf = pd.DataFrame(data.iloc[:,1])
y = data.iloc[:,0]
x = data.iloc[:,1]


# your solutions start here
# --------------------------------------------------------------------------- #

#1.a
summary = pc.my_summary_stats(data)

# 1.b
## SSE is the same as variance * sample size and MSE is the variance of Y 

pc.SSE_MSE(data['Y'])

# 1.c

pc.reg_tree(data=data, min_leavesize=10)


# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 3 Session #
