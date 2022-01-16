"""
CE: PC4.

University of St. Gallen.
"""

# Data Analytics II: PC Project 4

# import modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

import os 

# set working directory
PATH = os.getcwd()
sys.path.append(PATH)

# load own functions
import pc4_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc4_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc4.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #

# 1.b
pc.my_summary_stats(data)

# 1.c

pc.histogram(data, 'fte')

# 1. d

pc.sumupto_1(data)
pc.table_sumup(data)

# 1.e

pc.desc_sum(data)

# 1.f

data = pd.get_dummies(data, columns=['chain'])
data = data.rename(columns={"chain_1": "Burgerking", "chain_2": "KFC",
                     'chain_3': 'Royrogers', 'chain_4': 'Wendys'})
data = pd.get_dummies(data, drop_first=True, columns=['year'])


# 2.a
# ATE of avg min wage on fte by mean difference (NJ ad PA) in 1993
ate_md(data[data['year_93'] == 1]['fte'],data[data['year_93'] == 1]['state'])

# ATE of avg min wage on fte by mean difference (NJ ad PA) in 1992
ate_md(data[data['year_93'] == 0]['fte'],data[data['year_93'] == 0]['state'])


# 2.b
# ATE of avg min wage on fte by mean difference (1992 and 1993) in NJ
ate_md(data[data['state'] == 1]['fte'],data[data['state'] == 1]['year_93'])


# ATE of avg min wage on fte by mean difference (1992 and 1993) in PA
ate_md(data[data['state'] == 0]['fte'],data[data['state'] == 0]['year_93'])

# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 4 Session #
















































