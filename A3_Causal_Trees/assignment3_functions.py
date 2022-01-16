"""
Data Analytics II: PC3 Functions.

Spring Semester 2021.

University of St. Gallen.
"""

# import modules
import sys
import pandas as pd
import numpy as np


# write a function to produce summary stats:
# mean, variance, standard deviation, maximum and minimum,
# the number of missings, unique values and number of observations
def my_summary_stats(data):
    """
    Summary stats: mean, variance, standard deviation, maximum and minimum.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe for which descriptives will be computed
    Returns

    -------
    None. Prints descriptive table od the data
    """
    # generate storage for the stats as an empty dictionary
    my_descriptives = {}
    # loop over columns
    for col_id in data.columns:
        # fill in the dictionary with descriptive values by assigning the
        # column ids as keys for the dictionary
        my_descriptives[col_id] = [data[col_id].mean(),                  # mean
                                   data[col_id].var(),               # variance
                                   data[col_id].std(),                # st.dev.
                                   data[col_id].max(),                # maximum
                                   data[col_id].min(),                # minimum
                                   sum(data[col_id].isna()),          # missing
                                   len(data[col_id].unique()),  # unique values
                                   data[col_id].shape[0]]      # number of obs.
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    my_descriptives = pd.DataFrame(my_descriptives,
                                   index=['mean', 'var', 'std', 'max', 'min',
                                          'na', 'unique', 'obs']).transpose()
    # define na, unique and obs as integers such that no decimals get printed
    ints = ['na', 'unique', 'obs']
    # use the .astype() method of pandas dataframes to change the type
    my_descriptives[ints] = my_descriptives[ints].astype(int)
    # print the descriptives, (\n inserts a line break)
    print('Descriptive Statistics:', '-' * 80,
          round(my_descriptives, 2), '-' * 80, '\n\n', sep='\n')


# define an Output class for simultaneous console - file output
class Output():
    """Output class for simultaneous console/file output."""

    def __init__(self, path, name):

        self.terminal = sys.stdout
        self.output = open(path + name + ".txt", "w")

    def write(self, message):
        """Write both into terminal and file."""
        self.terminal.write(message)
        self.output.write(message)

    def flush(self):
        """Python 3 compatibility."""
        
# write a function that compute the SSE and the MSE for the prediction tree with only the root node

def SSE_MSE(outcomevector):
   """

    Parameters
    ----------
    outcomevector : TYPE: vector
        DESCRIPTION. vector containing the outcome variable

    Returns Sum of squared error (comparing the outcome's mean (prediction) with the oucomes) and the MSE
    -------
    None. Prints the SSE and the MSE

    """
   SSE = round(sum((outcomevector-outcomevector.mean())**2),2)
   MSE = round((SSE/outcomevector.shape[0]),2)
   print('SSE:', SSE, 'MSE:', MSE, '\n\n')


# function for regression tree with only one split

def reg_tree(data, min_leavesize=10):
    """


    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe containing variables of interest
    min_leavesize : TYPE: integer
        DESCRIPTION: Number of minimum leave size. The default is 10.

    Returns
    -------
    None. Prints the splitting value, the row of the x, the SSE, the prediction of the two leaves and the MSE

    """
    y = data.iloc[:, 0]
    x = data.iloc[:, 1]
    threshold = x.unique().tolist()
    threshold.sort()
    sse = []
    for i in range(min_leavesize, len(y)-min_leavesize):
        sp = threshold[i]
        sse_ = sum((y[x < sp]-y[x < sp].mean())**2)+sum((y[x >= sp]-y[x >= sp].mean())**2)
        sse.append(sse_)
    min_SSE = min(sse)
    split_at = threshold[np.argmin(sse)]
    print('Regression Tree splits at:', split_at, 'the value lies in row:',
          data.loc[data['X'] == split_at].index.to_numpy(), 'and the SSE is:',
          min_SSE, '\n')
    print('The prediction for y is:',
              y[x < split_at].mean(),
              'if x is < than', split_at,
              'and the prediction for y is:', y[x >= split_at].mean(),
              'if x is >= than', split_at)
    print('MSE:', (sum((y[x < split_at]-y[x < split_at].mean())**2))/len(y)+(sum((y[x >= split_at]-y[x >= split_at].mean())**2))/len(y))


