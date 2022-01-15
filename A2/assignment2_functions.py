"""
Data Analytics II: PC2 Functions.

Spring Semester 2021.

University of St. Gallen.
"""

# import modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats


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


# write a function to produce summary stats:
# mean, variance, standard deviation, maximum and minimum,
# the number of missings and unique values
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
                                   len(data[col_id].unique())]  # unique values
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    my_descriptives = pd.DataFrame(my_descriptives,
                                   index=["mean", "var", "std", "max",
                                          "min", "na", "unique"]).transpose()
    # print the descriptives, (\n inserts a line break)
    print('Descriptive Statistics:', '-' * 80,
          round(my_descriptives, 2), '-' * 80, '\n\n', sep='\n')


# own procedure to do histograms
def my_hist(data, varname, path, nbins=10):
    """
    Plot histograms.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe containing variables of interest
    varname : TYPE: string
        DESCRIPTION: variable name for which histogram should be plotted
    path : TYPE: string
        DESCRIPTION: path where the plot will be saved
    nbins : TYPE: integer
        DESCRIPTION. Number of bins. The default is 10.

    Returns
    -------
    None. Prints and saves histogram.
    """
    # produce nice histograms
    data[varname].plot.hist(grid=True, bins=nbins, rwidth=0.9, color='purple')
    # add labels
    plot.title('Histogram of ' + varname)
    plot.xlabel(varname)
    plot.ylabel('Counts')
    plot.grid(axis='y', alpha=0.75)
    # save the plot
    plot.savefig(path + '/histogram_of_' + varname + '.png')
    # print the plot
    plot.show()


# own procedure for a balance check
def balance_check(data, treatment, variables):
    """
    Check covariate balance.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data on which balancing checks should be conducted
    treatment : TYPE: string
        DESCRIPTION: name of the binary treatment variable
    variables : TYPE: tuple
        DESCRIPTION: names of the variables for balancing checks

    Returns
    -------
    Returns and Prints the Table of Descriptive Balancing Checks
    """
    # create storage for output as an empty dictionary for easy value fill
    balance = {}
    # loop over variables
    for varname in variables:
        # define according to treatment status by logical vector of True/False
        # set treated and control apart using the location for subsetting
        # using the .loc both labels as well as booleans are allowed
        treated = data.loc[data[treatment] == 1, varname]
        control = data.loc[data[treatment] == 0, varname]
        # compute difference in means between treated and control
        mdiff = treated.mean() - control.mean()
        # compute the corresponding standard deviation of the difference
        mdiff_std = (np.sqrt(treated.var() / len(treated)
                     + control.var() / len(control)))
        # compute the t-value for the difference
        mdiff_tval = mdiff / mdiff_std
        # get the degrees of freedom (unequal variances, Welch t-test)
        d_f = (mdiff_std**4 /
               (((treated.var()**2) / ((len(treated)**2)*(len(treated) - 1))) +
                ((control.var()**2) / ((len(control)**2)*(len(control) - 1)))))
        # compute pvalues based on the students t-distribution (requires scipy)
        # sf stands for the survival function (also defined as 1 - cdf)
        mdiff_pval = stats.t.sf(np.abs(mdiff_tval), d_f) * 2
        # compute the standardized difference
        sdiff = (mdiff / np.sqrt((treated.var() + control.var()) / 2)) * 100
        # combine values
        balance[varname] = [treated.mean(), control.mean(),
                            mdiff, mdiff_std, mdiff_tval, mdiff_pval, sdiff]
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    balance = pd.DataFrame(balance,
                           index=["Treated", "Control", "MeanDiff", "Std",
                                  "tVal", "pVal", "StdDiff"]).transpose()
    # print the descriptives (\n inserts a line break)
    print('Balancing Checks:', '-' * 80,
          round(balance, 2), '-' * 80, '\n\n', sep='\n')
    # return results
    return balance
