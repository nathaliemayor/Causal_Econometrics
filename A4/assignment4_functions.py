"""
Data Analytics II: PC4 Functions.

Spring Semester 2021.

University of St. Gallen.
"""

# import modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats
import math


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


# own procedure to do histograms
def my_hist(data, varname, path, nbins=10, label=""):
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
    label: Type: string
        DESCRIPTION. Label for the title. The default is none.

    Returns
    -------
    None. Prints and saves histogram.
    """
    # produce nice histograms
    data[varname].plot.hist(grid=True, bins=nbins, rwidth=0.9, color='grey')
    # add title
    if label == "":
        plot.title('Histogram of ' + varname)
    else:
        plot.title('Histogram of ' + varname + ' for ' + label)
    # add labels
    plot.xlabel(varname)
    plot.ylabel('Counts')
    plot.grid(axis='y', alpha=0.75)
    # save the plot
    if label == "":
        plot.savefig(path + '/histogram_of_' + varname + '.png')
    else:
        plot.savefig(path + '/histogram_of_' + varname + '_' + label + '.png')
    # print the plot
    plot.show()


# ATE estimation by mean differences
def ate_md(outcome, treatment, display=False):
    """
    Estimate ATE by differences in means.

    Parameters
    ----------
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    treatment : TYPE: pd.Series
        DESCRIPTION: vector of treatments
    display: TYPE: boolean
        DESCRIPTION: should results be printed?
        The default is False.

    Returns
    -------
    results : ATE with Standard Error
    """
    # outcomes y according to treatment status by logical vector of True/False
    # set treated and control apart using the location for subsetting
    # using the .loc both labels as well as booleans are allowed
    y_1 = outcome.loc[treatment == 1]
    y_0 = outcome.loc[treatment == 0]
    # compute ATE and its standard error and t-value
    ate = y_1.mean() - y_0.mean()
    ate_se = np.sqrt(y_1.var() / len(y_1) + y_0.var() / len(y_0))
    ate_tval = ate / ate_se
    # compute pvalues based on the normal distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    ate_pval = stats.norm.sf(abs(ate_tval)) * 2  # twosided
    # alternatively ttest_ind() could be used directly
    # stats.ttest_ind(a=y_1, b=y_0, equal_var=False)
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    result = pd.DataFrame([ate, ate_se, ate_tval, ate_pval],
                          index=['ATE', 'SE', 'tValue', 'pValue'],
                          columns=['MeanDiff']).transpose()
    # check if the results should be printed to the console
    # the following condition checks implicitly if display == True
    if display:
        # if True, return and print result (\n inserts a line break)
        print('ATE Estimate by Difference in Means:', '-' * 80,
              'Dependent Variable: ' + outcome.name, '-' * 80,
              round(result, 2), '-' * 80, '\n\n', sep='\n')
    # return the resulting dataframe too
    return result


# use own ols procedure
def my_ols(exog, outcome, intercept=True, display=True):
    """
    OLS estimation.

    Parameters
    ----------
    exog : TYPE: pd.DataFrame
        DESCRIPTION: covariates
    outcome : TYPE: pd.Series
        DESCRIPTION: outcome
    intercept : TYPE: boolean
        DESCRIPTION: should intercept be included? The default is True.
    display : TYPE: boolean
        DESCRIPTION: should results be displayed? The default is True.

    Returns
    -------
    result: ols estimates with standard errors
    """
    # check if intercept should be included
    # the following condition checks implicitly if intercept == True
    if intercept:
        # if True, prepend a vector of ones to the covariate matrix
        exog = pd.concat([pd.Series(np.ones(len(exog)), index=exog.index,
                                    name='intercept'), exog], axis=1)
    # compute (x'x)-1 by using the linear algebra from numpy
    x_inv = np.linalg.inv(np.dot(exog.T, exog))
    # estimate betas according to the OLS formula b=(x'x)-1(x'y)
    betas = np.dot(x_inv, np.dot(exog.T, outcome))
    # compute the residuals by subtracting fitted values from the outcomes
    res = outcome - np.dot(exog, betas)
    # estimate standard errors for the beta coefficients
    # se = square root of diag((u'u)(x'x)^(-1)/(N-p))
    s_e = np.sqrt(np.diagonal(np.dot(np.dot(res.T, res), x_inv) /
                              (exog.shape[0] - exog.shape[1])))
    # compute the t-values
    tval = betas / s_e
    # compute pvalues based on the students t-distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    pval = stats.t.sf(np.abs(tval),
                      (exog.shape[0] - exog.shape[1])) * 2
    # put results into dataframe and name the corresponding values
    result = pd.DataFrame([betas, s_e, tval, pval],
                          index=['coef', 'se', 't-value', 'p-value'],
                          columns=list(exog.columns.values)).transpose()
    # check if the results should be printed to the console
    # the following condition checks implicitly if display == True
    if display:
        # if True, print the results (\n inserts a line break)
        print('OLS Estimation Results:', '-' * 80,
              'Dependent Variable: ' + outcome.name, '-' * 80,
              round(result, 2), '-' * 80, '\n\n', sep='\n')
    # return the resulting dataframe too
    return result


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

# function for plotting a histogram

def histogram(data, variable):
    data_92 = data[data['year'] == 92]
    data_93 = data[data['year'] == 93]
    hist1 = data_92.hist(column=variable, bins=25, grid=False,
                       figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9,)
    plot.suptitle("year 1992")

    hist2 = data_93.hist(column=variable, bins=25, grid=False,
                       figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
    plot.suptitle('year 1993')
    print(hist1, hist2)

# function for checking if the dummies sum up to 1

def sumupto_1(data):
    """

    Parameters
    ----------
    data : TYPE: Dataframe
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if 1 == len(data[data['southj'] == 1])/len(data[data['state'] == 1]) + len(data[data['centralj'] == 1])/len(data[data['state'] == 1]) + len(data[data['northj'] == 1])/len(data[data['state'] == 1]):
        print("Dummies for NJ are well defined")
    else:
        print('Dummes for NJ are not well defined')
    if 1 == len(data[data['pa1'] == 1]) / len(data[data['state'] == 0]) + len(data[data['pa2'] == 1])/len(data[data['state'] == 0]):
        print('Dummies for PA are well defined')
    else:
        print('Dummies for PA are not well defined')
        
def table_sumup(data):
    data_variables_NJ = ['pa1', 'pa2']
    my_df = []
    for col_id in data_variables_NJ:
        share = [len(data[data[col_id] == 1])/len(data[data['state'] == 0])]
        my_df.append(share)
    my_df = pd.DataFrame(my_df,  index=['pa1', 'pa2'])
    sum1 = math.floor(sum(my_df.iloc[:, 0]))
    print(my_df, '\n', 'Sum:  ', sum1)
    if sum1 == 1:
        print("Dummies for PA are well defined")
    else:
        print('Dummies for PA are not well defined')

    data_variables_NJ = ['southj', 'centralj', 'northj']
    my_df = []
    for col_id in data_variables_NJ:
        share = [len(data[data[col_id] == 1])/len(data[data['state'] == 1])]
        my_df.append(share)
    my_df = pd.DataFrame(my_df,  index=['southj', 'centralj', 'northj'])
    sum1 = math.floor(sum(my_df.iloc[:, 0]))
    print(my_df, '\n', 'Sum:    ', sum1)
    if 1 == sum1:
        print('Dummies for NJ are well defined')
    else:
        print('Dummies for NJ are not well defined')



# 1.e function

def desc_sum(data):
    data_variables = ['fte', 'wage_st', 'hrsopen', 'price']
    my_df = {}
    dataset = [data[data['state'] == 1],
           data[data['state'] == 0],
           data[data['year'] == 92],
           data[data['year'] == 93]]
    for data in dataset:
        for col_id in data_variables:
            my_df[col_id] = [data[col_id].mean(),
                             len(data[col_id])]
            my_df = pd.DataFrame(my_df, index=['mean', 'obs'])
        print(my_df, '\n')







