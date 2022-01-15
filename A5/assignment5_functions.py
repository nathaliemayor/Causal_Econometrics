"""
Data Analytics II: PC5 Functions.

Spring Semester 2021.

University of St. Gallen.
"""

# import modules
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plot
from scipy import stats


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
    data[varname].plot.hist(grid=True, bins=nbins, rwidth=0.9, color='grey')
    # add labels
    plot.title('Histogram of ' + varname)
    plot.xlabel(varname)
    plot.ylabel('Counts')
    plot.grid(axis='y', alpha=0.75)
    # save the plot
    plot.savefig(path + 'histogram_of_' + varname + '.png')
    # print the plot
    plot.show()


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

# own function for corss table        
def CrossTable(data,column1,column2):
    '''
    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. dataset
    column1 : TYPE: pd.DataFrame
        DESCRIPTION. column name
    column2 : TYPE: pd.DataFrame
        DESCRIPTION. column name 

    Returns
    -------
    Cross table of the two columns

    '''
    a={},
    b={},
    c={},
    d={},
    cf={},
    d=len(data[(data[column2]==1)&(data[column1]==1)])
    b=len(data[(data[column2]==0)&(data[column1]==1)])
    c=len(data[(data[column2]==1)&(data[column1]==0)])
    a=len(data[(data[column2]==0)&(data[column1]==0)])
    cf={column1:[0,1] ,'0':[a,b],'1': [c,d]}
    print(pd.DataFrame(cf,index=['a','b'],columns=[column1,'0','1']).set_index(column1).rename_axis(column2, axis=1))
    
    


# creating function for 2SLS

def sls(data, y, z, d, x):
    """

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. Dataset
    y : TYPE: pd.DataFrame
        DESCRIPTION. Independent Variable
    z : TYPE: pd.DataFrame
        DESCRIPTION. Instrumental Variable
    d : TYPE: pd.DataFrame
        DESCRIPTION. Treatment Variable
    x : TYPE: pd.DataFrame
        DESCRIPTION. Covariates

    Returns
    -------
    None.

    """
    
    model1= sm.OLS(d, sm.add_constant(pd.concat([z, x], axis=1))).fit()
    dpred = model1.predict()
    dpred = pd.DataFrame(dpred)
    data1 = pd.concat([data, dpred], axis=1)
    data1 = data1.rename(columns={0:'dpred'})
    model2 = sm.OLS(y, sm.add_constant(pd.concat([dpred, x], axis=1))).fit()
    print(model1.summary()),
    print(model2.summary())
    



