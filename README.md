# Introduction

This repository contains Python code for a Causal Econometrics (M.A.) course at UNISG.

The data used can be found in the 'data' subfolders.

# Specifications on Folders

## [A1_Experiments](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A1_Experiments)

We analyze the effects of temporary employment programs for disadvantaged workers in the US (ref. National Supported Work Demonstration (NSW)). In particular, the NSW randomly assigned applicants to the training program. While the treatment group received 9 to 18 months of subsidized employment and some additional support, the control group did not receive any assistance from the program. In 1978 the earnings of the participants in the treatment and control group (re78) were collected and compared. Some pre-treatment variables for both groups were measured as well. 

## [A2_Selection_on_Observables](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A2_Selection_on_Observables)

Low birth weight is associated with negative labour market and educational outcomes during adult life. Therefore, the average effect of cigarette smoking during pregnancy on the child's birth weight is explored in the economics and epidemiology literature for example by Abrevaya (2006) [[1]](#1), da Veiga and Wilder (2008) [[2]](#2) and Walker, Tekin, and Wallace (2009) [[3]](#3) who find significantly negative effects. 

We estimate the ATE by OLS with and without covariates.

## [A3_Causal_Trees](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A3_Causal_Trees)

We predict the outcome Y with the covariate X, coding our own function to find out where an SSE optimizing Regression Tree algorithm place the splits. The function gives back the best splitting value of the covariate X, the resulting optimal SSE splitting value and the row index of the corresponding optimal X value. The resulting tree leaves contain at least certain number of observations that can be defined. 

## [A4_Differences-in-Differences](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A4_Differences-in-Differences)

## References

<a id="1">[1]</a> 
Abrevaya, J. (2006). Estimating the effect of smoking on birth outcomes using a matched panel data approach. Journal of Applied Econometrics 21 (4), 489-519.

<a id="2">[2]</a> 
da Veiga, P. V. & Wilder, R. P. (2008). Maternal smoking during pregnancy and birthweight: a propensity score matching approach. Maternal and Child Health Journal 12 (2), 194-203.

<a id="3">[3]</a> 
Walker, M., Tekin, E., & Wallace, S. (2009). Teen smoking and birth outcomes. Southern Economic Journal 75 (3), 892-907.
