# Introduction

This repository contains Python code for a Causal Econometrics (M.A.) course at UNISG.

# Specifications on Folders

## [Final_Project](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/Final_Project)

We compare two estimators (IPW, OLS) via a Monte Carlo simulation study within the CIA research design. We code our own estimators and generate three data generating processes such that, in the first DGP, the first estimator outperforms the second estimator, in the second DGP, the second estimator outperforms the first estimator and finally that the third DGP violates one identyfing assumption (conditional independence assumption). We then introduce some performance measures (MSE, bias and variance) in order to compare both estimators.

## [A1_Experiments](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A1_Experiments)

We analyze the effects of temporary employment programs for disadvantaged workers in the US (ref. National Supported Work Demonstration (NSW)). In particular, the NSW randomly assigned applicants to the training program. While the treatment group received 9 to 18 months of subsidized employment and some additional support, the control group did not receive any assistance from the program. In 1978 the earnings of the participants in the treatment and control group (re78) were collected and compared. Some pre-treatment variables for both groups were measured as well. 

## [A2_Selection_on_Observables](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A2_Selection_on_Observables)

Low birth weight is associated with negative labour market and educational outcomes during adult life. Therefore, the average effect of cigarette smoking during pregnancy on the child's birth weight is explored in the economics and epidemiology literature for example by Abrevaya (2006) [[1]](#1), da Veiga and Wilder (2008) [[2]](#2) and Walker, Tekin, and Wallace (2009) [[3]](#3) who find significantly negative effects. 

We estimate the ATE by OLS with and without covariates.

## [A3_Causal_Trees](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A3_Causal_Trees)

We predict the outcome Y with the covariate X, coding our own function to find out where an SSE optimizing Regression Tree algorithm place the splits. The function gives back the best splitting value of the covariate X, the resulting optimal SSE splitting value and the row index of the corresponding optimal X value. The resulting tree leaves contain at least certain number of observations that can be defined. 

## [A4_Differences-in-Differences](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A4_Differences-in-Differences)
We use a random sample of data coming from the case study of David Card and Alan B. Krueger (1994) [[4]](#4) that analyses the effect of higher minimum wages on employment in the US fast-food sector based on panel data. It uses a rise in New Jersey’s minimum wage in 1992 to evaluate employment changes in restaurants induced by the policy change compared to employment in Pennsylvania, where the minimum wage remained constant. 

First, we estimate the effect of higher minimum wages on full time equivalent employment in fast food restaurants by mean difference between New Jersey and Pennsylvania after the policy change. Second, we estimate the effect of higher minimum wages on full time equivalent employment in fast food restaurants by mean difference in New Jersey before and after the policy change. 

## [A5_Instrumental_Variables](https://github.com/nathaliemayor/Causal_Econometrics/tree/main/A5_Instrumental_Variables)

Angrist and Evans (1998) [[5]](#5) investigate the causal effect of fertility on labor supply of women. They exploit parental preferences for a heterogeneous sibling-sex composition to construct instrumental variables estimates of the effect of childbearing on labor-market outcomes. We use sample of US women in 1980 and 1990 based on data of their study. 

First, we estimate the coefficients by OLS and find the estimated average effect of having more than two kids on weeks worked per year. Second, we use the instrumental variable (Dummy: 1 if second birth was a multiple birth). We compute the 2SLS estimator by plugging the fitted OLS values of the first stage into the second stage.

## References

<a id="1">[1]</a> 
Abrevaya, J. (2006). Estimating the effect of smoking on birth outcomes using a matched panel data approach. Journal of Applied Econometrics 21 (4), 489-519.

<a id="2">[2]</a> 
da Veiga, P. V. & Wilder, R. P. (2008). Maternal smoking during pregnancy and birthweight: a propensity score matching approach. Maternal and Child Health Journal 12 (2), 194-203.

<a id="3">[3]</a> 
Walker, M., Tekin, E., & Wallace, S. (2009). Teen smoking and birth outcomes. Southern Economic Journal 75 (3), 892-907.

<a id="4">[4]</a> 
Card, David, and Alan B. Krueger (1994). Minimum Wages and Employment: A Case Study of the New Jersey and Pennsylvania Fast Food Industries. American Economic Review, 84:4, 772–793.

<a id="5">[5]</a> 
Angrist, J., & Evans, W. (1998). Children and Their Parents' Labor Supply: Evidence from Exogenous Variation in Family Size. The American Economic Review, 88(3), 450- 477.

