#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:03:37 2019

@author: Sven serneels, Ponalytics. 
"""

import matplotlib.pyplot as pp

# Small example
# Pproblem with 'GOLD' quotations
tickers = ['SA','FNV','WPM','GLD','XOM','KMB','AIG','SPY','SLV']
from pandas_datareader import data as psdat
start_date = '07-01-2018'
end_date = '03-30-2019'
data_example = psdat.DataReader(tickers,'yahoo',start_date,end_date)  ##Broken ?!!

# We'll use the daily close data for the example
# closecols = np.where(ps.MultiIndex.get_level_values(data_example.columns,'Attributes')=='Close')

data_example = data_example['Close']
# Last close price for 'GOLD' is wrong
data_example = data_example.iloc[0:(data_example.shape[0]-1)]
returns = data_example.pct_change(1)
returns = returns.iloc[1:returns.shape[0],]
log_returns = np.log(data_example).diff()
log_returns = log_returns.iloc[1:returns.shape[0],]

fig, ax = pp.subplots(figsize=(16,9))
for i in log_returns.columns:
    ax.plot(log_returns.index, log_returns.loc[:,i], label=i)
ax.set_xlabel('Date')
ax.set_ylabel('Logarithmic returns')
ax.legend()

fig, ax = pp.subplots(figsize=(16,9))
for i in returns.columns:
    ax.plot(returns.index, returns.loc[:,i], label=i)
ax.set_xlabel('Date')
ax.set_ylabel('Procentual returns')
ax.legend()

fig, ax = pp.subplots(figsize=(16,9))
for i in data_example.columns:
    ax.plot(data_example.index, data_example.loc[:,i], label=i)
ax.set_xlabel('Date')
ax.set_ylabel('Close Prices (USD)')
ax.legend()
 
# Benchmark PLS
dataw = np.matrix(returns.values.astype('float64'))
y = dataw[:,6]
X = dataw[:,0:6]
X = returns.iloc[:,0:6].to_numpy()
y = returns.iloc[:,6].to_numpy()
from sprm import robcent
centring = robcent()
Xs = centring.fit(X)
est3 = skc.PLSRegression(n_components=4)
est3.fit(Xs,(y-np.mean(y))/np.std(y))
est3.x_scores_
est3.x_weights_
est3.y_weights_
est3.y_scores_
est3.coef_
Xs*est3.coef_*np.std(y) + np.mean(y) 

# ppdire
runfile('/home/sven/Documents/PyDox/dicomo/dicomo.py', wdir='/home/sven/Documents/PyDox/dicomo')
runfile('/home/sven/Documents/PyDox/dicomo/ppdire.py', wdir='/home/sven/Documents/PyDox/dicomo')
runfile('/home/sven/Documents/PyDox/dicomo/capi.py', wdir='/home/sven/Documents/PyDox/dicomo')
est = ppdire(projection_index = capi, pi_arguments = {'max_degree' : 3,'projection_index': dicomo}, n_components=2, trimming=0,center_data=False,scale_data=False)
est.fit(X,y=y,ndir=1000,regopt='OLS')
est.x_weights_


# Big example
# Pproblem with 'GOLD' quotations
tickers = ['GS','EMN','CVX','SA','FNV','BXMT','TLT','WPM','XOM','KMB','AIG','SPY']
from pandas_datareader import data as psdat
start_date = '07-01-2018'
end_date = '03-30-2019'
data_example = psdat.DataReader(tickers,'yahoo',start_date,end_date)  ##Broken ?!!

# We'll use the daily close data for the example
# closecols = np.where(ps.MultiIndex.get_level_values(data_example.columns,'Attributes')=='Close')

data_example = data_example['Close']
# Last close price for 'GOLD' is wrong
data_example = data_example.iloc[0:(data_example.shape[0]-1)]
returns = data_example.pct_change(1)
returns = returns.iloc[1:returns.shape[0],]
log_returns = np.log(data_example).diff()
log_returns = log_returns.iloc[1:returns.shape[0],]

fig, ax = pp.subplots(figsize=(16,9))
for i in log_returns.columns:
    ax.plot(log_returns.index, log_returns.loc[:,i], label=i)
ax.set_xlabel('Date')
ax.set_ylabel('Logarithmic returns')
ax.legend()

fig, ax = pp.subplots(figsize=(16,9))
for i in returns.columns:
    ax.plot(returns.index, returns.loc[:,i], label=i)
ax.set_xlabel('Date')
ax.set_ylabel('Procentual returns')
ax.legend()

fig, ax = pp.subplots(figsize=(16,9))
for i in data_example.columns:
    ax.plot(data_example.index, data_example.loc[:,i], label=i)
ax.set_xlabel('Date')
ax.set_ylabel('Close Prices (USD)')
ax.legend()
 
# Benchmark PLS
dataw = np.matrix(returns.values.astype('float64'))
y = dataw[:,8]
X = dataw[:,np.setdiff1d(np.arange(0,12),8)]
X = returns.iloc[:,0:6].to_numpy()
y = returns.iloc[:,6].to_numpy()
from sprm import robcent
centring = robcent()
Xs = centring.fit(X)
est3 = skc.PLSRegression(n_components=4)
est3.fit(Xs,(y-np.mean(y))/np.std(y))
est3.x_scores_
est3.x_weights_
est3.y_weights_
est3.y_scores_
est3.coef_
Xs*est3.coef_*np.std(y) + np.mean(y) 

# ppdire
runfile('/home/sven/Documents/PyDox/dicomo/dicomo.py', wdir='/home/sven/Documents/PyDox/dicomo')
runfile('/home/sven/Documents/PyDox/dicomo/ppdire.py', wdir='/home/sven/Documents/PyDox/dicomo')
runfile('/home/sven/Documents/PyDox/dicomo/capi.py', wdir='/home/sven/Documents/PyDox/dicomo')
est = ppdire(projection_index = capi, pi_arguments = {'max_degree' : 3,'projection_index': dicomo}, n_components=1, trimming=0,center_data=True,scale_data=False)
est.fit(X,y=y,ndir=200)
est.x_weights_

# Portfolio Optimization
# start with Returns.csv as usual, then data from above

datar = data.iloc[:,1:8]
## install locally 
# pip install -e ./OptimalPortfolio 
# Run from level above folder that contains setup.py
runfile('/home/sven/Documents/PyDox/OptimalPortfolio/portfolioopt/opt_allocations.py', wdir='/home/sven/Documents/PyDox/OptimalPortfolio/portfolioopt')
runfile('/home/sven/Documents/PyDox/OptimalPortfolio/portfolioopt/exp_max.py', wdir='/home/sven/Documents/PyDox/OptimalPortfolio/portfolioopt')
runfile('/home/sven/Documents/PyDox/OptimalPortfolio/portfolioopt/moment_est.py', wdir='/home/sven/Documents/PyDox/OptimalPortfolio/portfolioopt')
location = datar.mean(axis=0)
scale = datar.cov()
skew = sample_skew(datar.iloc[:,1:7])  # Rreturns colwise skews
kurt = sample_kurt(datar.iloc[:,1:7])  # Rreturns colwise skews
portfolio = OptimalAllocations(6,location,scale,datar.columns[1:7])
opt_sharpe = portfolio.sharpe_opt(0)  
# Compare to other py package
from pypfopt.efficient_frontier import EfficientFrontier
efff = EfficientFrontier(location,scale)
efff.max_sharpe(0) # returns Dict 
efff.weights # Returns weights as array
# same thing
# opt_m4 = portfolio.moment ... WRONG in original package. Crashes and uses colwise M3,M4
coskew = sample_coM3(invariants)
cokurt = sample_coM4(invariants)
import timeit
t = timeit.default_timer()
opt_com = portfolio.comoment_optimisation(coskew,cokurt,.25,.25,.25,.25)
tt = timeit.default_timer() - t
efff.custom_objective(comoment_utility,location,scale,coskew,cokurt,.25,.25,.25,.25) # same

# data_example from above
(ne,pe)=data_example.shape
returns_example = stock_invariants(data_example,pe-1)
location = returns_example.mean(axis=0)
scale = returns_example.cov()
portfolio = OptimalAllocations(pe-1,location,scale,data_example.columns)
opt_sharpe = portfolio.sharpe_opt(0)  





