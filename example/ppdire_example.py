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


# ppdire
from ppdire import dicomo, capi, ppdire
est = ppdire(projection_index = capi, pi_arguments = {'max_degree' : 3,'projection_index': dicomo}, n_components=2, trimming=0,center_data=False,scale_data=False)
est.fit(X,y=y,ndir=1000,regopt='OLS')
est.x_weights_


# Big example
# Problem with 'GOLD' quotations
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

est = ppdire(projection_index = capi, pi_arguments = {'max_degree' : 3,'projection_index': dicomo}, n_components=1, trimming=0,center_data=True,scale_data=False)
est.fit(X,y=y,ndir=200)
est.x_weights_





