import warnings
import itertools
from datetime import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as sf
import statsmodels.tsa.api as smt
from arch import arch_model

from plots.plot_series import tsplot

df = pd.read_csv('datasets/AirPassengers.csv')
df['Month'] = pd.DatetimeIndex(df['Month'])
df = df.set_index('Month')
y = df.AirPassengers

split_date = dt(1957,12,1)

#Fit ARMA(2,2) to dataset
arma22 = smt.ARMA(y[:'1957-12-01'], order=(2,2)).fit(max_lag=30, method='mle', trend='nc',
                                    burnin=0) #burnin: num of samples to be discarded before Fit


#Residuals from ARMA(2,2) will be plugged in GARCH(2,2)
arma_residuals = arma22.resid

p = 3   # defining GARCH model parameters
o = 0
q = 3

garch_m = arch_model(arma_residuals, p=p,o=o,q=q, dist='StudentsT')
#dist: distribution of errors,'normal', 't', 'skewt' or 'ged'

res = garch_m.fit(update_freq=5, disp='final',last_obs=split_date)

# uncomment 2 lines below to print model summary to a text file
#with open('results_garch.txt','w') as file:
#    file.write(str(res.summary()))

# uncomment 3 lines below to plot and save model diagnostics
#res.plot_diagnostics(figsize=(15,12))
#plt.savefig('plots/residuals_garch.png')
#plt.close('all')

#---------------------------------------------------------
# Making predictions:
# Making predictions is a three step process.
#   1- Forecasting the Mean
#   2- Forecasting error variance
#   3- Constructing the signal back fro those two components

# 1- Forecast Mean
# We will use ARMA(2,2) model we have previously fitted to forecast mean

pred_mean = arma22.predict(start=pd.to_datetime('1957-01-01'),
                             end=pd.to_datetime('1980-01-01'))

# 2- Forecast Error Variance
# GARCH(2,2) model will compute error variances for the defined time interval

pred_error = res.forecast(horizon=5)
#garch_m.forecast

ax = y['1956':].plot(label='observed')
pred.residual_variance.plot(ax=ax, label='Forecast', alpha=.7)

#ax.fill_between(pred_ci.index,
    #            pred_ci.iloc[:,0],
    #            pred_ci.iloc[:,1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Values')
plt.legend()
figfile = 'plots/forecast_garch.png'
plt.savefig(figfile)
print('figure saved to:'+figfile)
plt.close('all')
