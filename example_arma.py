import warnings
import itertools

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

#Fit ARMA(2,2) to dataset
arma22 = smt.ARMA(y, order=(2,2)).fit(max_lag=30, method='mle', trend='nc',
                                    burnin=0) #burnin: num of samples to be discarded before Fit

with open('results_arma.txt','w') as file:
    file.write(str(arma22.summary()))


results.plot_diagnostics(figsize=(15,12))
plt.savefig('plots/residuals_arma.png')
plt.close('all')

#---------------------------------------------------------
# Making predictions

pred = arma22.predict(start=pd.to_datetime('1957-01-01'),
                        end=pd.to_datetime('1980-01-01'))


ax = y['1956':].plot(label='observed')
pred.plot(ax=ax, label='Forecast, alpha=.7')

#ax.fill_between(pred_ci.index,
    #            pred_ci.iloc[:,0],
    #            pred_ci.iloc[:,1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Values')
plt.legend()
plt.savefig('plots/forecast_arma.png')
plt.close('all')
