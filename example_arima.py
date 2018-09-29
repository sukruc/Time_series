import warnings
import itertools
from datetime import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as sf
import statsmodels.tsa.api as st
from arch import arch_model

from plots.plot_series import tsplot

df = pd.read_csv('datasets/AirPassengers.csv')
df['Month'] = pd.DatetimeIndex(df['Month'])
df = df.rename(columns={'Month': 'ds',
                        'AirPassengers': 'y'})
df = df.set_index('ds')


(p,d,q) = (1,1,1)
(P,D,Q,S) = (1,1,1,12)
model_arima = st.statespace.SARIMAX(df.y, order=(p,d,q), seasonal_order=(P,D,Q,S),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
results = model_arima.fit()

with open('results_arima.txt','w') as f:
    f.write(str(results.summary()))

results.plot_diagnostics(figsize=(15,12))
plt.savefig('plots/residuals_arima.png')
plt.close()

#pred = results.predict(start=pd.to_datetime('1957-01-01'), end=pd.to_datetime('1980-01-01'))

split_date = dt(1957,12,1)
pred = results.get_prediction(start=pd.to_datetime('1958-01-01'),end=pd.to_datetime('1972-01-01'),dynamic=False)

ax = y['1956':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)

#ax.fill_between(pred_ci.index,
    #            pred_ci.iloc[:,0],
    #            pred_ci.iloc[:,1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Values')
plt.legend()
figfile = 'plots/forecast_arima.png'
plt.savefig(figfile)
print('figure saved to:'+figfile)
plt.close('all')
