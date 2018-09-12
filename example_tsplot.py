#IN this example, various time series samples will be created and analyzed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as sf
import statsmodels.tsa.api as st

from plot_series import tsplot
#----------------------------------------------
#Gaussian White Noise
np.random.seed(1)

# plot of discrete white noise
random_series = np.random.normal(size=1000)
tsplot(random_series, lags=30)
print('Gaussian White Noise\n--------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}'.
        format(random_series.mean(),random_series.var(),random_series.std()))
plt.savefig('random_series_tsa.png',format='png')
plt.close()
#-----------------------------------------------
#Random Walk
np.random.seed(5)
n_samples = 10000

x = w = np.random.normal(size=n_samples)
for t in range(n_samples):
    x[t] = x[t-1] + w[t]

tsplot(x, lags=30)
plt.savefig('random_walk_tsa.png',format='png')
plt.close()

tsplot(np.diff(x),lags=30)
plt.savefig('random_walk_diff_tsa.png',format='png')
plt.close()
#------------------------------------------------
#Linear Model
w = np.random.randn(100)
y = np.empty_like(w)

b0 = -50.
b1 = 25.
for t in range(len(w)):
    y[t] = b0 + b1*t + w[t]

tsplot(y,lags=30)
plt.savefig('linear_model.png',format='png')
plt.close()
#---------------------------------------------------------
#Log-Linear Model
#create some sample data
idx = pd.date_range('2010-01-01','2018-01-01',freq='M')
#assume that data is increasing at exponential rate
sales = [np.exp(x/12) for x in range(1,len(idx)+1)]

#create data frame and plot
#df = pd.DataFrame(sales,columns='Sales',index=idx)

tsplot(sales,lags=30)
plt.savefig('log_linear_model.png',format='png')
plt.close()
#with plt.style.context('bmh'):
#    df.plot()
#    plt.title('Sample Sales')
