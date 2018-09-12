# Garch Sample Generator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as sf
import statsmodels.tsa.api as st
from arch import arch_model

from plot_series import tsplot
from garch_generator import garch_generator

np.random.seed(2)

a0 = 0.2
a1 = 0.5
a2 = 0.01
b1 = 0.3
b2 = 0.15
n = 10000

sigsq,eps = garch_generator(n=n,a0=a0,a1=a1,a2=a2,b1=b1,b2=b2)
#sigsq,eps= garch_sample_generator(n=n,a={'0':a0,'1':a1},b={'1':b1},seed=2)

tsplot(eps, lags=30)
plt.savefig('garch_model_tsa.png',format='png')

tsplot(eps**2,lags=30)
plt.savefig('garch_model_epssq.png',format='png')

#Fit a GARCH(2,2) model to simulated EPS series
#sigsq,eps= garch_sample_generator(n=n,a={'0':a0,'1':a1},b={'1':b1},seed=2)

am = arch_model(eps, p=2, o=0, q=2, dist='StudentsT')
res = am.fit(update_freq=5)
print(res.summary())
