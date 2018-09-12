
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as sf
import statsmodels.tsa.api as st
from arch import arch_model

from plot_series import tsplot

def garch_sample_generator(n, order=1, a={'0':0.2, '1':0.5},
                            b={'1':0.3}, seed=1,prefix=''):
    w = np.random.normal(size=n)
    eps = np.zeros_like(w)
    sigsq = np.zeros_like(w)

    # # FIXME: sample generator is not properly working
    for t in range(1,n):
        sigsq[t] += a['0']
        for i in range(order):
            print(i)
            sigsq[t] = sigsq[t] \
                      + a[str(i+1)]*(eps[t-i-1]**2)\
                      + b[str(i+1)]*sigsq[t-i-1]

        eps[t] = w[t] + np.sqrt(sigsq[t])

    #tsplot(eps,lags=30)
    return sigsq,eps
