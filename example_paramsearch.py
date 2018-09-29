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
from parameter_search._get_best_model import _get_best_model

df = pd.read_csv('datasets/AirPassengers.csv')
y = df.AirPassengers.values
y = y.astype('float32')

_get_best_model(y,pq_rng=5,d_rng=2)
