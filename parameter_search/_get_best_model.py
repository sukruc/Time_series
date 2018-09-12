import numpy as np
import statsmodels.tsa.api as smt

def _get_best_model(TS,pq_rng=5,d_rng=2):
    '''Finds best parameters for given series to be used in
    GARCH Model fitting. Parameters are suggested based on
    lowest AIC.
    best_aic, best_order, best_mdl = _get_best_model(TS,pq_rng,d_rng)
    ---------------
    Returns:
    best_aic: Lowest AIC obtained in grid search
    best_order: (p,d,q) values giving the lowest AIC
    best_mdl: statsmodels.tsa.api.ARIMA object with best parameters
    ---------------
    Parameters:
    TS: time series object
    pq_rng: (p,q) range to perform search within
    d_rng: d range to perform search within'''
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(pq_rng) # [0,1,2,3,4]
    d_rng = range(d_rng) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl
