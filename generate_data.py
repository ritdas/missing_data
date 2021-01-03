import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from numpy.random import default_rng


def get_missing_data_sample(n, get_x, get_y_given_x, get_obs_prob_given_x, rng):
    d = pd.DataFrame()
    d["x"] = get_x(n)
    d["obs_prob"] = get_obs_prob_given_x(d["x"])
    d["obs"] = rng.random(n) <= d["obs_prob"]  # indicator that instance has observed Y
    d["y"] = get_y_given_x(d["x"])
    d['obs_prob_pred'] = LogisticRegression(random_state=0).fit(d[["x"]], d["obs"]).predict_proba(d[['x']])[:,1]
    return d


def get_sampler(dist="SeaVan1", rng=default_rng(0), sigma=None, obs_prob=None, p=None):
    def sampler(n, dist=dist, rng=rng, sigma=sigma, obs_prob=obs_prob, p=p):
        if dist in ["SeaVan1", "SeaVan2", "SeaVanMod1", "SeaVanMod2"]:
            # Based on Seaman & Vansteelandt "Introduction to double robust
            # methods for incomplete data" Example 1
            get_x = lambda n: rng.integers(0, 3, size=n)
            get_obs_prob_given_x = lambda x: expit(4 - 4 * x)
            if dist == "SeaVan1":
                get_y_given_x = lambda x: rng.normal(loc=x, scale=sigma)
            elif dist == "SeaVan2":
                get_y_given_x = lambda x: rng.normal(loc=(x >= 1), scale=sigma)
            elif dist == "SeaVanMod1":
                get_y_given_x = lambda x: rng.normal(loc=4*x, scale=sigma)
            elif dist == "SeaVanMod2":
                get_y_given_x = lambda x: rng.normal(loc=x**2, scale=sigma)
        elif dist in ["MCAR_bernoulli"]:
            get_x = lambda n: np.ones(n) * np.NaN  # x's are ignored so...
            get_obs_prob_given_x = lambda x: obs_prob
            get_y_given_x = lambda x: rng.binomial(n=1, p=p, size=x.shape[0])
        elif dist in ["MCAR_normal_sqr"]:
            get_x = lambda n: rng.integers(0, 3, size=n)
            get_obs_prob_given_x = lambda x: obs_prob
            get_y_given_x = lambda x: rng.normal(loc=x**3, scale=sigma, size=x.shape[0])
        elif dist in ["MCAR_normal"]:
            get_x = lambda n: np.ones(n) * np.NaN  # x's are ignored so...
            get_obs_prob_given_x = lambda x: obs_prob
            get_y_given_x = lambda x: rng.normal(loc=p, scale=sigma, size=x.shape[0])
        elif dist in ["MAR_Bern"]:
            if obs_prob is None:
                obs_prob = [0.2, 0.1, 0.025, 0.1, 0.2]
            if p is None:
                p = [0.2, 0.4, 0.6, 0.3, 0.2]
            get_x = lambda n: rng.integers(0, 5, size=n)
            get_obs_prob_given_x = lambda x: [obs_prob[i] for i in x]
            get_y_given_x = lambda x: rng.binomial(n=1, p=[p[i] for i in x])

        d = get_missing_data_sample(n, get_x, get_y_given_x, get_obs_prob_given_x, rng)
        return d

    return sampler