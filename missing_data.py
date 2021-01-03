import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.special import expit
from numpy.random import default_rng
import seaborn as sns
import matplotlib.pyplot as plt


def add_cumulative_estimators(d):
    d["n"] = np.array(list(range(d.shape[0]))) + 1
    d["cum_obs_cnt"] = np.cumsum(d["obs"])
    d["cum_obs_val"] = np.cumsum(d["obs"] * d["y"].astype(float))
    d["obs_mean"] = d["cum_obs_val"] / d["cum_obs_cnt"]
    d["weight"] = 1.0 / d["obs_prob_pred"]
    d["cum_obs_weight"] = np.cumsum(d["obs"] * d["weight"])
    d["cum_obs_weighted_y"] = np.cumsum(d["obs"] * d["weight"] * d["y"])
    d["ipw_est"] = d["cum_obs_weighted_y"] / d["n"]
    d["ipw_est_b"] = d["cum_obs_weighted_y"] / d["cum_obs_weight"]
    d["ipw_est_b_b"] = d["cum_obs_weighted_y"] * d["n"] / d["cum_obs_weight"] ** 2
    d["ipw_est_b3"] = d["cum_obs_weighted_y"] * d["n"] ** 2 / d["cum_obs_weight"] ** 3
    return d


def mean_estimator_plotter(
    ax,
    d,
    id_col=None,
    value_cols=["y"],
    y_label="estimate",
    true_mean=None,
    include_scatter=True,
    cp=sns.color_palette(),
    param_dict={"markers": True},
):
    """
    Parameters
    ----------
    ax : Axes
        The axes to draw to
    data : array
       Pandas DataFrame with all the columns we want
    id_col : array
       Column name for x's; of None, sets to row numbers
    value_cols : List[str]
        Response columns
    param_dict : dict
       Dictionary of kwargs to pass to ax.plot
    Returns
    -------
    out : list
        list of artists added
    """
    if ax is None:
        fig, ax = plt.subplots()
    dd = pd.melt(d, id_vars=id_col, value_vars=value_cols)
    sns.lineplot(
        data=dd, x="n", y="value", hue="variable", style="variable", ax=ax, **param_dict
    )
    if true_mean is not None:
        ax.axhline(true_mean, label="true mean", color=cp[3])
    observed_points = d[d["obs"]]["n"]
    observed_vals = d[d["obs"]]["y"]
    if include_scatter:
        ax.scatter(
             observed_points,
            observed_vals,
            color=cp[2],
            zorder=10,
            label="values observed",
        )
    ax.legend()
    return ax

def add_mean_estimators(d):
    ## Assumes columns created by genderate_data.get_missing_data_sample: "obs", "y", "x", "obs_prob"
    ## ONLY WORKS when x is a single column.  get_mean_estimators is more general...

    est = {}
    n = d.shape[0]
    obs_cnt = np.sum(d["obs"])
    sum_obs_val = np.sum(d["obs"] * d["y"])
    est["num_obs"] = obs_cnt
    est["obs_mean"] = sum_obs_val / obs_cnt

    d["weight"] = 1 / d["obs_prob_pred"]
    d["weight_obs"] = d["obs"] * d["weight"]
    d["weight_obs_val"] = d["weight_obs"] * d["y"]
    
    sum_obs_weight = np.sum(d["weight_obs"] )
    sum_obs_weighted_y = np.sum(d["weight_obs_val"])
    est["sum_obs_weight"] = sum_obs_weight
    est["ipw_mean"] = sum_obs_weighted_y / n
    est["ipw_b_mean"] = sum_obs_weighted_y / sum_obs_weight
    est["ipw_b_b_mean"] = sum_obs_weighted_y * n / sum_obs_weight ** 2

    
    if d["x"].isnull().values.any():
        return est

    # Do regression imputation
    reg = LinearRegression(fit_intercept=True, normalize=False).fit(
        X=d["x"], y=d["y"], sample_weight=d["obs"]
    )
    full_impute = reg.predict(d["x"])
    missing_impute = full_impute.copy()
    missing_impute[d["obs"]] = d[d["obs"]]["y"]  # don't impute the observed values
    est["impute_missing"] = np.mean(missing_impute)

    # Do IPW regression imputation
    reg = LinearRegression(fit_intercept=True, normalize=False).fit(
        X=d["x"], y=d["y"], sample_weight=d["weight_obs"]
    )
    full_impute_w = reg.predict(d["x"])
    missing_impute_w = full_impute_w.copy()
    missing_impute_w[d["obs"]] = d[d["obs"]]["y"]  # don't impute the observed values
    est["impute_missing_w"] = np.mean(missing_impute_w)

    # Do doubly robust estimate
    beta = 1
    d["dr_vec"] = d["weight_obs_val"] - beta * (d["weight_obs"] - 1) * full_impute
    d["dr_vec_w"] = d["weight_obs_val"] - beta *(d["weight_obs"] - 1) * full_impute_w
    est["ipw_dr_1"] = np.sum(d["dr_vec"])/n
    est["ipw_dr_w_1"] = np.sum(d["dr_vec_w"])/n

    f_centered = d["weight_obs_val"] - np.mean(d["weight_obs_val"])
    h_centered  = d["weight_obs"] * full_impute - np.mean(d["weight_obs"] * full_impute)

    beta = np.sum(f_centered * h_centered) / np.sum(h_centered)

    dr_vec = WRY - beta * (weights * obs - 1) * full_impute
    dr_vec_w = WRY - beta *(weights * obs - 1) * full_impute_w

    d["dr_vec_opt"] = d["weight_obs_val"] - beta * (d["weight_obs"] - 1) * full_impute
    d["dr_vec_w_opt"] = d["weight_obs_val"] - beta *(d["weight_obs"] - 1) * full_impute_w
    est["ipw_dr_opt"] = np.sum(d["dr_vec_opt"])/n
    est["ipw_dr_w_opt"] = np.sum(d["dr_vec_w_opt"])/n

    est["ipw_dr_b"] = np.sum(dr_vec) * est["weight"] / n**2
    est["ipw_dr_w_b"] = np.sum(dr_vec_w) * est["weight"] / n**2


def get_mean_estimators(vals, covar, propensities, obs):
    est = {}
    n = len(vals)
    ## All the estimators below assume we know the propensities (i.e. "missing by design")
    est["obs_mean"] = np.mean(vals[obs])
    est["ipw_mean"] = np.sum(vals[obs] / propensities[obs]) / n
    est["ipw_b_mean"] = np.average(vals[obs], weights=1 / propensities[obs])
    est["weight"] = np.sum(1 / propensities[obs])
    est["ipw_b_b_mean"] = (
        np.average(vals[obs], weights=1 / propensities[obs]) * n / est["weight"]
    )
    est["ipw_b3_mean"] = (
        np.average(vals[obs], weights=1 / propensities[obs])
        * n ** 2
        / est["weight"] ** 2
    )
    est["num_obs"] = np.sum(obs)

    if covar is None:
        return est

    # Do Regression  imputation
    reg = LinearRegression(fit_intercept=True, normalize=False).fit(
        X=covar[obs], y=vals[obs]
    )
    full_impute = reg.predict(covar)
    # est['impute_all'] = np.mean(full_impute)
    missing_impute = full_impute.copy()
    missing_impute[obs] = vals[obs]  # don't impute the observed values
    est["impute_missing"] = np.mean(missing_impute)

    # Do IPW regression imputation
    reg = LinearRegression(fit_intercept=True, normalize=False).fit(
        X=covar[obs], y=vals[obs], sample_weight=1 / propensities[obs]
    )
    full_impute_w = reg.predict(covar)
    # est['impute_all_w'] = np.mean(full_impute)
    missing_impute = full_impute_w.copy()
    missing_impute[obs] = vals[obs]  # don't impute the observed values
    est["impute_missing_w"] = np.mean(missing_impute)

    # Do doubly robust estimate
    weights = 1 / propensities
    beta = 1
    WRY = weights * obs * vals
    dr_vec = WRY - beta * (weights * obs - 1) * full_impute
    dr_vec_w = WRY - beta *(weights * obs - 1) * full_impute_w
    est["ipw_dr_1"] = np.sum(dr_vec)/n
    est["ipw_dr_w_1"] = np.sum(dr_vec_w)/n

    beta = 1
    dr_vec = WRY - beta * (weights * obs - 1) * full_impute
    dr_vec_w = WRY - beta *(weights * obs - 1) * full_impute_w

    est["ipw_dr_opt"] = np.sum(dr_vec)/n
    est["ipw_dr_w_opt"] = np.sum(dr_vec_w)/n

    est["ipw_dr_b"] = np.sum(dr_vec) * est["weight"] / n**2
    est["ipw_dr_w_b"] = np.sum(dr_vec_w) * est["weight"] / n**2

    return est


def eval_estimators(estimates, true_val):
    perf = {}
    perf["mean"] = np.mean(estimates)
    perf["SD"] = np.std(estimates)
    perf["SE"] = perf["SD"] / np.sqrt(estimates.shape[0])
    perf["bias"] = np.mean(estimates) - true_val
    perf["mean_abs_err"] = np.mean(np.abs(estimates - true_val))
    perf["RMS error"] = np.sqrt(np.mean((estimates - true_val) ** 2))
    return perf