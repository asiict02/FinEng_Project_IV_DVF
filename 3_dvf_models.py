"""
3_dvf_models.py — DVF Model Specifications (Dumas et al., 1998)
Defines 5 polynomial DVF models (M0–M4) mapping (log-moneyness x, maturity T) → sigma.
Imported by: 5_estimation.py, 6_evaluation.py, 7_visualization.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# x = log(K/S): log-moneyness, centred at 0 for ATM options
MODEL_SPECS = {
    "M0": {"description": "sigma = a0", "n_params": 1, "param_names": ["a0"]},
    "M1": {"description": "sigma = a0 + a1*x  [x = log(K/S)]", "n_params": 2, "param_names": ["a0", "a1"]},
    "M2": {"description": "sigma = a0 + a1*x + a2*x^2  [x = log(K/S)]", "n_params": 3, "param_names": ["a0", "a1", "a2"]},
    "M3": {"description": "sigma = a0 + a1*x + a2*x^2 + a3*T  [x = log(K/S)]", "n_params": 4, "param_names": ["a0", "a1", "a2", "a3"]},
    "M4": {"description": "sigma = a0 + a1*x + a2*x^2 + a3*T + a4*x*T  [x = log(K/S)]", "n_params": 5, "param_names": ["a0", "a1", "a2", "a3", "a4"]},
}

def predict_sigma(params, moneyness, T, model_id):
    """
    DVF volatility function: maps (moneyness K/S, maturity T) → sigma.

    Parameters
    params     : array-like of fitted coefficients (length = n_params for model_id)
    moneyness  : scalar or array of K/S values
    T          : scalar or array of maturities in years
    model_id   : one of 'M0', 'M1', 'M2', 'M3', 'M4'
    """
    p = params
    x = np.log(moneyness)   # log-moneyness: the polynomial basis variable

    if   model_id == "M0": sigma = p[0]
    elif model_id == "M1": sigma = p[0] + p[1] * x
    elif model_id == "M2": sigma = p[0] + p[1] * x + p[2] * x**2
    elif model_id == "M3": sigma = p[0] + p[1] * x + p[2] * x**2 + p[3] * T
    elif model_id == "M4": sigma = p[0] + p[1] * x + p[2] * x**2 + p[3] * T + p[4] * x * T
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. Must be one of {list(MODEL_SPECS)}")

    return np.clip(sigma * np.ones_like(np.asarray(moneyness, dtype=float)), 0.001, 5.0)


def apply_model_to_df(df: pd.DataFrame, params, model_id: str) -> pd.DataFrame:
    """
    Applies DVF model to a DataFrame, adding ModelSigma and ModelPrice columns.

    Passes K/S (moneyness) to predict_sigma, which converts to log-moneyness
    internally. Black-Scholes is then evaluated at ModelSigma to get ModelPrice.
    """
    df = df.copy()
    S   = df["S0"].values
    K   = df["Strike"].values
    T   = df["T"].values
    r   = df["Rf"].values
    q   = df["q"].values

    moneyness = K / S
    sigma = predict_sigma(params, moneyness, T, model_id)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * np.exp(-q * T) * norm.cdf(d1)  - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    df["ModelSigma"] = sigma
    df["ModelPrice"] = np.where(df["OptionType"] == "call", call, put)
    return df
