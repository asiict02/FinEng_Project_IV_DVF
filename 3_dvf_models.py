"""
3_dvf_models.py — DVF Model Specifications (Dumas et al., 1998)
Defines 5 polynomial DVF models (M0-M4) mapping (moneyness, T) → sigma.
Imported by: 5_estimation.py, 6_evaluation.py, 7_visualization.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

MODEL_SPECS = {
    "M0": {"description": "sigma = a0",                               "n_params": 1, "param_names": ["a0"]},
    "M1": {"description": "sigma = a0 + a1*K",                        "n_params": 2, "param_names": ["a0","a1"]},
    "M2": {"description": "sigma = a0 + a1*K + a2*K^2",               "n_params": 3, "param_names": ["a0","a1","a2"]},
    "M3": {"description": "sigma = a0 + a1*K + a2*K^2 + a3*T",        "n_params": 4, "param_names": ["a0","a1","a2","a3"]},
    "M4": {"description": "sigma = a0 + a1*K + a2*K^2 + a3*T + a4*K*T","n_params": 5, "param_names": ["a0","a1","a2","a3","a4"]},
}

def predict_sigma(params, K, T, model_id):
    """DVF volatility function: maps (moneyness K/S, maturity T) → sigma.
    Clipped to [0.001, 5.0] to keep Black-Scholes numerically stable."""
    p = params
    if   model_id == "M0": sigma = p[0]
    elif model_id == "M1": sigma = p[0] + p[1]*K
    elif model_id == "M2": sigma = p[0] + p[1]*K + p[2]*K**2
    elif model_id == "M3": sigma = p[0] + p[1]*K + p[2]*K**2 + p[3]*T
    elif model_id == "M4": sigma = p[0] + p[1]*K + p[2]*K**2 + p[3]*T + p[4]*K*T
    else: raise ValueError(f"Unknown model_id '{model_id}'")
    return np.clip(sigma, 0.001, 5.0)

def apply_model_to_df(df, params, model_id):
    """Applies DVF model to a DataFrame, adding ModelSigma and ModelPrice columns."""
    df = df.copy()
    S, K, T, r, q = df["S0"].values, df["Strike"].values, df["T"].values, df["Rf"].values, df["q"].values
    sigma = predict_sigma(params, K/S, T, model_id)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S*np.exp(-q*T)*norm.cdf(d1)  - K*np.exp(-r*T)*norm.cdf(d2)
    put  = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    df["ModelSigma"] = sigma
    df["ModelPrice"] = np.where(df["OptionType"]=="call", call, put)
    return df
