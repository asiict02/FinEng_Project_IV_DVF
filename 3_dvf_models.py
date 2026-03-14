"""
3_dvf_models.py
===============
Step 3: DVF Model Specifications — Dumas et al. (1998)

Defines five polynomial DVF models (M0–M4) mapping (K, T) → sigma,
then plugging sigma into Black-Scholes to obtain model prices.

Imported by: 4_loss_functions.py, 5_estimation.py
"""

import numpy as np
import pandas as pd
import importlib
implied_vol = importlib.import_module("2_implied_vol")
get_bs_price = implied_vol.get_bs_price

# ── Model metadata (used by estimation.py for param initialisation) ────────────

MODEL_SPECS = {
    "M0": {"description": "sigma = a0",                          "n_params": 1, "param_names": ["a0"]},
    "M1": {"description": "sigma = a0 + a1*K",                   "n_params": 2, "param_names": ["a0","a1"]},
    "M2": {"description": "sigma = a0 + a1*K + a2*K^2",          "n_params": 3, "param_names": ["a0","a1","a2"]},
    "M3": {"description": "sigma = a0 + a1*K + a2*K^2 + a3*T",   "n_params": 4, "param_names": ["a0","a1","a2","a3"]},
    "M4": {"description": "sigma = a0 + a1*K + a2*K^2 + a3*T + a4*K*T", "n_params": 5, "param_names": ["a0","a1","a2","a3","a4"]},
}

# ── Core: DVF sigma predictor ──────────────────────────────────────────────────

def predict_sigma(params: np.ndarray, K, T, model_id: str):
    """
    Returns predicted volatility for scalar or array inputs (K, T).
    Clipped to [0.001, 5.0] to keep Black-Scholes numerically valid.
    """
    p = params
    if model_id == "M0":
        sigma = p[0]
    elif model_id == "M1":
        sigma = p[0] + p[1]*K
    elif model_id == "M2":
        sigma = p[0] + p[1]*K + p[2]*K**2
    elif model_id == "M3":
        sigma = p[0] + p[1]*K + p[2]*K**2 + p[3]*T
    elif model_id == "M4":
        sigma = p[0] + p[1]*K + p[2]*K**2 + p[3]*T + p[4]*K*T
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. Choose from: M0, M1, M2, M3, M4")
    return np.clip(sigma, 0.001, 5.0)

# ── Price predictor: DVF sigma → BS price ─────────────────────────────────────

def predict_price(params: np.ndarray, K: float, T: float,
                  S: float, r: float, q: float,
                  option_type: str, model_id: str) -> float:
    """Returns BS option price at the DVF-predicted sigma."""
    moneyness = K / S                                          # ← rescale here
    sigma     = predict_sigma(params, moneyness, T, model_id) # ← use moneyness
    return get_bs_price(S, K, T, r, q, sigma, option_type)    # ← K unchanged in BS


# predict_iv = predict_sigma (DVF sigma IS the model IV by construction)
predict_iv = predict_sigma


# ── Vectorised DataFrame helper ────────────────────────────────────────────────

def apply_model_to_df(df: pd.DataFrame, params: np.ndarray, model_id: str) -> pd.DataFrame:
    from scipy.stats import norm
    df = df.copy()

    S, K, T = df["S0"].values, df["Strike"].values, df["T"].values
    r, q = df["Rf"].values, df["q"].values

    moneyness = K / S  # ← rescale here
    sigma = predict_sigma(params, moneyness, T, model_id)  # ← use moneyness

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    is_call = (df["OptionType"] == "call").values

    df["ModelSigma"] = sigma
    df["ModelPrice"] = np.where(is_call, call, put)
    return df


# ── Initial parameter guess for optimiser ─────────────────────────────────────

def get_initial_params(model_id: str, df: pd.DataFrame) -> np.ndarray:
    """Intercept starts at mean IV; all slope terms start at 0."""
    init    = np.zeros(MODEL_SPECS[model_id]["n_params"])
    init[0] = df["IV"].mean() if "IV" in df.columns else 0.20
    return init