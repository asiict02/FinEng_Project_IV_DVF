"""
dvf_models.py
=============
Step 3: Deterministic Volatility Function (DVF) Model Specifications

This script defines the five DVF model specifications from Dumas et al. (1998).
Each model is a parametric function that predicts volatility sigma as a function
of strike K and time-to-maturity T. The predicted sigma is then plugged into
Black-Scholes to get a model option price.

The key idea: instead of using a flat (constant) volatility like standard BS,
DVF models allow sigma to vary across strikes and maturities — capturing the
empirically observed volatility smile and term structure.

Inputs:
    data/processed/options_with_iv.csv   (produced by implied_vol.py)

Outputs:
    This script does NOT produce a standalone output CSV.
    Its functions are IMPORTED by estimation.py and evaluation.py.

Usage (import in other scripts):
    from dvf_models import predict_sigma, predict_price, MODEL_SPECS
"""

import numpy as np
import pandas as pd

# Import Black-Scholes pricing from our own implied_vol.py
# This keeps all BS logic in one place — no duplication
from implied_vol import get_bs_price


# =============================================================================
# SECTION 1: MODEL DEFINITIONS
# =============================================================================
# The five DVF specifications from Dumas et al. (1998), Table 1.
# Each model predicts sigma as a polynomial in K (strike) and T (maturity).
#
# Why polynomial in K and T?
#   - The volatility smile shows IV varies with strike (K effect)
#   - The term structure shows IV varies with maturity (T effect)
#   - Polynomial models are simple, interpretable, and easy to fit
#
# Notation:
#   a0, a1, a2, a3, a4 = parameters to be estimated
#   K  = strike price
#   T  = time to maturity (years)
#   KT = interaction term between strike and maturity

# MODEL_SPECS is a dictionary mapping model name → (formula description, n_params)
# This is used by estimation.py to know how many parameters to initialise
MODEL_SPECS = {
    "M0": {
        "description": "Flat volatility (Black-Scholes benchmark): sigma = a0",
        "n_params":    1,
        "param_names": ["a0"],
    },
    "M1": {
        "description": "Linear in strike: sigma = a0 + a1*K",
        "n_params":    2,
        "param_names": ["a0", "a1"],
    },
    "M2": {
        "description": "Quadratic in strike: sigma = a0 + a1*K + a2*K^2",
        "n_params":    3,
        "param_names": ["a0", "a1", "a2"],
    },
    "M3": {
        "description": "Quadratic in strike + maturity: sigma = a0 + a1*K + a2*K^2 + a3*T",
        "n_params":    4,
        "param_names": ["a0", "a1", "a2", "a3"],
    },
    "M4": {
        "description": "Full model with K*T interaction: sigma = a0 + a1*K + a2*K^2 + a3*T + a4*K*T",
        "n_params":    5,
        "param_names": ["a0", "a1", "a2", "a3", "a4"],
    },
}


# =============================================================================
# SECTION 2: DVF SIGMA PREDICTOR
# =============================================================================

def predict_sigma(params: np.ndarray, K: float, T: float, model_id: str) -> float:
    """
    Given a set of estimated parameters, a strike K, and maturity T,
    returns the predicted volatility sigma according to the chosen DVF model.

    Parameters
    ----------
    params   : array of fitted coefficients (length depends on model)
    K        : strike price (e.g. 4800.0)
    T        : time to maturity in years (e.g. 0.25 for 3 months)
    model_id : one of 'M0', 'M1', 'M2', 'M3', 'M4'

    Returns
    -------
    float : predicted sigma (volatility), clipped to [0.001, 5.0] to avoid
            degenerate values entering Black-Scholes

    How it works:
      Each model is just a polynomial. We unpack the parameter array into
      named coefficients (a0, a1, ...) and apply the formula.
      The clip at the end ensures sigma stays in a numerically valid range —
      negative or zero sigma would cause BS to crash.
    """

    if model_id == "M0":
        # ── Model 0: Flat (constant) volatility ──────────────────────────────
        # This is the Black-Scholes assumption: one single vol for all K, T.
        # It serves as the benchmark — all other models should beat this.
        a0 = params[0]
        sigma = a0

    elif model_id == "M1":
        # ── Model 1: Linear in strike ─────────────────────────────────────────
        # Allows sigma to increase or decrease linearly with K.
        # Captures a simple linear smile/skew.
        a0, a1 = params[0], params[1]
        sigma = a0 + a1 * K

    elif model_id == "M2":
        # ── Model 2: Quadratic in strike ──────────────────────────────────────
        # Allows sigma to have a U-shape (smile) or inverted-U across K.
        # The a2*K^2 term captures the curvature of the smile.
        a0, a1, a2 = params[0], params[1], params[2]
        sigma = a0 + a1 * K + a2 * K**2

    elif model_id == "M3":
        # ── Model 3: Quadratic in strike + maturity level ─────────────────────
        # Adds a3*T to model the term structure of volatility.
        # If a3 > 0, longer-dated options have higher vol (normal term structure).
        # If a3 < 0, shorter-dated options have higher vol (inverted, e.g. during stress).
        a0, a1, a2, a3 = params[0], params[1], params[2], params[3]
        sigma = a0 + a1 * K + a2 * K**2 + a3 * T

    elif model_id == "M4":
        # ── Model 4: Full model with K*T interaction ──────────────────────────
        # The interaction term a4*K*T allows the slope of the smile to change
        # with maturity — i.e., the skew is steeper for short maturities than
        # long ones. This is the most flexible model in Dumas et al. (1998).
        a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
        sigma = a0 + a1 * K + a2 * K**2 + a3 * T + a4 * K * T

    else:
        raise ValueError(f"Unknown model_id '{model_id}'. Choose from: M0, M1, M2, M3, M4")

    # Clip sigma to a valid range.
    # sigma <= 0 would make Black-Scholes undefined (log of negative, sqrt of 0).
    # sigma > 5.0 (500%) is almost certainly a numerical artefact.
    sigma = float(np.clip(sigma, 0.001, 5.0))
    return sigma


# =============================================================================
# SECTION 3: MODEL PRICE PREDICTOR
# =============================================================================

def predict_price(params: np.ndarray, K: float, T: float,
                  S: float, r: float, q: float,
                  option_type: str, model_id: str) -> float:
    """
    Full pipeline: DVF sigma → Black-Scholes price.

    This is the function that estimation.py minimises:
      1. predict_sigma() gives us the model's predicted volatility
      2. get_bs_price()  gives us the option price at that volatility

    The output is what we compare to the observed market price.

    Parameters
    ----------
    params      : DVF model parameters (to be estimated)
    K           : strike price
    T           : time to maturity in years
    S           : spot price at observation date
    r           : continuously compounded risk-free rate
    q           : continuous dividend yield
    option_type : 'call' or 'put'
    model_id    : 'M0' to 'M4'

    Returns
    -------
    float : model-predicted option price
    """
    # Step 1: get predicted vol from the DVF polynomial
    sigma = predict_sigma(params, K, T, model_id)

    # Step 2: plug predicted vol into Black-Scholes to get the model price
    price = get_bs_price(S, K, T, r, q, sigma, option_type)

    return price


# =============================================================================
# SECTION 4: MODEL IV PREDICTOR
# =============================================================================

def predict_iv(params: np.ndarray, K: float, T: float, model_id: str) -> float:
    """
    Returns the model-implied volatility directly (without going through price).

    For DVF models, the "model IV" is simply the output of the DVF polynomial
    itself — because by construction, the DVF IS the volatility surface.

    This is used in loss functions L2, L4, L5 which compute errors in
    volatility space rather than price space.

    Parameters
    ----------
    params   : DVF model parameters
    K        : strike price
    T        : time to maturity in years
    model_id : 'M0' to 'M4'

    Returns
    -------
    float : model-predicted implied volatility
    """
    # The DVF model's predicted sigma IS the implied volatility by construction.
    # No inversion needed — unlike market prices where we had to use brentq.
    return predict_sigma(params, K, T, model_id)


# =============================================================================
# SECTION 5: VECTORISED HELPERS (for use on full DataFrames)
# =============================================================================

def apply_model_to_df(df: pd.DataFrame, params: np.ndarray,
                      model_id: str) -> pd.DataFrame:
    """
    Applies the DVF model to an entire DataFrame of options at once.
    Returns the DataFrame with two new columns:
      - ModelSigma : predicted volatility from DVF
      - ModelPrice : predicted option price from BS(ModelSigma)

    This is a convenience function so estimation.py doesn't need loops.

    Parameters
    ----------
    df       : DataFrame with columns S0, Strike, T, Rf, q, OptionType
    params   : DVF model parameters
    model_id : 'M0' to 'M4'

    Returns
    -------
    pd.DataFrame with added ModelSigma and ModelPrice columns
    """
    df = df.copy()

    # Apply predict_sigma row-by-row using vectorised lambda
    # K = Strike column, T = T column (maturity in years)
    df["ModelSigma"] = df.apply(
        lambda row: predict_sigma(params, row["Strike"], row["T"], model_id),
        axis=1
    )

    # Apply Black-Scholes price using the predicted sigma
    df["ModelPrice"] = df.apply(
        lambda row: get_bs_price(
            row["S0"], row["Strike"], row["T"],
            row["Rf"], row["q"],
            row["ModelSigma"], row["OptionType"]
        ),
        axis=1
    )

    return df


# =============================================================================
# SECTION 6: PARAMETER INITIALISATION HELPER
# =============================================================================

def get_initial_params(model_id: str, df: pd.DataFrame) -> np.ndarray:
    """
    Returns a sensible starting point for the optimiser in estimation.py.

    Why does this matter?
      scipy.optimize.minimize is a local optimiser — it finds the nearest
      minimum from the starting point. A bad starting point can lead to a
      local minimum instead of the global one.

    Strategy:
      - a0 (intercept): initialise at the mean IV in the data (~0.20)
      - a1, a2 (strike terms): start at 0 (no slope assumed)
      - a3 (maturity term): start at 0 (flat term structure assumed)
      - a4 (interaction): start at 0

    Parameters
    ----------
    model_id : 'M0' to 'M4'
    df       : the options DataFrame (used to set a0 near the mean IV)

    Returns
    -------
    np.ndarray of initial parameter values
    """
    # Use the mean IV from data as the starting intercept
    # This is much better than starting at 0 or 1 arbitrarily
    mean_iv = df["IV"].mean() if "IV" in df.columns else 0.20

    n = MODEL_SPECS[model_id]["n_params"]

    # All parameters start at 0 except a0 which starts at mean_iv
    init = np.zeros(n)
    init[0] = mean_iv   # a0 = intercept ≈ average volatility level

    return init


# =============================================================================
# SECTION 7: QUICK DIAGNOSTIC — run this file directly to check models work
# =============================================================================

if __name__ == "__main__":
    print("\n=== DVF Model Specifications — Quick Test ===\n")

    # Example option: SPX at 5000, strike 5000, 3-month expiry
    S_test = 5000.0
    K_test = 5000.0   # at-the-money
    T_test = 0.25     # 3 months
    r_test = 0.05     # 5% risk-free rate
    q_test = 0.013    # 1.3% dividend yield

    print(f"Test option: S={S_test}, K={K_test}, T={T_test}y, r={r_test}, q={q_test}\n")

    # Dummy parameter sets — just to verify the functions run correctly
    test_params = {
        "M0": np.array([0.20]),
        "M1": np.array([0.50, -0.00005]),
        "M2": np.array([0.80, -0.0002, 1e-8]),
        "M3": np.array([0.80, -0.0002, 1e-8, -0.05]),
        "M4": np.array([0.80, -0.0002, 1e-8, -0.05, 1e-6]),
    }

    print(f"{'Model':<5} {'Params':<6} {'Pred. Sigma':>12} {'Call Price':>12} {'Put Price':>12}")
    print("-" * 55)

    for model_id, params in test_params.items():
        sigma = predict_sigma(params, K_test, T_test, model_id)
        call  = predict_price(params, K_test, T_test, S_test, r_test, q_test, "call", model_id)
        put   = predict_price(params, K_test, T_test, S_test, r_test, q_test, "put",  model_id)
        print(f"{model_id:<5} {len(params):<6} {sigma:>12.4f} {call:>12.4f} {put:>12.4f}")

    print("\n✓ All models functional.")
    print("\nAvailable for import:")
    print("  from dvf_models import predict_sigma    # sigma from DVF polynomial")
    print("  from dvf_models import predict_price    # BS price at predicted sigma")
    print("  from dvf_models import predict_iv       # model IV (= predict_sigma)")
    print("  from dvf_models import apply_model_to_df  # vectorised, for DataFrames")
    print("  from dvf_models import get_initial_params # starting point for optimiser")
    print("  from dvf_models import MODEL_SPECS      # model metadata dictionary")
