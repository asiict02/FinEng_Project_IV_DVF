"""
loss_functions.py
=================
Step 4: Loss Function Implementations

This script implements the five loss functions used in Christoffersen & Jacobs (2004).
The central insight of that paper is: the choice of loss function used during
*estimation* affects which model appears "best" when evaluated — even if you
evaluate all models with the *same* loss function afterwards.

In other words, there is no single correct loss function. The comparison matrix
(rows = estimation loss, columns = evaluation loss) is the main empirical output.

This script does NOT produce a standalone CSV.
Its functions are IMPORTED by estimation.py and evaluation.py.

Usage (import in other scripts):
    from loss_functions import compute_loss, LOSS_FUNCTIONS
"""

import numpy as np
import pandas as pd

# We import get_bs_vega from implied_vol.py to compute vega weights for L5.
# This avoids rewriting the vega formula — keep all BS logic in one file.
from implied_vol import get_bs_vega


# =============================================================================
# SECTION 1: LOSS FUNCTION CATALOGUE
# =============================================================================
# This dictionary maps loss function names to human-readable descriptions.
# estimation.py and evaluation.py iterate over this dictionary.

LOSS_FUNCTIONS = {
    "L1": "MSE on Prices     — mean( (ModelPrice - MarketPrice)^2 )",
    "L2": "MSE on IV         — mean( (ModelIV - MarketIV)^2 )",
    "L3": "MAE on Prices     — mean( |ModelPrice - MarketPrice| )",
    "L4": "MAE on IV         — mean( |ModelIV - MarketIV| )",
    "L5": "Vega-weighted IVMSE — mean( Vega^2 * (ModelIV - MarketIV)^2 )",
}


# =============================================================================
# SECTION 2: INDIVIDUAL LOSS FUNCTIONS
# =============================================================================

def loss_mse_price(model_prices: np.ndarray, market_prices: np.ndarray) -> float:
    """
    L1: Mean Squared Error on option PRICES.

    Formula: (1/N) * sum[ (C_model_i - C_market_i)^2 ]

    Why use price-based loss?
      - Directly measures pricing accuracy in dollar terms.
      - Tends to give more weight to expensive (longer-dated, deeper) options
        because their price errors are larger in absolute terms.
      - This creates a bias: the model may fit expensive options well but
        perform poorly on cheap near-the-money short-dated options.

    Parameters
    ----------
    model_prices  : array of model-predicted option prices
    market_prices : array of observed market mid-prices

    Returns
    -------
    float : mean squared error in price units squared (e.g. dollars^2)
    """
    # Compute squared difference for each option, then take the mean
    errors = model_prices - market_prices
    return float(np.mean(errors ** 2))


def loss_mse_iv(model_ivs: np.ndarray, market_ivs: np.ndarray) -> float:
    """
    L2: Mean Squared Error on IMPLIED VOLATILITIES.

    Formula: (1/N) * sum[ (IV_model_i - IV_market_i)^2 ]

    Why use IV-based loss?
      - IV is dimensionless and comparable across strikes/maturities.
      - Treats all options equally regardless of their price level.
      - Christoffersen & Jacobs argue this is generally preferable because
        it avoids the systematic overweighting of expensive options.
      - However, plain IV-MSE still weights all options equally, even deep
        OTM options that may be illiquid and noisy.

    Parameters
    ----------
    model_ivs  : array of model-predicted implied volatilities
    market_ivs : array of observed implied volatilities (from implied_vol.py)

    Returns
    -------
    float : mean squared error in volatility units squared
    """
    errors = model_ivs - market_ivs
    return float(np.mean(errors ** 2))


def loss_mae_price(model_prices: np.ndarray, market_prices: np.ndarray) -> float:
    """
    L3: Mean Absolute Error on option PRICES.

    Formula: (1/N) * sum[ |C_model_i - C_market_i| ]

    Why MAE instead of MSE?
      - MAE is less sensitive to outliers than MSE (no squaring).
      - A single option with a very large pricing error (e.g. deep ITM)
        will dominate the MSE but has less influence on MAE.
      - Useful for robustness checks.

    Parameters
    ----------
    model_prices  : array of model-predicted option prices
    market_prices : array of observed market mid-prices

    Returns
    -------
    float : mean absolute error in price units (e.g. dollars)
    """
    errors = np.abs(model_prices - market_prices)
    return float(np.mean(errors))


def loss_mae_iv(model_ivs: np.ndarray, market_ivs: np.ndarray) -> float:
    """
    L4: Mean Absolute Error on IMPLIED VOLATILITIES.

    Formula: (1/N) * sum[ |IV_model_i - IV_market_i| ]

    Combines the robustness of MAE with the dimensionless scale of IV.
    Less sensitive to outliers than L2 (IV-MSE).

    Parameters
    ----------
    model_ivs  : array of model-predicted implied volatilities
    market_ivs : array of observed implied volatilities

    Returns
    -------
    float : mean absolute error in volatility units
    """
    errors = np.abs(model_ivs - market_ivs)
    return float(np.mean(errors))


def loss_vega_weighted_ivmse(model_ivs: np.ndarray, market_ivs: np.ndarray,
                              vegas: np.ndarray) -> float:
    """
    L5: Vega-Weighted Implied Volatility MSE.

    Formula: (1/N) * sum[ Vega_i^2 * (IV_model_i - IV_market_i)^2 ]

    This is the KEY loss function recommended by Christoffersen & Jacobs (2004).

    Why vega-weighting?
      Vega = dPrice/dSigma = how much the option price changes per unit of vol.
      Near-the-money (ATM) options have HIGH vega → they are most sensitive to
      volatility mis-specification → they get more weight.
      Deep OTM options have LOW vega → small errors in their IV barely affect
      the price → they get less weight.

    The economic intuition: we should care most about getting vol right where
    it matters most for pricing accuracy. Vega-weighting achieves exactly this.

    This loss function also has a practical link to price-space:
      Using a Taylor expansion: dPrice ≈ Vega * dSigma
      So: (Price error)^2 ≈ Vega^2 * (IV error)^2
      Minimising L5 ≈ minimising price errors, but in a more stable way than L1.

    Parameters
    ----------
    model_ivs  : array of model-predicted implied volatilities
    market_ivs : array of observed implied volatilities
    vegas      : array of Black-Scholes vega values (from implied_vol.py)

    Returns
    -------
    float : vega-weighted mean squared IV error
    """
    iv_errors = model_ivs - market_ivs

    # Weight each squared IV error by the squared vega of that option
    # Squaring vega ensures all weights are positive
    weighted_errors = (vegas ** 2) * (iv_errors ** 2)

    return float(np.mean(weighted_errors))


# =============================================================================
# SECTION 3: UNIFIED DISPATCHER
# =============================================================================

def compute_loss(loss_id: str, df: pd.DataFrame) -> float:
    """
    Unified loss function dispatcher.

    Given a loss function ID and a DataFrame with the required columns,
    computes and returns the scalar loss value.

    This is the main function called by estimation.py (during fitting)
    and evaluation.py (during out-of-sample evaluation).

    Required columns in df:
      - ModelPrice  : model-predicted option price (from dvf_models.py)
      - MidPrice    : observed market mid-price
      - ModelSigma  : model-predicted IV (= DVF sigma)
      - IV          : observed implied volatility (from implied_vol.py)
      - Vega        : Black-Scholes vega (from implied_vol.py) [needed for L5]

    Parameters
    ----------
    loss_id : one of 'L1', 'L2', 'L3', 'L4', 'L5'
    df      : DataFrame with the required columns

    Returns
    -------
    float : scalar loss value
    """

    # Extract arrays from DataFrame for efficient computation
    model_prices  = df["ModelPrice"].values
    market_prices = df["MidPrice"].values
    model_ivs     = df["ModelSigma"].values   # DVF predicted sigma = model IV
    market_ivs    = df["IV"].values           # observed IV from brentq inversion

    if loss_id == "L1":
        # MSE on prices — dollar-space, weights expensive options more
        return loss_mse_price(model_prices, market_prices)

    elif loss_id == "L2":
        # MSE on IV — vol-space, equal weight to all options
        return loss_mse_iv(model_ivs, market_ivs)

    elif loss_id == "L3":
        # MAE on prices — robust version of L1
        return loss_mae_price(model_prices, market_prices)

    elif loss_id == "L4":
        # MAE on IV — robust version of L2
        return loss_mae_iv(model_ivs, market_ivs)

    elif loss_id == "L5":
        # Vega-weighted IV MSE — the Christoffersen & Jacobs recommendation
        # Requires the Vega column produced by implied_vol.py
        if "Vega" not in df.columns:
            raise KeyError("Column 'Vega' not found. Run implied_vol.py first.")
        vegas = df["Vega"].values
        return loss_vega_weighted_ivmse(model_ivs, market_ivs, vegas)

    else:
        raise ValueError(f"Unknown loss_id '{loss_id}'. Choose from: L1, L2, L3, L4, L5")


# =============================================================================
# SECTION 4: COMPUTE ALL LOSSES AT ONCE
# =============================================================================

def compute_all_losses(df: pd.DataFrame) -> dict:
    """
    Computes all five loss functions on the given DataFrame at once.
    Returns a dictionary {loss_id: loss_value}.

    Useful for the evaluation matrix in evaluation.py — after fitting a model
    with one loss function, you evaluate it with all five loss functions.

    Parameters
    ----------
    df : DataFrame with ModelPrice, MidPrice, ModelSigma, IV, Vega columns

    Returns
    -------
    dict : {'L1': ..., 'L2': ..., 'L3': ..., 'L4': ..., 'L5': ...}
    """
    results = {}
    for loss_id in LOSS_FUNCTIONS.keys():
        try:
            results[loss_id] = compute_loss(loss_id, df)
        except Exception as e:
            # If a loss can't be computed (e.g. missing Vega column), store NaN
            results[loss_id] = np.nan
            print(f"  ⚠ Could not compute {loss_id}: {e}")
    return results


# =============================================================================
# SECTION 5: QUICK TEST — run this file directly
# =============================================================================

if __name__ == "__main__":
    print("\n=== Loss Functions — Quick Test ===\n")

    # Create a small synthetic dataset to verify loss functions work correctly
    np.random.seed(42)
    N = 100   # 100 synthetic options

    # Simulate market prices and IVs
    market_prices = np.random.uniform(10, 200, N)
    market_ivs    = np.random.uniform(0.10, 0.40, N)
    vegas         = np.random.uniform(0.5, 50, N)

    # Simulate model predictions with small errors added
    model_prices  = market_prices  + np.random.normal(0, 2, N)
    model_ivs     = market_ivs     + np.random.normal(0, 0.01, N)

    # Build a DataFrame as estimation.py and evaluation.py would provide
    df_test = pd.DataFrame({
        "MidPrice":   market_prices,
        "IV":         market_ivs,
        "Vega":       vegas,
        "ModelPrice": model_prices,
        "ModelSigma": model_ivs,
    })

    print(f"{'Loss ID':<6} {'Description':<52} {'Value':>12}")
    print("-" * 72)

    for loss_id, description in LOSS_FUNCTIONS.items():
        value = compute_loss(loss_id, df_test)
        print(f"{loss_id:<6} {description:<52} {value:>12.6f}")

    print("\n✓ All loss functions functional.")
    print("\nFunctions available for import:")
    print("  from loss_functions import compute_loss       # single loss value")
    print("  from loss_functions import compute_all_losses # all 5 losses at once")
    print("  from loss_functions import LOSS_FUNCTIONS     # loss ID → description")
