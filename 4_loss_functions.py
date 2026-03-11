"""
loss_functions.py
=================
Step 4: Loss Function Implementations

We implement only L2 and L5 — the two academically meaningful loss functions
from Christoffersen & Jacobs (2004):

  L2: IV-MSE         — mean( (IV_model - IV_market)^2 )
      Standard benchmark. Treats all options equally in volatility space.

  L5: Vega-IVMSE     — mean( Vega^2 * (IV_model - IV_market)^2 )
      The paper's key recommendation. Weights errors by vega^2, giving more
      importance to near-the-money options where pricing accuracy matters most.

Why not L1/L3/L4?
  - L1 (price MSE) systematically overweights expensive options, introducing
    bias toward fitting long-dated deep ITM options at the expense of the smile.
  - L3/L4 (MAE variants) add robustness but don't change the core conclusion.
  - For a university project, L2 vs L5 is the central comparison that
    replicates Christoffersen & Jacobs' main finding.

This file is imported by estimation.py and evaluation.py.
"""

import numpy as np
import pandas as pd

# The two loss functions we use — a simple dict so other scripts can iterate over them
LOSS_FUNCTIONS = {
    "L2": "IV-MSE          — mean( (IV_model - IV_market)^2 )",
    "L5": "Vega-IVMSE      — mean( Vega^2 * (IV_model - IV_market)^2 )",
}


# ── L2: IV Mean Squared Error ──────────────────────────────────────────────────

def loss_iv_mse(model_ivs: np.ndarray, market_ivs: np.ndarray) -> float:
    """
    Computes mean squared error between model and market implied volatilities.

    All options are weighted equally — a 1% IV error on a deep OTM option
    counts the same as a 1% error on an ATM option.

    model_ivs  : array of predicted vols from the DVF model
    market_ivs : array of observed IVs from implied_vol.py
    """
    return float(np.mean((model_ivs - market_ivs) ** 2))


# ── L5: Vega-Weighted IV MSE ───────────────────────────────────────────────────

def loss_vega_ivmse(model_ivs: np.ndarray, market_ivs: np.ndarray,
                    vegas: np.ndarray) -> float:
    """
    Computes vega-weighted mean squared IV error.

    Why vega-weighting?
      Vega = dPrice/dSigma = how sensitive the option price is to a vol change.
      ATM options have the highest vega, so a vol error there causes the
      largest price mis-pricing. Weighting by vega^2 ensures we penalise
      errors where they hurt the most.

    Economic link to price errors (Taylor approximation):
      dPrice ≈ Vega * dSigma
      => (price error)^2 ≈ Vega^2 * (IV error)^2
      Minimising L5 ≈ minimising price errors, but more stably than L1.

    model_ivs  : array of predicted vols
    market_ivs : array of observed IVs
    vegas      : array of BS vega values (from implied_vol.py)
    """
    return float(np.mean(vegas**2 * (model_ivs - market_ivs) ** 2))


# ── Unified dispatcher ─────────────────────────────────────────────────────────

def compute_loss(loss_id: str, model_ivs: np.ndarray, market_ivs: np.ndarray,
                 vegas: np.ndarray = None) -> float:
    """
    Calls the right loss function given a loss_id string.

    loss_id    : 'L2' or 'L5'
    model_ivs  : predicted vols from DVF model
    market_ivs : observed market IVs
    vegas      : required for L5, ignored for L2
    """
    if loss_id == "L2":
        return loss_iv_mse(model_ivs, market_ivs)

    elif loss_id == "L5":
        if vegas is None:
            raise ValueError("vegas array required for L5.")
        return loss_vega_ivmse(model_ivs, market_ivs, vegas)

    else:
        raise ValueError(f"Unknown loss_id '{loss_id}'. Use 'L2' or 'L5'.")


# ── Compute both losses at once ────────────────────────────────────────────────

def compute_all_losses(model_ivs: np.ndarray, market_ivs: np.ndarray,
                       vegas: np.ndarray) -> dict:
    """
    Returns both L2 and L5 in one call.
    Used by evaluation.py after generating model predictions on the test set.
    """
    return {
        "L2": loss_iv_mse(model_ivs, market_ivs),
        "L5": loss_vega_ivmse(model_ivs, market_ivs, vegas),
    }


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Loss Functions — Quick Test ===\n")
    np.random.seed(42)
    N           = 200
    market_ivs  = np.random.uniform(0.10, 0.40, N)
    model_ivs   = market_ivs + np.random.normal(0, 0.01, N)
    vegas       = np.random.uniform(0.5, 50, N)

    losses = compute_all_losses(model_ivs, market_ivs, vegas)
    for lid, val in losses.items():
        print(f"  {lid}: {val:.8f}  —  {LOSS_FUNCTIONS[lid]}")

    print("\nImport in other scripts:")
    print("  from loss_functions import compute_loss, compute_all_losses, LOSS_FUNCTIONS")
