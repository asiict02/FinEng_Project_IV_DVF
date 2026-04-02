"""
4_loss_functions.py — Loss Functions (Christoffersen & Jacobs, 2004)

L2: IV-MSE       — mean( (sigma_model - sigma_market)^2 )
L5: Vega-IVMSE   — mean( (vega_i/S_i)^2 * (sigma_model - sigma_market)^2 )

Imported by: 5_estimation.py, 6_evaluation.py
"""

import numpy as np

# Iterable registry for estimation.py
LOSS_FUNCTIONS = {
    "L2": "IV-MSE       — mean( (sigma_model - sigma_market)^2 )",
    "L5": "Vega-IVMSE   — mean( (vega/S)^2 * (sigma_model - sigma_market)^2 )",
}

def compute_loss(loss_id: str,
                 model_ivs: np.ndarray,
                 market_ivs: np.ndarray,
                 vegas: np.ndarray = None,
                 spot: np.ndarray = None) -> float:
    """
    Dispatcher: returns the scalar loss for loss_id in {'L2', 'L5'}.

    Parameters
    ----------
    loss_id    : 'L2' or 'L5'
    model_ivs  : model-predicted implied volatilities (N,)
    market_ivs : market implied volatilities (N,)
    vegas      : Black-Scholes vega in dollar terms, i.e. ∂C/∂σ * S (N,)    [required for L5]
    spot       : underlying spot price S (N,)                               [required for L5]
    """
    err2 = (model_ivs - market_ivs) ** 2

    if loss_id == "L2":
        return float(np.mean(err2))

    elif loss_id == "L5":
        if vegas is None or spot is None:
            raise ValueError("Both 'vegas' and 'spot' arrays are required for L5.")
        vega_norm = vegas / spot
        return float(np.mean(vega_norm**2 * err2))

    raise ValueError(f"Unknown loss_id '{loss_id}'. Use one of {list(LOSS_FUNCTIONS)}.")


def compute_all_losses(model_ivs: np.ndarray,
                       market_ivs: np.ndarray,
                       vegas: np.ndarray,
                       spot: np.ndarray) -> dict:
    """
    Returns both L2 and L5 losses in one call.

    Parameters
    ----------
    model_ivs  : model-predicted implied volatilities (N,)
    market_ivs : market implied volatilities (N,)
    vegas      : Black-Scholes vega in dollar terms (N,)
    spot       : underlying spot price S (N,)
    """
    err2      = (model_ivs - market_ivs) ** 2
    vega_norm = vegas / spot
    return {
        "L2": float(np.mean(err2)),
        "L5": float(np.mean(vega_norm**2 * err2)),
    }
