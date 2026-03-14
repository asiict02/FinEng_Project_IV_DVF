"""
4_loss_functions.py
===================
Step 4: Loss Functions — Christoffersen & Jacobs (2004)

L2: IV-MSE       — mean( (IV_model - IV_market)^2 )
L5: Vega-IVMSE   — mean( Vega^2 * (IV_model - IV_market)^2 )

Imported by: 5_estimation.py
"""

import numpy as np

# Iterable by estimation.py
LOSS_FUNCTIONS = {
    "L2": "IV-MSE      — mean( (IV_model - IV_market)^2 )",
    "L5": "Vega-IVMSE  — mean( Vega^2 * (IV_model - IV_market)^2 )",
}


def compute_loss(loss_id: str, model_ivs: np.ndarray, market_ivs: np.ndarray,
                 vegas: np.ndarray = None) -> float:
    """Dispatcher: returns the scalar loss for loss_id in {'L2', 'L5'}."""
    err2 = (model_ivs - market_ivs) ** 2
    if loss_id == "L2":
        return float(np.mean(err2))
    elif loss_id == "L5":
        if vegas is None:
            raise ValueError("vegas array required for L5.")
        return float(np.mean(vegas**2 * err2))
    raise ValueError(f"Unknown loss_id '{loss_id}'. Use 'L2' or 'L5'.")


def compute_all_losses(model_ivs: np.ndarray, market_ivs: np.ndarray,
                       vegas: np.ndarray) -> dict:
    """Returns both L2 and L5 in one call."""
    err2 = (model_ivs - market_ivs) ** 2
    return {
        "L2": float(np.mean(err2)),
        "L5": float(np.mean(vegas**2 * err2)),
    }
