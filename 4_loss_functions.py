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
    Dispatcher that computes a scalar loss for a given loss_id. L2 is the
    mean squared IV error weighted equally across all options. L5 is the
    vega-weighted IV MSE — errors on high-vega (ATM) options are penalised
    more heavily, reflecting their greater market impact. Both legs and spot
    arrays are required for L5; L2 ignores them.
    """
    err2 = (model_ivs - market_ivs) ** 2

    if loss_id == "L2":
        return float(np.mean(err2))

    elif loss_id == "L5":
        if vegas is None or spot is None:
            raise ValueError("Both 'vegas' and 'spot' arrays are required for L5.")
        vega_norm = vegas / spot             # Normalise by spot so the weight is scale-invariant across different price levels
        return float(np.mean(vega_norm**2 * err2))

    raise ValueError(f"Unknown loss_id '{loss_id}'. Use one of {list(LOSS_FUNCTIONS)}.")


def compute_all_losses(model_ivs: np.ndarray,
                       market_ivs: np.ndarray,
                       vegas: np.ndarray,
                       spot: np.ndarray) -> dict:
    """
    Convenience wrapper that computes both L2 and L5 in a single call by
    reusing the squared error array. Used during out-of-sample evaluation
    in 6_evaluation.py to populate the full 2x2 loss matrix per model.
    """
    err2      = (model_ivs - market_ivs) ** 2
    vega_norm = vegas / spot
    return {
        "L2": float(np.mean(err2)),
        "L5": float(np.mean(vega_norm**2 * err2)),
    }
