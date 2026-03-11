"""
estimation.py
=============
Step 5: DVF Model Estimation

Fits 10 combinations (5 models x 2 loss functions: L2 and L5) on the
in-sample training data using vectorised NumPy and L-BFGS-B optimisation.

Why only L2 and L5?
  L2 (IV-MSE) is the standard benchmark from Dumas et al. (1998).
  L5 (Vega-IVMSE) is the key recommendation from Christoffersen & Jacobs (2004).
  Together they let us answer: does the choice of loss function change which
  model wins? That is the central question of the project.

Inputs:  data/processed/options_with_iv.csv
Outputs: data/processed/fitted_params.csv      (10 rows: one per model x loss)
         data/processed/insample_losses.csv     (10 rows: in-sample loss values)
         data/processed/options_train.csv
         data/processed/options_test.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from loss_functions import LOSS_FUNCTIONS, compute_loss

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH  = "data/processed/options_with_iv.csv"
TRAIN_RATIO = 0.7    # first 70% of dates = train, last 30% = test
MAX_ITER    = 2000   # max optimiser iterations per fit
TOL         = 1e-8   # convergence tolerance

# Parameter bounds per model.
# a0 (intercept) must be positive — it represents a volatility level.
# a1..a4 are slope/curvature terms and can be negative (e.g. downward skew).
BOUNDS = {
    "M0": [(0.001, 5.0)],
    "M1": [(0.001, 5.0), (-1.0, 1.0)],
    "M2": [(0.001, 5.0), (-1.0, 1.0), (-1.0, 1.0)],
    "M3": [(0.001, 5.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
    "M4": [(0.001, 5.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
}


# ── Train / test split ─────────────────────────────────────────────────────────

def split(df):
    """
    Splits data into train and test sets by observation date.

    IMPORTANT: we split by date, not randomly.
    A random split would mix future dates into the training set,
    giving the model information it wouldn't have in real life (look-ahead bias).
    """
    dates  = sorted(df["ObsDate"].unique())
    cutoff = dates[max(1, int(len(dates) * TRAIN_RATIO)) - 1]

    # All dates up to and including cutoff go to train; the rest to test
    train = df[df["ObsDate"] <= cutoff].copy()
    test  = df[df["ObsDate"] >  cutoff].copy()

    print(f"  Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"  Cutoff date: {str(cutoff)[:10]}")
    return train, test


# ── Vectorised DVF sigma ───────────────────────────────────────────────────────

def dvf_sigma(params, K, T, model_id):
    """
    Returns predicted volatility for arrays of strikes K and maturities T.

    Using NumPy array operations means the entire column is computed in one
    shot instead of one row at a time — this is the main speed gain.
    The optimiser calls this function hundreds of times per fit, so speed matters.

    params   : 1-D array [a0, a1, ...] — the coefficients being optimised
    K, T     : NumPy arrays of strikes and maturities
    model_id : 'M0' to 'M4'
    """
    if model_id == "M0":
        # Flat (constant) vol — the Black-Scholes benchmark
        # np.full_like creates an array the same shape as K, filled with a0
        sigma = np.full_like(K, params[0])

    elif model_id == "M1":
        # Linear in strike: allows a simple tilt to the smile
        sigma = params[0] + params[1] * K

    elif model_id == "M2":
        # Quadratic in strike: can capture the smile curvature (U-shape or inverted)
        sigma = params[0] + params[1] * K + params[2] * K**2

    elif model_id == "M3":
        # Quadratic in K + linear in T: adds a term structure dimension
        # a3 > 0 means longer maturities have higher vol (normal term structure)
        sigma = params[0] + params[1] * K + params[2] * K**2 + params[3] * T

    elif model_id == "M4":
        # Full model: the K*T interaction term lets the smile slope vary with maturity
        # This is the most flexible model in Dumas et al. (1998)
        sigma = (params[0] + params[1] * K + params[2] * K**2
                 + params[3] * T  + params[4] * K * T)

    # Hard clip: sigma must be positive and finite for Black-Scholes to work
    return np.clip(sigma, 0.001, 5.0)


# ── Vectorised Black-Scholes price ─────────────────────────────────────────────

def bs_price_vec(S, K, T, r, q, sigma, is_call):
    """
    Computes Black-Scholes option prices for entire arrays in one call.

    This avoids looping over rows entirely — NumPy handles the vectorisation.

    is_call : boolean array (True = call, False = put)
    """
    # Standard BS d1 and d2 terms (all operations are element-wise on arrays)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Compute both call and put prices for all rows simultaneously
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    # np.where selects call price where is_call=True, put price otherwise
    return np.where(is_call, call, put)


# ── Objective function factory ─────────────────────────────────────────────────

def make_objective(model_id, loss_id, arrays):
    """
    Builds and returns the objective function for the optimiser.

    We use a "factory" pattern so that the arrays are captured once in a closure
    and reused across all iterations — the optimiser just calls objective(params).

    The key design choice: arrays are pre-extracted from the DataFrame BEFORE
    the optimiser starts, so each iteration only does pure NumPy maths.
    """
    # Unpack arrays — these are captured in the closure below
    S, K, T, r, q = arrays["S"], arrays["K"], arrays["T"], arrays["r"], arrays["q"]
    is_call        = arrays["is_call"]
    market_ivs     = arrays["IV"]       # observed IVs from implied_vol.py
    vegas          = arrays["Vega"]     # BS vega (needed for L5 weighting)

    def objective(params):
        # Step 1: DVF sigma prediction (vectorised — one array operation)
        sigma = dvf_sigma(params, K, T, model_id)

        # Step 2: For L2/L5, the model "IV" IS sigma directly.
        #         DVF models define volatility as a function of K and T,
        #         so we don't need to invert BS — sigma is already the model IV.
        model_ivs = sigma

        # Step 3: Compute the chosen loss (L2 or L5) as a scalar
        return compute_loss(loss_id, model_ivs, market_ivs, vegas)

    return objective


# ── Fit one (model, loss) pair ─────────────────────────────────────────────────

def fit_one(model_id, loss_id, arrays, mean_iv):
    """
    Runs the optimiser for one (model_id, loss_id) combination.

    Starting point: a0 = mean IV from data, all other params = 0.
    This is much better than starting from zero — the optimiser reaches
    convergence faster when a0 already approximates the average vol level.
    """
    n     = len(BOUNDS[model_id])
    x0    = np.zeros(n)
    x0[0] = mean_iv   # initialise intercept near the data's average IV

    result = minimize(
        fun     = make_objective(model_id, loss_id, arrays),
        x0      = x0,
        method  = "L-BFGS-B",   # gradient-based, supports parameter bounds
        bounds  = BOUNDS[model_id],
        options = {"maxiter": MAX_ITER, "ftol": TOL, "gtol": TOL},
    )

    return {"params": result.x, "converged": result.success, "loss": result.fun}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Step 5: Estimation (L2 + L5 only) ===\n")
    os.makedirs("data/processed", exist_ok=True)

    # Load data from implied_vol.py
    df = pd.read_csv(INPUT_PATH, parse_dates=["ObsDate", "ExDt"])
    print(f"Loaded {len(df)} options.\n")

    # Date-based train/test split
    df_train, df_test = split(df)
    df_train.to_csv("data/processed/options_train.csv", index=False)
    df_test.to_csv( "data/processed/options_test.csv",  index=False)

    # Extract all training columns as NumPy arrays ONCE.
    # The optimiser will call the objective hundreds of times —
    # reading from a dict of arrays is much faster than querying a DataFrame.
    arrays = {
        "S":       df_train["S0"].values,
        "K":       df_train["Strike"].values,
        "T":       df_train["T"].values,
        "r":       df_train["Rf"].values,
        "q":       df_train["q"].values,
        "is_call": (df_train["OptionType"] == "call").values,
        "IV":      df_train["IV"].values,
        "Vega":    df_train["Vega"].values,
    }
    mean_iv = arrays["IV"].mean()

    # Run all 10 fits (5 models x 2 losses)
    param_rows, loss_rows = [], []
    total, count = len(BOUNDS) * len(LOSS_FUNCTIONS), 0

    print(f"Fitting {total} combinations (5 models x 2 losses)...\n")

    for model_id in BOUNDS:
        for loss_id in LOSS_FUNCTIONS:
            count += 1
            res = fit_one(model_id, loss_id, arrays, mean_iv)
            status = "OK" if res["converged"] else "NOT CONVERGED"
            print(f"  [{count:2d}/{total}] {model_id} x {loss_id}  "
                  f"loss = {res['loss']:.8f}  [{status}]")

            # Pad params to 5 columns (a0..a4) — unused slots get NaN
            p = np.full(5, np.nan)
            p[:len(res["params"])] = res["params"]

            param_rows.append({
                "model_id":  model_id, "loss_id": loss_id,
                "a0": p[0],  "a1": p[1], "a2": p[2], "a3": p[3], "a4": p[4],
                "converged": res["converged"],
            })
            loss_rows.append({
                "model_id":      model_id,
                "loss_id":       loss_id,
                "insample_loss": res["loss"],
            })

    # Save outputs
    df_params = pd.DataFrame(param_rows)
    df_losses = pd.DataFrame(loss_rows)
    df_params.to_csv("data/processed/fitted_params.csv",   index=False)
    df_losses.to_csv("data/processed/insample_losses.csv", index=False)

    # Print summary table: rows = models, columns = L2 / L5
    table = df_losses.pivot(index="model_id", columns="loss_id", values="insample_loss")
    table = table.reindex(index=list(BOUNDS), columns=["L2", "L5"])
    print("\n── In-Sample Loss Table (lower = better) ──")
    print(table.to_string(float_format="{:.8f}".format))

    print("\nDone. Run evaluation.py next.")


if __name__ == "__main__":
    main()
