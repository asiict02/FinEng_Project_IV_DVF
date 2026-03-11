"""
estimation.py
=============
Step 5: DVF Model Estimation (Fitting)

For every combination of DVF model (M0–M4) and loss function (L1–L5),
this script fits the model parameters on the IN-SAMPLE portion of the data
using numerical optimisation (scipy.optimize.minimize with L-BFGS-B).

The result is 5 models × 5 loss functions = 25 fitted parameter sets.

Inputs:
    data/processed/options_with_iv.csv   (produced by implied_vol.py)

Outputs:
    data/processed/fitted_params.csv     ← one row per (model, loss) pair
    data/processed/insample_losses.csv   ← in-sample loss for each pair

Usage:
    python estimation.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Import our own modules — all model and loss logic lives there
from dvf_models    import predict_sigma, apply_model_to_df, get_initial_params, MODEL_SPECS
from loss_functions import compute_loss, compute_all_losses, LOSS_FUNCTIONS

warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

INPUT_PATH          = "data/processed/options_with_iv.csv"
OUTPUT_PARAMS_PATH  = "data/processed/fitted_params.csv"
OUTPUT_LOSSES_PATH  = "data/processed/insample_losses.csv"

# In-sample / out-of-sample split ratio
# 0.7 means the first 70% of unique observation dates are used for training,
# the remaining 30% are reserved for out-of-sample evaluation in evaluation.py
TRAIN_RATIO = 0.7

# Optimiser settings
# L-BFGS-B: a gradient-based method that supports bounds on parameters.
# It's efficient for low-dimensional problems like ours (1–5 parameters).
OPTIMISER_METHOD = "L-BFGS-B"
MAX_ITER         = 2000    # maximum optimisation iterations
TOL              = 1e-8    # convergence tolerance

# Parameter bounds: keep sigma in a sensible range to avoid degenerate solutions.
# For all parameters we allow a wide range; the intercept a0 must stay positive.
# Bounds are expressed as (lower, upper) per parameter.
PARAM_BOUNDS = {
    "a0": (0.001, 5.0),   # intercept: must be positive (it's a volatility level)
    "a1": (-1.0,  1.0),   # strike slope: allow negative (downward skew)
    "a2": (-1.0,  1.0),   # strike curvature
    "a3": (-1.0,  1.0),   # maturity slope
    "a4": (-1.0,  1.0),   # interaction K*T
}


# =============================================================================
# SECTION 2: TRAIN / TEST SPLIT
# =============================================================================

def split_train_test(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """
    Splits the dataset into in-sample (training) and out-of-sample (test) sets
    based on observation dates, NOT randomly.

    Why date-based split (not random)?
      In finance, we always want the training set to come BEFORE the test set
      in time. Randomly shuffling would create look-ahead bias — the model
      would effectively "see the future" during training.

    How it works:
      1. Find all unique observation dates, sorted in ascending order.
      2. Take the first 70% of dates as training, last 30% as test.
      3. Return two DataFrames filtered by these date sets.

    Parameters
    ----------
    df          : full options DataFrame
    train_ratio : fraction of dates to use for training (default 0.7 = 70%)

    Returns
    -------
    df_train : in-sample DataFrame
    df_test  : out-of-sample DataFrame
    """
    # Get sorted unique observation dates
    all_dates = sorted(df["ObsDate"].unique())
    n_dates   = len(all_dates)

    # Split index: first n_train dates go to training
    n_train   = max(1, int(np.floor(n_dates * train_ratio)))
    train_dates = set(all_dates[:n_train])
    test_dates  = set(all_dates[n_train:])

    df_train = df[df["ObsDate"].isin(train_dates)].copy()
    df_test  = df[df["ObsDate"].isin(test_dates)].copy()

    print(f"  Train dates ({len(train_dates)}): {sorted([str(d)[:10] for d in train_dates])}")
    print(f"  Test  dates ({len(test_dates)}):  {sorted([str(d)[:10] for d in test_dates])}")
    print(f"  Train rows: {len(df_train)} | Test rows: {len(df_test)}")

    return df_train, df_test


# =============================================================================
# SECTION 3: OBJECTIVE FUNCTION FACTORY
# =============================================================================

def make_objective(model_id: str, loss_id: str, df: pd.DataFrame):
    """
    Creates and returns an objective function for the optimiser.

    Why a factory function?
      scipy.optimize.minimize requires a function f(params) → scalar.
      We need to create a different objective for each (model_id, loss_id) pair,
      while keeping df fixed. A factory function does this cleanly by using
      Python closures — the returned function "remembers" model_id, loss_id, df.

    Parameters
    ----------
    model_id : 'M0' to 'M4'
    loss_id  : 'L1' to 'L5'
    df       : training DataFrame

    Returns
    -------
    callable : objective(params) → scalar loss value
    """

    def objective(params: np.ndarray) -> float:
        """
        Given a parameter vector, apply the DVF model to all training options
        and compute the chosen loss function.

        This is what the optimiser minimises by adjusting params.
        """
        # Step 1: Apply DVF model to get ModelSigma and ModelPrice for every row
        # apply_model_to_df adds 'ModelSigma' and 'ModelPrice' columns
        df_pred = apply_model_to_df(df, params, model_id)

        # Step 2: Compute the scalar loss value
        loss_val = compute_loss(loss_id, df_pred)

        # Return the loss — the optimiser will try to make this as small as possible
        return loss_val

    return objective


# =============================================================================
# SECTION 4: SINGLE MODEL FITTING
# =============================================================================

def fit_model(model_id: str, loss_id: str, df_train: pd.DataFrame) -> dict:
    """
    Fits one (model_id, loss_id) combination on the training data.

    Steps:
      1. Get initial parameter values (smart starting point near mean IV)
      2. Define bounds for each parameter
      3. Call scipy.optimize.minimize with L-BFGS-B
      4. Return fitted parameters and diagnostics

    Parameters
    ----------
    model_id : 'M0' to 'M4'
    loss_id  : 'L1' to 'L5'
    df_train : training DataFrame

    Returns
    -------
    dict with keys: model_id, loss_id, params, insample_loss, success, n_iter
    """
    # ── 1. Initial parameters ──────────────────────────────────────────────────
    # Start near the average IV level in the data to give the optimiser
    # a sensible starting point (avoids slow convergence from 0)
    x0 = get_initial_params(model_id, df_train)

    # ── 2. Parameter bounds ────────────────────────────────────────────────────
    # Build a list of (lower, upper) bounds, one entry per parameter
    param_names = MODEL_SPECS[model_id]["param_names"]
    bounds = [PARAM_BOUNDS[name] for name in param_names]

    # ── 3. Objective function ──────────────────────────────────────────────────
    objective = make_objective(model_id, loss_id, df_train)

    # ── 4. Run the optimiser ───────────────────────────────────────────────────
    # L-BFGS-B = Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bounds.
    # It approximates the Hessian (second derivatives) to take efficient steps.
    # Good for smooth, low-dimensional problems like fitting 1–5 parameters.
    result = minimize(
        fun     = objective,       # function to minimise
        x0      = x0,              # starting parameter values
        method  = OPTIMISER_METHOD,
        bounds  = bounds,          # keep parameters in valid range
        options = {
            "maxiter": MAX_ITER,
            "ftol":    TOL,        # stop if loss improvement < TOL
            "gtol":    TOL,        # stop if gradient < TOL
        }
    )

    # ── 5. Evaluate fitted model on training data ──────────────────────────────
    # Recompute the loss at the optimal parameters to confirm the result
    df_fitted = apply_model_to_df(df_train, result.x, model_id)
    insample_loss = compute_loss(loss_id, df_fitted)

    return {
        "model_id":      model_id,
        "loss_id":       loss_id,
        "params":        result.x,        # fitted parameter array
        "insample_loss": insample_loss,   # loss at fitted params
        "converged":     result.success,  # did the optimiser converge?
        "n_iter":        result.nit,      # number of iterations taken
        "message":       result.message,  # optimiser status message
    }


# =============================================================================
# SECTION 5: FIT ALL MODEL × LOSS COMBINATIONS
# =============================================================================

def fit_all(df_train: pd.DataFrame) -> tuple:
    """
    Loops over all 25 (model, loss) combinations and fits each one.

    Returns
    -------
    results_list : list of dicts, one per (model, loss) pair
    df_params    : DataFrame with fitted parameters (one row per pair)
    df_losses    : DataFrame with in-sample losses (one row per pair)
    """
    results_list = []

    total = len(MODEL_SPECS) * len(LOSS_FUNCTIONS)
    count = 0

    for model_id in MODEL_SPECS.keys():
        for loss_id in LOSS_FUNCTIONS.keys():
            count += 1
            print(f"  [{count}/{total}] Fitting {model_id} with loss {loss_id}...", end=" ")

            result = fit_model(model_id, loss_id, df_train)
            results_list.append(result)

            status = "✓" if result["converged"] else "⚠ (not converged)"
            print(f"loss = {result['insample_loss']:.6f}  {status}")

    # ── Build params DataFrame ─────────────────────────────────────────────────
    # Each row = one (model, loss) pair; columns = model_id, loss_id, a0, a1, ...
    param_rows = []
    for r in results_list:
        row = {"model_id": r["model_id"], "loss_id": r["loss_id"]}
        param_names = MODEL_SPECS[r["model_id"]]["param_names"]
        for name, val in zip(param_names, r["params"]):
            row[name] = val
        row["converged"] = r["converged"]
        row["n_iter"]    = r["n_iter"]
        param_rows.append(row)

    df_params = pd.DataFrame(param_rows)

    # ── Build losses DataFrame ─────────────────────────────────────────────────
    # Simple matrix: rows = model, cols = loss, values = in-sample loss
    loss_rows = []
    for r in results_list:
        loss_rows.append({
            "model_id":      r["model_id"],
            "loss_id":       r["loss_id"],
            "insample_loss": r["insample_loss"],
        })
    df_losses = pd.DataFrame(loss_rows)

    return results_list, df_params, df_losses


# =============================================================================
# SECTION 6: DISPLAY IN-SAMPLE RESULTS TABLE
# =============================================================================

def print_insample_table(df_losses: pd.DataFrame) -> None:
    """
    Prints a formatted table of in-sample losses:
    rows = models (M0–M4), columns = loss functions (L1–L5).

    This mirrors Table 2 from Christoffersen & Jacobs (2004).
    """
    # Pivot to wide format: rows = model_id, columns = loss_id
    table = df_losses.pivot(index="model_id", columns="loss_id", values="insample_loss")
    table = table.reindex(index=list(MODEL_SPECS.keys()),
                          columns=list(LOSS_FUNCTIONS.keys()))

    print("\n── In-Sample Loss Table ──────────────────────────────────────────")
    print("   (rows = DVF model, columns = estimation loss function)\n")
    print(table.to_string(float_format="{:.6f}".format))
    print("\n  Lower = better fit on training data.")


# =============================================================================
# SECTION 7: MAIN
# =============================================================================

def main():
    print("\n=== Step 5: DVF Model Estimation ===\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["ObsDate", "ExDt"])
    print(f"  Loaded {len(df)} options with valid IV.\n")

    # ── Split train / test ─────────────────────────────────────────────────────
    print("Splitting into in-sample (train) and out-of-sample (test) sets...")
    df_train, df_test = split_train_test(df, TRAIN_RATIO)

    # Save the test set so evaluation.py can load it directly
    os.makedirs("data/processed", exist_ok=True)
    df_test.to_csv("data/processed/options_test.csv", index=False)
    df_train.to_csv("data/processed/options_train.csv", index=False)
    print(f"  ✓ Train set saved → data/processed/options_train.csv")
    print(f"  ✓ Test  set saved → data/processed/options_test.csv\n")

    # ── Fit all 25 (model, loss) combinations ─────────────────────────────────
    print("Fitting all 25 (model × loss) combinations on training data...\n")
    results_list, df_params, df_losses = fit_all(df_train)

    # ── Print in-sample results table ─────────────────────────────────────────
    print_insample_table(df_losses)

    # ── Save outputs ───────────────────────────────────────────────────────────
    df_params.to_csv(OUTPUT_PARAMS_PATH, index=False)
    df_losses.to_csv(OUTPUT_LOSSES_PATH, index=False)
    print(f"\n✓ Fitted parameters saved → {OUTPUT_PARAMS_PATH}")
    print(f"✓ In-sample losses saved  → {OUTPUT_LOSSES_PATH}")

    print("\n=== Done. Next step: run evaluation.py ===")


if __name__ == "__main__":
    main()
