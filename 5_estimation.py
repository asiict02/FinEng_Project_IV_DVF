"""
5_estimation.py
===============
Step 5: DVF Model Estimation

Fits 10 combinations (5 models x 2 loss functions: L2, L5) on training data
using vectorised NumPy and L-BFGS-B optimisation.

Inputs:  <BASE_DIR>/DataSet/data/processed/options_with_iv.csv
Outputs: <BASE_DIR>/DataSet/data/processed/fitted_params.csv
         <BASE_DIR>/DataSet/data/processed/insample_losses.csv
         <BASE_DIR>/DataSet/data/processed/options_train.csv
         <BASE_DIR>/DataSet/data/processed/options_test.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Digit-prefixed imports (same pattern as 3_dvf_models.py) ──────────────────
# Python cannot import modules whose filenames start with a digit using the
# standard 'import' statement. We use importlib and add the script's own
# directory to sys.path so sibling files are always found regardless of the
# working directory PyCharm launches from.

import importlib

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_dvf   = importlib.import_module("3_dvf_models")
_loss  = importlib.import_module("4_loss_functions")

predict_sigma  = _dvf.predict_sigma
LOSS_FUNCTIONS = _loss.LOSS_FUNCTIONS
compute_loss   = _loss.compute_loss

# ── Paths ──────────────────────────────────────────────────────────────────────
# Anchored to the script's own location on disk — works regardless of which
# directory PyCharm (or the terminal) uses as the working directory.

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet")
PROC_DIR    = os.path.join(DATASET_DIR, "data", "processed")

INPUT_PATH   = os.path.join(PROC_DIR, "options_with_iv.csv")
TRAIN_PATH   = os.path.join(PROC_DIR, "options_train.csv")
TEST_PATH    = os.path.join(PROC_DIR, "options_test.csv")
PARAMS_PATH  = os.path.join(PROC_DIR, "fitted_params.csv")
LOSSES_PATH  = os.path.join(PROC_DIR, "insample_losses.csv")

# ── Configuration ──────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.7    # first 70% of observation dates → train, rest → test
MAX_ITER    = 2000   # maximum L-BFGS-B iterations per fit
TOL         = 1e-8   # convergence tolerance (ftol and gtol)

# Parameter bounds per model.
# a0 (intercept) must be strictly positive — it represents a volatility level.
# a1..a4 are slope/curvature/interaction terms and can be negative.
BOUNDS = {
    "M0": [(0.001, 5.0)],
    "M1": [(0.001, 5.0), (-5.0, 5.0)],
    "M2": [(0.001, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
    "M3": [(0.001, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
    "M4": [(0.001, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
}


# ── Train / test split by date (no look-ahead bias) ───────────────────────────

def split(df: pd.DataFrame):
    """
    Splits options into train and test by observation date.

    We split by date — not randomly — to avoid look-ahead bias: a random split
    would leak future prices into the training set, which would never happen
    in a real trading environment.
    """
    dates  = sorted(df["ObsDate"].unique())
    cutoff = dates[max(1, int(len(dates) * TRAIN_RATIO)) - 1]
    train  = df[df["ObsDate"] <= cutoff].copy()
    test   = df[df["ObsDate"] >  cutoff].copy()
    print(f"  Train : {len(train)} rows  ({df['ObsDate'].nunique()} dates total)")
    print(f"  Test  : {len(test)} rows")
    print(f"  Cutoff: {str(cutoff)[:10]}")
    return train, test


# ── Fit one (model, loss) combination ─────────────────────────────────────────

def fit_one(model_id: str, loss_id: str, arrays: dict, mean_iv: float):
    """
    Runs L-BFGS-B optimisation for one (model_id, loss_id) pair.

    The objective function is built as a closure over the pre-extracted NumPy
    arrays so each call does pure array arithmetic — no DataFrame access.
    This is the key speed optimisation: the optimiser calls objective() hundreds
    of times per fit, and each call must be as fast as possible.

    Returns
    -------
    params    : fitted coefficient array
    loss_val  : final (minimised) loss value
    converged : True if the optimiser reported successful convergence
    """
    K, T       = arrays["K"], arrays["T"]
    market_ivs = arrays["IV"]
    vegas      = arrays["Vega"]

    def objective(params):
        # predict_sigma returns a NumPy array of predicted vols (one per option)
        # For DVF models, predicted sigma IS the model IV — no BS inversion needed
        model_ivs = predict_sigma(params, K, T, model_id)
        return compute_loss(loss_id, model_ivs, market_ivs, vegas)

    # Initialise: intercept a0 at mean IV, all slope terms at 0
    x0    = np.zeros(len(BOUNDS[model_id]))
    x0[0] = mean_iv

    res = minimize(
        objective, x0,
        method  = "L-BFGS-B",
        bounds  = BOUNDS[model_id],
        options = {"maxiter": MAX_ITER, "ftol": TOL, "gtol": TOL},
    )
    return res.x, res.fun, res.success


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Step 5: Estimation (L2 + L5) ===\n")
    os.makedirs(PROC_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from:\n  {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=["ObsDate", "ExDt"])
    print(f"  Loaded {len(df)} options across {df['ObsDate'].nunique()} observation date(s).\n")

    # ── Train / test split ─────────────────────────────────────────────────────
    df_train, df_test = split(df)
    df_train.to_csv(TRAIN_PATH, index=False)
    df_test.to_csv( TEST_PATH,  index=False)
    print(f"\n  Saved train → {TRAIN_PATH}")
    print(f"  Saved test  → {TEST_PATH}\n")

    # ── Pre-extract arrays (done once before the optimiser loop) ───────────────
    # Reading from a dict of arrays is much faster inside the objective function
    # than querying a DataFrame column on every call.
    arrays = {
        "K": (df_train["Strike"] / df_train["S0"]).values,  # moneyness, ~0.7 to 1.3
        "T": df_train["T"].values,
        "IV": df_train["IV"].values,
        "Vega": df_train["Vega"].values,
    }
    mean_iv = float(arrays["IV"].mean())

    # ── Fit all 10 combinations ────────────────────────────────────────────────
    param_rows, loss_rows = [], []
    total = len(BOUNDS) * len(LOSS_FUNCTIONS)
    print(f"Fitting {total} combinations (5 models x 2 losses)...\n")

    for count, (model_id, loss_id) in enumerate(
        ((m, l) for m in BOUNDS for l in LOSS_FUNCTIONS), start=1
    ):
        params, loss_val, converged = fit_one(model_id, loss_id, arrays, mean_iv)
        status = "OK" if converged else "NOT CONVERGED"
        print(f"  [{count:2d}/{total}] {model_id} x {loss_id}  "
              f"loss = {loss_val:.8f}  [{status}]")

        # Pad params to 5 columns (a0..a4) — unused slots filled with NaN
        # This keeps the output CSV rectangular regardless of model complexity
        p = np.full(5, np.nan)
        p[:len(params)] = params

        param_rows.append({
            "model_id": model_id, "loss_id": loss_id,
            "a0": p[0], "a1": p[1], "a2": p[2], "a3": p[3], "a4": p[4],
            "converged": converged,
        })
        loss_rows.append({
            "model_id":      model_id,
            "loss_id":       loss_id,
            "insample_loss": loss_val,
        })

    # ── Save outputs ───────────────────────────────────────────────────────────
    df_params = pd.DataFrame(param_rows)
    df_losses = pd.DataFrame(loss_rows)
    df_params.to_csv(PARAMS_PATH, index=False)
    df_losses.to_csv(LOSSES_PATH, index=False)
    print(f"\n  Saved fitted params  → {PARAMS_PATH}")
    print(f"  Saved in-sample loss → {LOSSES_PATH}")

    # ── Summary pivot table ────────────────────────────────────────────────────
    table = df_losses.pivot(index="model_id", columns="loss_id", values="insample_loss")
    table = table.reindex(index=list(BOUNDS), columns=["L2", "L5"])
    print("\n── In-Sample Loss Table (lower = better) ──────────────────")
    print(table.to_string(float_format="{:.8f}".format))
    print("\nDone. Run evaluation.py next.")


if __name__ == "__main__":
    main()