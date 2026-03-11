"""
evaluation.py
=============
Step 6: Out-of-Sample Evaluation

Takes the 10 fitted (model x loss) parameter sets from estimation.py and
evaluates each on the held-out test data using BOTH L2 and L5.

The result is a 2x2 evaluation matrix per model:

                  Evaluated with L2 | Evaluated with L5
  Estimated with L2       A         |        B
  Estimated with L5       C         |        D

  A, D = diagonal (same loss for estimation and evaluation)
  B, C = off-diagonal (different losses) — this is the key comparison

If B and C differ a lot from A and D, the choice of loss function changes
which model appears best — that is the central finding to discuss in the report.

Inputs:  data/processed/options_test.csv
         data/processed/fitted_params.csv
Outputs: data/processed/oos_loss_matrix.csv
         data/processed/oos_all_losses.csv
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from loss_functions import LOSS_FUNCTIONS, compute_all_losses

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_TEST_PATH    = "data/processed/options_test.csv"
INPUT_PARAMS_PATH  = "data/processed/fitted_params.csv"
OUTPUT_MATRIX_PATH = "data/processed/oos_loss_matrix.csv"
OUTPUT_DETAIL_PATH = "data/processed/oos_all_losses.csv"

# Model order for display
MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4"]


# ── Load fitted parameters ─────────────────────────────────────────────────────

def load_params(filepath):
    """
    Loads the fitted parameters CSV and returns a nested dict:
        params[model_id][loss_id] = np.array([a0, a1, ...])

    Only loads the non-NaN parameters for each model (e.g. M0 only has a0).
    """
    df = pd.read_csv(filepath)
    params = {}

    for _, row in df.iterrows():
        mid = row["model_id"]
        lid = row["loss_id"]

        # Collect a0..a4 and drop NaN values (unused parameter slots)
        p = np.array([row[f"a{i}"] for i in range(5)])
        p = p[~np.isnan(p)]   # remove NaN padding

        if mid not in params:
            params[mid] = {}
        params[mid][lid] = p

    return params


# ── Vectorised DVF sigma (same as estimation.py) ───────────────────────────────

def dvf_sigma(params, K, T, model_id):
    """
    Predicts sigma for arrays K and T using the fitted DVF polynomial.
    Must match the exact same formulas used in estimation.py.
    """
    if   model_id == "M0": sigma = np.full_like(K, params[0])
    elif model_id == "M1": sigma = params[0] + params[1] * K
    elif model_id == "M2": sigma = params[0] + params[1] * K + params[2] * K**2
    elif model_id == "M3": sigma = params[0] + params[1] * K + params[2] * K**2 + params[3] * T
    elif model_id == "M4": sigma = (params[0] + params[1] * K + params[2] * K**2
                                    + params[3] * T + params[4] * K * T)
    return np.clip(sigma, 0.001, 5.0)


# ── Evaluate one fitted model on the test set ──────────────────────────────────

def evaluate_one(model_id, est_loss_id, params, arrays):
    """
    Applies the fitted DVF model to the test set and computes both L2 and L5.

    Steps:
      1. Predict sigma for every test option using the fitted params
      2. The predicted sigma IS the model IV (that's what DVF defines)
      3. Compare model IVs to observed market IVs using L2 and L5

    Returns a dict: {model_id, est_loss_id, L2, L5}
    """
    # Predict sigma on test set (vectorised — no loops)
    model_ivs = dvf_sigma(params, arrays["K"], arrays["T"], model_id)

    # Compute both evaluation losses
    losses = compute_all_losses(model_ivs, arrays["IV"], arrays["Vega"])

    return {
        "model_id":    model_id,
        "est_loss_id": est_loss_id,
        **losses,        # unpacks L2 and L5 into the dict
    }


# ── Evaluate all 10 combinations ──────────────────────────────────────────────

def evaluate_all(params_dict, arrays):
    """
    Loops over all 10 (model, estimation_loss) combinations and evaluates
    each on the test set with both L2 and L5.

    Returns a DataFrame with columns: model_id | est_loss_id | L2 | L5
    """
    rows  = []
    total = len(MODEL_ORDER) * len(LOSS_FUNCTIONS)
    count = 0

    for model_id in MODEL_ORDER:
        for est_loss_id in LOSS_FUNCTIONS:
            count += 1
            p      = params_dict[model_id][est_loss_id]
            result = evaluate_one(model_id, est_loss_id, p, arrays)
            rows.append(result)
            print(f"  [{count:2d}/{total}] {model_id} (est:{est_loss_id})  "
                  f"L2={result['L2']:.8f}  L5={result['L5']:.8f}")

    return pd.DataFrame(rows)


# ── Build and print the 2x2 matrix per model ──────────────────────────────────

def print_comparison_matrix(df_results):
    """
    For each model, prints the 2x2 matrix:

                  Eval L2 | Eval L5
      Est L2        A     |    B
      Est L5        C     |    D

    Diagonal (A, D): model estimated and evaluated with the SAME loss.
    Off-diagonal (B, C): different estimation and evaluation loss.

    Discussion point for the report:
      - If A << C for Eval-L2: models fitted with L2 perform better when
        judged by L2 than models fitted with L5 → loss choice biases rankings.
      - The magnitude of B vs D tells you how much it hurts to use the
        "wrong" estimation loss when your true goal is to minimise L5.
    """
    print("\n" + "=" * 60)
    print("Out-of-Sample Evaluation Matrix (2x2 per model)")
    print("Rows = estimation loss | Columns = evaluation loss")
    print("=" * 60)

    for model_id in MODEL_ORDER:
        subset = df_results[df_results["model_id"] == model_id]
        matrix = (subset.set_index("est_loss_id")[["L2", "L5"]]
                        .reindex(["L2", "L5"]))

        print(f"\n  Model {model_id}")
        print(f"  {'':20s}  {'Eval L2':>14}  {'Eval L5':>14}")
        print(f"  {'-'*48}")
        for est_lid in ["L2", "L5"]:
            row = matrix.loc[est_lid]
            # Mark the diagonal cell (matched estimation/evaluation)
            l2_str = f"[{row['L2']:.6f}]" if est_lid == "L2" else f" {row['L2']:.6f} "
            l5_str = f"[{row['L5']:.6f}]" if est_lid == "L5" else f" {row['L5']:.6f} "
            print(f"  Est {est_lid:16s}  {l2_str:>14}  {l5_str:>14}")
        print(f"  (diagonal [ ] = matched estimation/evaluation loss)")


def print_rankings(df_results):
    """
    For each evaluation loss, ranks the 5 models by average OOS performance.
    Shows whether the ranking changes between L2 and L5 evaluation — if it does,
    that is direct evidence that loss function choice matters.
    """
    print("\n── Model Rankings by Evaluation Loss ──────────────────────")
    for eval_loss in ["L2", "L5"]:
        avg    = df_results.groupby("model_id")[eval_loss].mean()
        ranked = avg.sort_values()
        rank_str = "  >  ".join([f"{m} ({v:.6f})" for m, v in ranked.items()])
        print(f"\n  Eval {eval_loss}: {rank_str}")
    print("\n  If rankings differ between L2 and L5 → loss function choice matters.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Step 6: Out-of-Sample Evaluation ===\n")

    # Load test data and fitted params
    df_test     = pd.read_csv(INPUT_TEST_PATH, parse_dates=["ObsDate", "ExDt"])
    params_dict = load_params(INPUT_PARAMS_PATH)
    print(f"Test set: {len(df_test)} options.\n")

    # Pre-extract test arrays once (same pattern as estimation.py)
    arrays = {
        "K":    df_test["Strike"].values,
        "T":    df_test["T"].values,
        "IV":   df_test["IV"].values,
        "Vega": df_test["Vega"].values,
    }

    # Evaluate all 10 combinations
    print("Evaluating all 10 (model x estimation loss) combinations...\n")
    df_results = evaluate_all(params_dict, arrays)

    # Print results
    print_comparison_matrix(df_results)
    print_rankings(df_results)

    # Save outputs
    df_results.to_csv(OUTPUT_DETAIL_PATH, index=False)

    # Also save as a multi-index matrix for easy import in visualization.py
    matrix = df_results.set_index(["model_id", "est_loss_id"])[["L2", "L5"]]
    matrix.to_csv(OUTPUT_MATRIX_PATH)

    print(f"\nSaved: {OUTPUT_DETAIL_PATH}")
    print(f"Saved: {OUTPUT_MATRIX_PATH}")
    print("\nDone. Run visualization.py next.")


if __name__ == "__main__":
    main()
