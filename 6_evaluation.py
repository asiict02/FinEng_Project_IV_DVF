"""
evaluation.py
=============
Step 6: Out-of-Sample Evaluation

This script takes every set of fitted parameters from estimation.py and
evaluates them on the held-out (out-of-sample) test data using ALL five
loss functions — not just the one used for estimation.

The result is the 5×5 evaluation matrix: for each (estimation loss, evaluation loss)
pair, we get an out-of-sample loss value. This is the central empirical result
of Christoffersen & Jacobs (2004) and the main table in your report.

Key finding to look for:
  - Diagonal cells = model estimated and evaluated with the SAME loss.
  - Off-diagonal cells = estimation and evaluation use DIFFERENT losses.
  - If rankings change dramatically off-diagonal, the loss function matters a lot.

Inputs:
    data/processed/options_test.csv      (produced by estimation.py)
    data/processed/fitted_params.csv     (produced by estimation.py)

Outputs:
    data/processed/oos_loss_matrix.csv   ← the main result table
    data/processed/oos_all_losses.csv    ← full detail, one row per (model, est_loss, eval_loss)

Usage:
    python evaluation.py
"""

import warnings
import numpy as np
import pandas as pd

from dvf_models     import apply_model_to_df, MODEL_SPECS
from loss_functions import compute_loss, compute_all_losses, LOSS_FUNCTIONS

warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

INPUT_TEST_PATH   = "data/processed/options_test.csv"
INPUT_PARAMS_PATH = "data/processed/fitted_params.csv"
OUTPUT_MATRIX_PATH = "data/processed/oos_loss_matrix.csv"
OUTPUT_DETAIL_PATH = "data/processed/oos_all_losses.csv"


# =============================================================================
# SECTION 2: LOAD FITTED PARAMETERS
# =============================================================================

def load_fitted_params(filepath: str) -> dict:
    """
    Loads the fitted parameters CSV produced by estimation.py and converts
    it into a nested dictionary for easy lookup.

    Returns a dict structured as:
        params_dict[model_id][loss_id] = np.array([a0, a1, ...])

    For example:
        params_dict["M2"]["L3"] = array([0.25, -0.00004, 2e-9])

    This makes it easy to look up "what parameters did model M2 learn
    when estimated with loss L3?"
    """
    df_params = pd.read_csv(filepath)

    params_dict = {}

    for _, row in df_params.iterrows():
        model_id = row["model_id"]
        loss_id  = row["loss_id"]

        # Extract only the parameter columns (a0, a1, ...) for this model
        # MODEL_SPECS tells us the names of those columns
        param_names = MODEL_SPECS[model_id]["param_names"]

        # Build the parameter array in the correct order [a0, a1, a2, ...]
        params = np.array([row[name] for name in param_names])

        # Store in nested dict
        if model_id not in params_dict:
            params_dict[model_id] = {}
        params_dict[model_id][loss_id] = params

    return params_dict


# =============================================================================
# SECTION 3: EVALUATE ONE (MODEL, ESTIMATION LOSS) PAIR WITH ALL EVAL LOSSES
# =============================================================================

def evaluate_one(model_id: str, est_loss_id: str,
                 params: np.ndarray, df_test: pd.DataFrame) -> dict:
    """
    Takes one fitted (model, estimation_loss) pair and evaluates it on
    the out-of-sample test set using ALL five evaluation loss functions.

    Steps:
      1. Apply the fitted DVF model to the test set to get ModelSigma and ModelPrice.
      2. Compute all five loss functions on the test set predictions.

    Parameters
    ----------
    model_id    : 'M0' to 'M4'
    est_loss_id : loss function used during estimation ('L1' to 'L5')
    params      : fitted parameter array from estimation.py
    df_test     : out-of-sample test DataFrame

    Returns
    -------
    dict : {eval_loss_id: oos_loss_value} for all five eval losses,
           plus metadata (model_id, est_loss_id)
    """
    # Apply the DVF model with fitted params to get predictions on test data
    # This adds ModelSigma (predicted vol) and ModelPrice (predicted price) columns
    df_pred = apply_model_to_df(df_test, params, model_id)

    # Compute all five evaluation losses on the test predictions
    all_losses = compute_all_losses(df_pred)

    # Add metadata so we know which (model, estimation loss) produced these results
    result = {
        "model_id":   model_id,
        "est_loss_id": est_loss_id,
    }
    result.update(all_losses)   # adds L1, L2, L3, L4, L5 to the result dict

    return result


# =============================================================================
# SECTION 4: EVALUATE ALL 25 COMBINATIONS
# =============================================================================

def evaluate_all(params_dict: dict, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Loops over all 25 (model, estimation_loss) combinations and evaluates
    each on the test set with all 5 evaluation losses.

    Returns a DataFrame with 25 rows and 7 columns:
        model_id | est_loss_id | L1 | L2 | L3 | L4 | L5
        M0       | L1          | .. | .. | .. | .. | ..
        M0       | L2          | .. | .. | .. | .. | ..
        ...

    Parameters
    ----------
    params_dict : nested dict from load_fitted_params()
    df_test     : out-of-sample test DataFrame

    Returns
    -------
    pd.DataFrame with all OOS evaluation results
    """
    all_results = []
    total = len(MODEL_SPECS) * len(LOSS_FUNCTIONS)
    count = 0

    for model_id in MODEL_SPECS.keys():
        for est_loss_id in LOSS_FUNCTIONS.keys():
            count += 1
            print(f"  [{count}/{total}] Evaluating {model_id} (est: {est_loss_id})...", end=" ")

            # Get the fitted parameters for this (model, estimation loss) pair
            params = params_dict[model_id][est_loss_id]

            # Evaluate on test set with all 5 losses
            result = evaluate_one(model_id, est_loss_id, params, df_test)
            all_results.append(result)

            # Print a summary (show only the diagonal — same est/eval loss)
            diag_loss = result[est_loss_id]
            print(f"OOS {est_loss_id} = {diag_loss:.6f}")

    return pd.DataFrame(all_results)


# =============================================================================
# SECTION 5: BUILD THE COMPARISON MATRICES
# =============================================================================

def build_loss_matrix(df_results: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """
    For a single DVF model, builds the 5×5 loss matrix:
      rows    = estimation loss (what loss was used to fit the model)
      columns = evaluation loss (what loss we use to judge the model OOS)
      values  = out-of-sample loss

    This is the key table from Christoffersen & Jacobs (2004).

    How to read it:
      - Look at each ROW: these are all evaluations of models fitted with the same loss.
        The diagonal cell is the "natural" evaluation; others show cross-evaluation.
      - Look at each COLUMN: these compare models fitted with different losses
        but evaluated the same way. Rankings down each column reveal whether
        the choice of estimation loss changes which model "wins".

    Parameters
    ----------
    df_results : output of evaluate_all()
    model_id   : which model to build the matrix for (e.g. 'M2')

    Returns
    -------
    pd.DataFrame : 5×5 matrix
    """
    # Filter to just the rows for this model
    df_model = df_results[df_results["model_id"] == model_id].copy()

    # Pivot: rows = est_loss_id, columns = eval_loss_id (L1..L5), values = OOS loss
    matrix = df_model.pivot(
        index   = "est_loss_id",
        columns = None,           # we'll manually select eval loss columns
        values  = None
    )

    # Manually build the matrix by selecting the L1–L5 columns
    matrix = df_model.set_index("est_loss_id")[list(LOSS_FUNCTIONS.keys())]
    matrix = matrix.reindex(index=list(LOSS_FUNCTIONS.keys()))

    # Rename index and columns for clarity
    matrix.index.name   = "Estimated with \\ Evaluated with"
    matrix.columns.name = ""

    return matrix


def build_combined_matrix(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a single large comparison matrix across ALL models and all
    (estimation loss, evaluation loss) pairs.

    This gives a bird's-eye view of the full experiment.

    Structure:
      rows    = (model_id, est_loss_id) — 25 combinations
      columns = evaluation losses L1–L5
    """
    df_matrix = df_results.set_index(["model_id", "est_loss_id"])
    df_matrix = df_matrix[list(LOSS_FUNCTIONS.keys())]
    return df_matrix


# =============================================================================
# SECTION 6: PRINT FORMATTED TABLES
# =============================================================================

def print_model_matrix(matrix: pd.DataFrame, model_id: str) -> None:
    """Prints a formatted 5×5 loss matrix for one model."""
    print(f"\n── OOS Loss Matrix: {model_id} ({MODEL_SPECS[model_id]['description']}) ──")
    print("   Rows = estimation loss | Columns = evaluation loss\n")
    print(matrix.to_string(float_format="{:.6f}".format))
    print("\n  * Diagonal (●): same loss for estimation and evaluation.")
    print("  * Off-diagonal: cross-evaluation — key for Christoffersen & Jacobs discussion.")


def print_rankings(df_results: pd.DataFrame) -> None:
    """
    For each evaluation loss, prints the ranking of models M0–M4 by
    average OOS performance (averaged across all estimation losses).

    This shows whether the best model is stable across evaluation metrics,
    or whether rankings flip depending on which loss you use to judge them.
    """
    print("\n── Model Rankings by Evaluation Loss ────────────────────────────")
    print("   (Average OOS loss across all estimation losses; lower = better)\n")

    for eval_loss in LOSS_FUNCTIONS.keys():
        # Average OOS loss for each model across all estimation losses
        avg = (df_results.groupby("model_id")[eval_loss]
               .mean()
               .sort_values()   # sort ascending — lowest loss is best
               .reindex(list(MODEL_SPECS.keys())))  # reorder to M0...M4

        ranked = avg.sort_values()
        rank_str = " > ".join([f"{m}({v:.5f})" for m, v in ranked.items()])
        print(f"  {eval_loss}: {rank_str}")


# =============================================================================
# SECTION 7: MAIN
# =============================================================================

def main():
    print("\n=== Step 6: Out-of-Sample Evaluation ===\n")

    # ── Load test data ─────────────────────────────────────────────────────────
    print(f"Loading test data from {INPUT_TEST_PATH}...")
    df_test = pd.read_csv(INPUT_TEST_PATH, parse_dates=["ObsDate", "ExDt"])
    print(f"  Loaded {len(df_test)} out-of-sample options.\n")

    # ── Load fitted parameters ─────────────────────────────────────────────────
    print(f"Loading fitted parameters from {INPUT_PARAMS_PATH}...")
    params_dict = load_fitted_params(INPUT_PARAMS_PATH)
    print(f"  Loaded parameters for {len(params_dict)} models.\n")

    # ── Evaluate all 25 combinations ───────────────────────────────────────────
    print("Evaluating all 25 (model × estimation loss) combinations on test data...\n")
    df_results = evaluate_all(params_dict, df_test)

    # ── Print per-model matrices ───────────────────────────────────────────────
    for model_id in MODEL_SPECS.keys():
        matrix = build_loss_matrix(df_results, model_id)
        print_model_matrix(matrix, model_id)

    # ── Print model rankings ───────────────────────────────────────────────────
    print_rankings(df_results)

    # ── Save outputs ───────────────────────────────────────────────────────────
    # Save the full detail table (25 rows × 7 columns)
    df_results.to_csv(OUTPUT_DETAIL_PATH, index=False)

    # Save the combined matrix (indexed by model + estimation loss)
    df_combined = build_combined_matrix(df_results)
    df_combined.to_csv(OUTPUT_MATRIX_PATH)

    print(f"\n✓ OOS detail table saved → {OUTPUT_DETAIL_PATH}")
    print(f"✓ OOS matrix saved       → {OUTPUT_MATRIX_PATH}")
    print("\n=== Done. Next step: run visualization.py ===")


if __name__ == "__main__":
    main()
