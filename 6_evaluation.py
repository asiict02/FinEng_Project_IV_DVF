"""
6_evaluation.py — Out-of-Sample Evaluation (Christoffersen & Jacobs, 2004)
Evaluates all 10 fitted (model x est_loss) combinations on the test set
using both L2 and L5, producing a 2x2 matrix per model.
"""

import os, sys, importlib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_dvf  = importlib.import_module("3_dvf_models")
_loss = importlib.import_module("4_loss_functions")

PROC_DIR = os.path.join(_HERE, "DataSet", "data", "processed")

# ── Load data ──────────────────────────────────────────────────────────────────
df_test   = pd.read_csv(os.path.join(PROC_DIR, "options_test.csv"),  parse_dates=["ObsDate","ExDt"])
df_params = pd.read_csv(os.path.join(PROC_DIR, "fitted_params.csv"))

print(f"Test set: {len(df_test)} options\n")

# ── Evaluate all 10 combinations ──────────────────────────────────────────────
rows = []
for _, p in df_params.iterrows():
    model_id, loss_id = p["model_id"], p["loss_id"]
    params = np.array([p[n] for n in _dvf.MODEL_SPECS[model_id]["param_names"]])

    df_pred    = _dvf.apply_model_to_df(df_test, params, model_id)
    all_losses = _loss.compute_all_losses(df_pred["ModelSigma"].values,
                                          df_pred["IV"].values,
                                          df_pred["Vega"].values)
    rows.append({"model_id": model_id, "est_loss": loss_id, **all_losses})
    print(f"  {model_id} (est:{loss_id})  L2={all_losses['L2']:.6f}  L5={all_losses['L5']:.6f}")

df_results = pd.DataFrame(rows)

# ── 2x2 matrix per model ───────────────────────────────────────────────────────
print("\n── 2×2 OOS Loss Matrices (rows=est loss, cols=eval loss) ──────────────")
for model_id in _dvf.MODEL_SPECS:
    matrix = (df_results[df_results["model_id"] == model_id]
              .set_index("est_loss")[["L2", "L5"]])
    print(f"\n  {model_id}  ({_dvf.MODEL_SPECS[model_id]['description']})")
    print(matrix.to_string(float_format="{:.6f}".format))

# ── Save ───────────────────────────────────────────────────────────────────────
df_results.to_csv(os.path.join(PROC_DIR, "oos_all_losses.csv"), index=False)
print(f"\n✓ Saved → {os.path.join(PROC_DIR, 'oos_all_losses.csv')}")
