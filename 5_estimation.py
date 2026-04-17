"""
5_estimation.py — DVF Model Estimation
Fits 10 combinations (5 models × 2 loss functions) on training data (70%)
using L-BFGS-B optimisation. Saves parameters and in-sample losses.

Outputs: fitted_params.csv, insample_losses.csv, options_train.csv, options_test.csv
"""

import os, sys, importlib, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_dvf  = importlib.import_module("3_dvf_models")
_loss = importlib.import_module("4_loss_functions")

PROC_DIR    = os.path.join(_HERE, "DataSet", "data", "processed")
TRAIN_RATIO = 0.7
N_STARTS    = 5
RNG_SEED    = 42

# Parameter bounds: a0 (intercept) > 0 since it anchors ATM vol level;
# other coefficients allowed in [-5, 5] to accommodate log-moneyness scale
BOUNDS = {
    "M0": [(0.001, 5.0)],
    "M1": [(0.001, 5.0), (-5.0, 5.0)],
    "M2": [(0.001, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
    "M3": [(0.001, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
    "M4": [(0.001, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
}


def _fit_one(model_id: str, loss_id: str, arrays: dict, mean_iv: float,
             rng: np.random.Generator) -> tuple:
    """
    Fit a single (model_id, loss_id) pair using L-BFGS-B with N_STARTS restarts.
    Returns (best_result, n_params).
    """
    n_params = len(BOUNDS[model_id])

    def obj(p):
        sigma_hat = _dvf.predict_sigma(p, arrays["K"], arrays["T"], model_id)
        return _loss.compute_loss(loss_id,
                                  sigma_hat,
                                  arrays["IV"],
                                  arrays["Vega"],
                                  arrays["S"])

    best_res = None
    for start_idx in range(N_STARTS):
        if start_idx == 0:
            x0 = np.zeros(n_params)
            x0[0] = mean_iv   # a0 initialised at mean IV — most informative single start
        else:
            x0 = rng.uniform(0.01, 0.5, size=n_params)
            x0[0] = float(np.clip(rng.normal(mean_iv, 0.05), 0.01, 0.8))

        res = minimize(obj, x0, method="L-BFGS-B", bounds=BOUNDS[model_id],
                       options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8})

        if best_res is None or res.fun < best_res.fun:
            best_res = res

    return best_res, n_params

def main():
    df = pd.read_csv(os.path.join(PROC_DIR, "options_with_iv.csv"),
                     parse_dates=["ObsDate", "ExDt"])

    # ── Train/test split by date (no look-ahead bias) ─────────────────────────
    dates  = sorted(df["ObsDate"].unique())
    cutoff = dates[max(1, int(len(dates) * TRAIN_RATIO)) - 1]
    train  = df[df["ObsDate"] <= cutoff].copy()
    test   = df[df["ObsDate"] >  cutoff].copy()
    train.to_csv(os.path.join(PROC_DIR, "options_train.csv"), index=False)
    test.to_csv( os.path.join(PROC_DIR, "options_test.csv"),  index=False)
    print(f"Train: {len(train)} rows  |  Test: {len(test)} rows  "
          f"|  Cutoff: {str(cutoff)[:10]}\n")

    # ── Pre-extract arrays (no DataFrame access inside optimiser) ─────────────
    # Note: predict_sigma receives K/S (moneyness); log is taken inside the model.
    arrays = {
        "K":    (train["Strike"] / train["S0"]).values,
        "T":    train["T"].values,
        "IV":   train["IV"].values,
        "Vega": train["Vega"].values,
        "S":    train["S0"].values,
    }
    mean_iv = float(arrays["IV"].mean())

    rng = np.random.default_rng(RNG_SEED)

    # ── Fit all 10 combinations ────────────────────────────────────────────────
    param_rows, loss_rows = [], []

    for model_id in BOUNDS:
        for loss_id in _loss.LOSS_FUNCTIONS:
            res, n_params = _fit_one(model_id, loss_id, arrays, mean_iv, rng)

            # Pad params to fixed width 5 for consistent CSV schema
            p = np.full(5, np.nan)
            p[:n_params] = res.x

            param_rows.append({
                "model_id": model_id, "loss_id": loss_id,
                "a0": p[0], "a1": p[1], "a2": p[2], "a3": p[3], "a4": p[4],
                "converged": res.success,
            })
            n_obs = len(arrays["IV"])
            ll = -0.5 * n_obs * (np.log(2 * np.pi * res.fun) + 1)
            aic = -2 * ll + 2 * n_params
            bic = -2 * ll + np.log(n_obs) * n_params

            loss_rows.append({
                "model_id": model_id, "loss_id": loss_id,
                "insample_loss": res.fun,
                "AIC": aic,
                "BIC": bic,
            })
            status = "OK" if res.success else "NOT CONVERGED"
            print(f"  {model_id} × {loss_id}  loss={res.fun:.8f}  [{status}]  "
                  f"(best of {N_STARTS} starts)")

    # ── Save ───────────────────────────────────────────────────────────────────
    pd.DataFrame(param_rows).to_csv(os.path.join(PROC_DIR, "fitted_params.csv"),   index=False)
    pd.DataFrame(loss_rows).to_csv( os.path.join(PROC_DIR, "insample_losses.csv"), index=False)

    # ── In-sample summary table ────────────────────────────────────────────────
    df_loss = pd.DataFrame(loss_rows)

    tbl = df_loss.pivot(index="model_id", columns="loss_id", values="insample_loss")
    print(f"\n── In-Sample Loss (lower = better) ──\n"
          f"{tbl.to_string(float_format='{:.8f}'.format)}")

    for ic in ["AIC", "BIC"]:
        tbl_ic = df_loss.pivot(index="model_id", columns="loss_id", values=ic)
        print(f"\n── {ic} (lower = better, penalises complexity) ──\n"
              f"{tbl_ic.to_string(float_format='{:.2f}'.format)}")

    print(f"\n  Note: L5 = mean((vega/S)^2 * IV_err^2)  — comparable scale to L2.")

if __name__ == "__main__":
    main()
