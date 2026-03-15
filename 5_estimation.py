"""
5_estimation.py — DVF Model Estimation
Fits 10 combinations (5 models x 2 loss functions) on training data (70%)
using L-BFGS-B optimisation. Saves parameters and in-sample losses.
Outputs: fitted_params.csv, insample_losses.csv, options_train.csv, options_test.csv
"""

import os, sys, importlib, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)

_dvf  = importlib.import_module("3_dvf_models")
_loss = importlib.import_module("4_loss_functions")

PROC_DIR   = os.path.join(_HERE, "DataSet", "data", "processed")
TRAIN_RATIO = 0.7
BOUNDS = {
    "M0": [(0.001,5.0)],
    "M1": [(0.001,5.0),(-5.0,5.0)],
    "M2": [(0.001,5.0),(-5.0,5.0),(-5.0,5.0)],
    "M3": [(0.001,5.0),(-5.0,5.0),(-5.0,5.0),(-5.0,5.0)],
    "M4": [(0.001,5.0),(-5.0,5.0),(-5.0,5.0),(-5.0,5.0),(-5.0,5.0)],
}

def main():
    df = pd.read_csv(os.path.join(PROC_DIR, "options_with_iv.csv"), parse_dates=["ObsDate","ExDt"])

    # ── Train/test split by date (no look-ahead bias) ─────────────────────────
    dates   = sorted(df["ObsDate"].unique())
    cutoff  = dates[max(1, int(len(dates)*TRAIN_RATIO))-1]
    train   = df[df["ObsDate"] <= cutoff].copy()
    test    = df[df["ObsDate"] >  cutoff].copy()
    train.to_csv(os.path.join(PROC_DIR, "options_train.csv"), index=False)
    test.to_csv( os.path.join(PROC_DIR, "options_test.csv"),  index=False)
    print(f"Train: {len(train)} rows  |  Test: {len(test)} rows  |  Cutoff: {str(cutoff)[:10]}\n")

    # ── Pre-extract arrays (speed: no DataFrame access inside optimiser) ───────
    arrays = {"K": (train["Strike"]/train["S0"]).values,   # moneyness
              "T": train["T"].values,
              "IV": train["IV"].values,
              "Vega": train["Vega"].values}
    mean_iv = float(arrays["IV"].mean())

    # ── Fit all 10 combinations ────────────────────────────────────────────────
    param_rows, loss_rows = [], []
    for model_id in BOUNDS:
        for loss_id in _loss.LOSS_FUNCTIONS:
            x0    = np.zeros(len(BOUNDS[model_id])); x0[0] = mean_iv
            obj = lambda p, m=model_id, l=loss_id: _loss.compute_loss(l,
                      _dvf.predict_sigma(p, arrays["K"], arrays["T"], m),
                      arrays["IV"], arrays["Vega"])
            res   = minimize(obj, x0, method="L-BFGS-B", bounds=BOUNDS[model_id],
                             options={"maxiter":2000,"ftol":1e-8,"gtol":1e-8})
            p = np.full(5, np.nan); p[:len(res.x)] = res.x
            param_rows.append({"model_id":model_id,"loss_id":loss_id,
                                "a0":p[0],"a1":p[1],"a2":p[2],"a3":p[3],"a4":p[4],
                                "converged":res.success})
            loss_rows.append({"model_id":model_id,"loss_id":loss_id,"insample_loss":res.fun})
            print(f"  {model_id} x {loss_id}  loss={res.fun:.8f}  [{'OK' if res.success else 'NOT CONVERGED'}]")

    pd.DataFrame(param_rows).to_csv(os.path.join(PROC_DIR,"fitted_params.csv"),   index=False)
    pd.DataFrame(loss_rows).to_csv( os.path.join(PROC_DIR,"insample_losses.csv"), index=False)

    # ── In-sample summary table ────────────────────────────────────────────────
    tbl = pd.DataFrame(loss_rows).pivot(index="model_id",columns="loss_id",values="insample_loss")
    print(f"\n── In-Sample Loss (lower = better) ──\n{tbl.to_string(float_format='{:.8f}'.format)}")

if __name__ == "__main__":
    main()
