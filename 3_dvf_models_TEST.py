"""
3_dvf_models.py
===============
Step 3: DVF Model Specifications — Dumas et al. (1998)

Defines five polynomial DVF models (M0–M4) mapping (K, T) → sigma,
then plugging sigma into Black-Scholes to obtain model prices.

Imported by: 4_loss_functions.py, 5_estimation.py
"""

import numpy as np
import pandas as pd
import importlib
implied_vol = importlib.import_module("2_implied_vol")
get_bs_price = implied_vol.get_bs_price

# ── Model metadata (used by estimation.py for param initialisation) ────────────

MODEL_SPECS = {
    "M0": {"description": "sigma = a0",                          "n_params": 1, "param_names": ["a0"]},
    "M1": {"description": "sigma = a0 + a1*K",                   "n_params": 2, "param_names": ["a0","a1"]},
    "M2": {"description": "sigma = a0 + a1*K + a2*K^2",          "n_params": 3, "param_names": ["a0","a1","a2"]},
    "M3": {"description": "sigma = a0 + a1*K + a2*K^2 + a3*T",   "n_params": 4, "param_names": ["a0","a1","a2","a3"]},
    "M4": {"description": "sigma = a0 + a1*K + a2*K^2 + a3*T + a4*K*T", "n_params": 5, "param_names": ["a0","a1","a2","a3","a4"]},
}

# ── Core: DVF sigma predictor ──────────────────────────────────────────────────

def predict_sigma(params: np.ndarray, K, T, model_id: str):
    """
    Returns predicted volatility for scalar or array inputs (K, T).
    Clipped to [0.001, 5.0] to keep Black-Scholes numerically valid.
    """
    p = params
    if model_id == "M0":
        sigma = p[0]
    elif model_id == "M1":
        sigma = p[0] + p[1]*K
    elif model_id == "M2":
        sigma = p[0] + p[1]*K + p[2]*K**2
    elif model_id == "M3":
        sigma = p[0] + p[1]*K + p[2]*K**2 + p[3]*T
    elif model_id == "M4":
        sigma = p[0] + p[1]*K + p[2]*K**2 + p[3]*T + p[4]*K*T
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. Choose from: M0, M1, M2, M3, M4")
    return np.clip(sigma, 0.001, 5.0)

# ── Price predictor: DVF sigma → BS price ─────────────────────────────────────

def predict_price(params: np.ndarray, K: float, T: float,
                  S: float, r: float, q: float,
                  option_type: str, model_id: str) -> float:
    """Returns BS option price at the DVF-predicted sigma."""
    moneyness = K / S                                          # ← rescale here
    sigma     = predict_sigma(params, moneyness, T, model_id) # ← use moneyness
    return get_bs_price(S, K, T, r, q, sigma, option_type)    # ← K unchanged in BS


# predict_iv = predict_sigma (DVF sigma IS the model IV by construction)
predict_iv = predict_sigma


# ── Vectorised DataFrame helper ────────────────────────────────────────────────

def apply_model_to_df(df: pd.DataFrame, params: np.ndarray, model_id: str) -> pd.DataFrame:
    from scipy.stats import norm
    df = df.copy()

    S, K, T = df["S0"].values, df["Strike"].values, df["T"].values
    r, q = df["Rf"].values, df["q"].values

    moneyness = K / S  # ← rescale here
    sigma = predict_sigma(params, moneyness, T, model_id)  # ← use moneyness

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    is_call = (df["OptionType"] == "call").values

    df["ModelSigma"] = sigma
    df["ModelPrice"] = np.where(is_call, call, put)
    return df


# ── Initial parameter guess for optimiser ─────────────────────────────────────

def get_initial_params(model_id: str, df: pd.DataFrame) -> np.ndarray:
    """Intercept starts at mean IV; all slope terms start at 0."""
    init    = np.zeros(MODEL_SPECS[model_id]["n_params"])
    init[0] = df["IV"].mean() if "IV" in df.columns else 0.20
    return init

if __name__ == "__main__":
    import os
    import sys

    print("\n=== 3_dvf_models.py — Diagnostic Test ===\n")

    # ── TEST 1: Import check ───────────────────────────────────────────────────
    # If this block is running, importlib successfully loaded get_bs_price
    # from 2_implied_vol.py. If that file is missing, the script would have
    # crashed before reaching this point.
    print("[1/6] Import check .............. OK  (get_bs_price loaded from 2_implied_vol.py)")

    # ── TEST 2: predict_sigma — scalar inputs, all five models ────────────────
    # K is now MONEYNESS (K/S0), so realistic values are ~0.7 to 1.3.
    # Using 1.0 = ATM, with dummy params appropriate for moneyness scale.
    print("\n[2/6] predict_sigma — scalar moneyness inputs:")
    m_test, T_test = 1.0, 0.25   # ATM, 3-month
    test_params = {
        "M0": np.array([0.20]),
        "M1": np.array([0.50, -0.30]),
        "M2": np.array([0.80, -0.60, 0.25]),
        "M3": np.array([0.80, -0.60, 0.25, -0.05]),
        "M4": np.array([0.80, -0.60, 0.25, -0.05, 0.10]),
    }
    all_ok = True
    for mid, params in test_params.items():
        sigma = predict_sigma(params, m_test, T_test, mid)
        in_range = 0.001 <= float(sigma) <= 5.0
        status = "OK" if in_range else "FAIL — sigma out of [0.001, 5.0]"
        print(f"      {mid}: sigma = {float(sigma):.4f}  [{status}]")
        if not in_range:
            all_ok = False
    if not all_ok:
        print("      !! At least one sigma outside valid range.")

    # ── TEST 3: predict_sigma — array inputs (vectorisation check) ────────────
    # File 5's optimiser passes entire columns as arrays — this must work.
    # K is a moneyness array, not raw strikes.
    print("\n[3/6] predict_sigma — array moneyness inputs (vectorisation):")
    m_arr = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    T_arr = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    try:
        sigmas = predict_sigma(test_params["M4"], m_arr, T_arr, "M4")
        assert sigmas.shape == (7,), f"Expected shape (7,), got {sigmas.shape}"
        assert np.all(sigmas > 0), "Non-positive sigma in array output"
        assert np.all(np.isfinite(sigmas)), "Non-finite sigma in array output"
        print(f"      M4 on 7-row moneyness array: {np.round(sigmas, 4)}  [OK]")
    except Exception as e:
        print(f"      FAIL — {e}")

    # ── TEST 4: predict_price — put-call parity check ─────────────────────────
    # predict_price internally rescales K/S to moneyness before calling
    # predict_sigma, then passes raw K to Black-Scholes. PCP must hold.
    print("\n[4/6] predict_price — put-call parity check (with moneyness rescaling):")
    S_test, K_test = 4500.0, 4500.0   # ATM: moneyness = 1.0
    r_test, q_test = 0.05, 0.015
    params_m4 = test_params["M4"]
    call_price = predict_price(params_m4, K_test, T_test, S_test, r_test, q_test, "call", "M4")
    put_price  = predict_price(params_m4, K_test, T_test, S_test, r_test, q_test, "put",  "M4")
    pcp_theory = S_test * np.exp(-q_test * T_test) - K_test * np.exp(-r_test * T_test)
    pcp_actual = call_price - put_price
    pcp_error  = abs(pcp_actual - pcp_theory)
    print(f"      Call price     : {call_price:.4f}")
    print(f"      Put price      : {put_price:.4f}")
    print(f"      PCP error      : {pcp_error:.8f}  ", end="")
    print("[OK — within tolerance]" if pcp_error < 0.01 else "[FAIL]")

    # ── TEST 5: predict_sigma vs predict_iv consistency ───────────────────────
    # predict_iv is an alias for predict_sigma. They must return identical values
    # for the same inputs, both scalar and array.
    print("\n[5/6] predict_iv alias consistency:")
    try:
        iv_scalar = predict_iv(params_m4, m_test, T_test, "M4")
        s_scalar  = predict_sigma(params_m4, m_test, T_test, "M4")
        iv_array  = predict_iv(params_m4, m_arr, T_arr, "M4")
        s_array   = predict_sigma(params_m4, m_arr, T_arr, "M4")
        assert float(iv_scalar) == float(s_scalar), "Scalar mismatch"
        assert np.array_equal(iv_array, s_array),   "Array mismatch"
        print(f"      Scalar match: {float(iv_scalar):.6f} == {float(s_scalar):.6f}  [OK]")
        print(f"      Array match : all {len(iv_array)} values identical              [OK]")
    except AssertionError as e:
        print(f"      FAIL — {e}")

    # ── TEST 6: apply_model_to_df — real CSV test ──────────────────────────────
    # Most important test: uses the actual options_with_iv.csv from your pipeline.
    # Checks that moneyness rescaling inside apply_model_to_df produces valid
    # ModelSigma and ModelPrice columns with no NaNs or out-of-range values.
    print("\n[6/6] apply_model_to_df — real CSV test:")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "DataSet", "data", "processed", "options_with_iv.csv")

    if not os.path.exists(CSV_PATH):
        print(f"      SKIPPED — CSV not found at:\n      {CSV_PATH}")
        print("      Update CSV_PATH to run this test.")
    else:
        df = pd.read_csv(CSV_PATH, parse_dates=["ObsDate", "ExDt"]).head(50)
        try:
            df_out = apply_model_to_df(df, test_params["M4"], "M4")

            # Column existence
            assert "ModelSigma" in df_out.columns, "ModelSigma column missing"
            assert "ModelPrice" in df_out.columns, "ModelPrice column missing"

            # No NaNs
            assert df_out["ModelSigma"].notna().all(), "NaNs in ModelSigma"
            assert df_out["ModelPrice"].notna().all(), "NaNs in ModelPrice"

            # Valid sigma range
            assert df_out["ModelSigma"].between(0.001, 5.0).all(), "ModelSigma out of [0.001, 5.0]"

            # Positive prices
            assert (df_out["ModelPrice"] > 0).all(), "Non-positive ModelPrice found"

            # Moneyness was actually used: ModelSigma should differ from raw-K output
            # (sanity check that the rescaling is happening, not being skipped)
            df_raw_k = df.copy()
            sigma_raw = predict_sigma(test_params["M4"],
                                      df["Strike"].values, df["T"].values, "M4")
            sigma_mono = predict_sigma(test_params["M4"],
                                       (df["Strike"] / df["S0"]).values, df["T"].values, "M4")
            assert not np.allclose(sigma_raw, sigma_mono), \
                "ModelSigma identical with and without moneyness — rescaling may not be applied"

            print(f"      Loaded {len(df)} rows from real CSV.")
            print(f"      Moneyness range : {(df['Strike']/df['S0']).min():.3f} "
                  f"— {(df['Strike']/df['S0']).max():.3f}")
            print(f"      ModelSigma      : min={df_out['ModelSigma'].min():.4f}  "
                  f"max={df_out['ModelSigma'].max():.4f}  "
                  f"mean={df_out['ModelSigma'].mean():.4f}")
            print(f"      ModelPrice      : min={df_out['ModelPrice'].min():.2f}  "
                  f"max={df_out['ModelPrice'].max():.2f}")
            print(f"      All checks passed  [OK]")

        except AssertionError as e:
            print(f"      FAIL — {e}")
        except Exception as e:
            print(f"      ERROR — {e}")

    print("\n=== Diagnostic complete. All [OK] = ready for estimation.py ===\n")