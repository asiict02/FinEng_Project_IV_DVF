"""
2_implied_vol.py
==============
Step 2: Implied Volatility Extraction

Reads the cleaned option data produced by data_collection.py and computes
the implied volatility (IV) for each option by numerically inverting the
Black-Scholes formula.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ── 0. PATHS ───────────────────────────────────────────────────────────────────
# Mirrors the path construction in data_collection.py so this script resolves
# correctly regardless of the working directory it is launched from.

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet")

INPUT_PATH  = os.path.join(DATASET_DIR, "data/processed/options_final.csv")
OUTPUT_PATH = os.path.join(DATASET_DIR, "data/processed/options_with_iv.csv")

# ── 1. CONFIGURATION ───────────────────────────────────────────────────────────

IV_MIN    = 0.01           # lower bracket for Brent's solver — must match 1_data_collection.py !!
IV_MAX    = 5.00           # 500% — discard solutions above this
IV_INIT   = 0.25           # 25%  — kept for reference; not used by brentq
PRICE_TOL = 1e-6           # tolerance for BS price vs market price match

# NOTE ON IV_data COLUMN
# data_collection.py now passes through a column called IV_data (Refinitiv IV).
# This column is carried along untouched — it is for reference / benchmarking
# only and is NEVER used in any calculation in this script.

# ── 2. BLACK-SCHOLES IMPLEMENTATION ───────────────────────────────────────────

#Black-Scholes Formula (Merton's Model)
def bs_d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Computes d1 in the Black-Scholes formula."""
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def bs_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Computes d2 = d1 - sigma * sqrt(T)."""
    return bs_d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, option_type: str) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = bs_d2(S, K, T, r, q, sigma)

    if option_type == "call":
        price = (S * np.exp(-q * T) * norm.cdf(d1)
                 - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        price = (K * np.exp(-r * T) * norm.cdf(-d2)
                 - S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return float(price)

# Computing Vega
def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, q, sigma)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))

# Computing Delta
def bs_delta(S: float, K: float, T: float, r: float, q: float,
             sigma: float, option_type: str) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, q, sigma)
    if option_type == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    else:
        return float(-np.exp(-q * T) * norm.cdf(-d1))

# ── 3. IMPLIED VOLATILITY SOLVER ──────────────────────────────────────────────

def implied_vol_brentq(market_price: float, S: float, K: float, T: float,
                       r: float, q: float, option_type: str) -> float:
    if T <= 0:
        return np.nan

    # Objective: BS_price(sigma) - market_price = 0
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, q, sigma, option_type) - market_price

    try:
        f_low  = objective(IV_MIN)
        f_high = objective(IV_MAX)

        # brentq requires a sign change within the bracket
        if f_low * f_high > 0:
            return np.nan

        iv = brentq(objective, IV_MIN, IV_MAX, xtol=PRICE_TOL, maxiter=500)

        if not (IV_MIN <= iv <= IV_MAX):
            return np.nan

        return float(iv)

    except (ValueError, RuntimeError):
        return np.nan

# ── 4. PUT-CALL PARITY CHECK ──────────────────────────────────────────────────

def pcp_violation(row_call: pd.Series, row_put: pd.Series) -> float:
    """
    Cross-check for matched call/put pairs at the same (K, T).

    Put-call parity (PCP) implies:
        C - P = S * exp(-qT) - K * exp(-rT)

    Returns the absolute PCP violation in price units.
    Large violations indicate data errors or bid-ask spread issues.
    """
    S, K, T = row_call["S0"], row_call["Strike"], row_call["T"]
    r, q    = row_call["Rf"], row_call["q"]
    C, P    = row_call["MidPrice"], row_put["MidPrice"]
    theoretical = S * np.exp(-q * T) - K * np.exp(-r * T)
    return abs((C - P) - theoretical)

# ── 5. MAIN IV COMPUTATION ────────────────────────────────────────────────────

def compute_implied_vols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the implied volatility solver row-by-row and appends:
      - IV        : implied volatility (annualised)
      - Vega      : Black-Scholes vega at the solved IV (used in Step 4)
      - Delta     : Black-Scholes delta at the solved IV
      - IV_valid  : bool flag — True if IV was successfully solved

    Rows where IV cannot be computed are retained but flagged (IV_valid=False).
    The caller decides whether to drop or keep invalid rows.

    Note: IV_data (Refinitiv reference IV) is left untouched in the dataframe.
    """
    df = df.copy()

    n      = len(df)
    ivs    = np.full(n, np.nan)
    vegas  = np.full(n, np.nan)
    deltas = np.full(n, np.nan)

    for i, row in df.iterrows():
        iv = implied_vol_brentq(
            market_price = row["MidPrice"],
            S            = row["S0"],
            K            = row["Strike"],
            T            = row["T"],
            r            = row["Rf"],
            q            = row["q"],
            option_type  = row["OptionType"],
        )
        ivs[i] = iv

        if not np.isnan(iv):
            vegas[i]  = bs_vega( row["S0"], row["Strike"], row["T"], row["Rf"], row["q"], iv)
            deltas[i] = bs_delta(row["S0"], row["Strike"], row["T"], row["Rf"], row["q"], iv,
                                 row["OptionType"])

    df["IV"]       = ivs
    df["Vega"]     = vegas
    df["Delta"]    = deltas
    df["IV_valid"] = ~np.isnan(ivs)

    return df

# ── 6. DIAGNOSTICS ────────────────────────────────────────────────────────────

def print_diagnostics(df_raw: pd.DataFrame, df_valid: pd.DataFrame) -> None:
    """Prints a summary of IV computation success rates."""
    total      = len(df_raw)
    solved     = int(df_raw["IV_valid"].sum())
    failed     = total - solved
    pct_solved = 100 * solved / total if total > 0 else 0

    print(f"\n── IV Extraction Diagnostics ─────────────────────────────")
    print(f"  Total options processed : {total}")
    print(f"  IV solved successfully  : {solved}  ({pct_solved:.1f}%)")
    print(f"  IV failed / discarded   : {failed}")
    print(f"  Options kept (valid IV) : {len(df_valid)}")

    if solved > 0:
        print(f"\n  IV statistics (valid options):")
        print(f"    Min IV  : {df_valid['IV'].min():.4f}  ({df_valid['IV'].min()*100:.2f}%)")
        print(f"    Max IV  : {df_valid['IV'].max():.4f}  ({df_valid['IV'].max()*100:.2f}%)")
        print(f"    Mean IV : {df_valid['IV'].mean():.4f}  ({df_valid['IV'].mean()*100:.2f}%)")
        print(f"    Std IV  : {df_valid['IV'].std():.4f}  ({df_valid['IV'].std()*100:.2f}%)")

        # IV_data comparison — surface-level sanity check only
        if "IV_data" in df_valid.columns:
            ref = df_valid["IV_data"].dropna()
            if not ref.empty:
                diff = (df_valid.loc[ref.index, "IV"] - ref).abs()
                print(f"\n  IV vs IV_data (Refinitiv reference, {len(ref)} matched rows):")
                print(f"    Mean abs diff : {diff.mean():.4f}  ({diff.mean()*100:.2f}%)")
                print(f"    Max abs diff  : {diff.max():.4f}  ({diff.max()*100:.2f}%)")
                print(f"    Note: IV_data is NOT used in any calculation — reference only.")

    print(f"\n  Breakdown by option type:")
    for opt_type in ["call", "put"]:
        subset = df_valid[df_valid["OptionType"] == opt_type]
        if not subset.empty:
            print(f"    {opt_type.capitalize()}s : {len(subset)} options, "
                  f"mean IV = {subset['IV'].mean():.4f}")

    print(f"\n  Breakdown by observation date:")
    for obs_date, group in df_valid.groupby("ObsDate"):
        print(f"    {str(obs_date)[:10]} : {len(group)} options, "
              f"mean IV = {group['IV'].mean():.4f}, "
              f"IV range = [{group['IV'].min():.4f}, {group['IV'].max():.4f}]")


# ── 7. PUBLIC WRAPPERS (importable by downstream scripts) ─────────────────────

def get_bs_price(S: float, K: float, T: float, r: float, q: float,
                 sigma: float, option_type: str) -> float:
    return bs_price(S, K, T, r, q, sigma, option_type)


def get_bs_vega(S: float, K: float, T: float, r: float, q: float,
                sigma: float) -> float:
    return bs_vega(S, K, T, r, q, sigma)


def get_iv(market_price: float, S: float, K: float, T: float,
           r: float, q: float, option_type: str) -> float:
    return implied_vol_brentq(market_price, S, K, T, r, q, option_type)

# ── 8. MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("\n=== Step 2: Implied Volatility Extraction ===\n")

    # Load data from Step 1
    print(f"Loading data from:\n  {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=["ObsDate", "ExDt"])
    print(f"  Loaded {len(df)} options across {df['ObsDate'].nunique()} observation date(s).")

    if "IV_data" in df.columns:
        print(f"  IV_data column present ({df['IV_data'].notna().sum()} non-null) "
              f"— carried through for reference, not used in calculations.")

    # Compute implied volatilities
    print(f"\nSolving for implied volatility (Brent's method)...")
    print(f"  IV bracket : [{IV_MIN:.4f}, {IV_MAX:.2f}]  "
          f"(IV_MIN sourced from data_collection.MIN_IV_FILTER)")
    print(f"  This may take a moment for large datasets...")
    df_with_iv = compute_implied_vols(df)

    # Separate valid and invalid rows
    df_valid = df_with_iv[df_with_iv["IV_valid"]].copy()

    # Print diagnostics
    print_diagnostics(df_with_iv, df_valid)

    # Save output — only keep options with valid IV
    df_valid.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(df_valid)} options with valid IV →\n  {OUTPUT_PATH}")

    print("\n=== Done. Load IV data in subsequent scripts with: ===")
    print("  import pandas as pd")
    print("  from implied_vol import OUTPUT_PATH")
    print('  df = pd.read_csv(OUTPUT_PATH, parse_dates=["ObsDate", "ExDt"])')
    print("\nFunctions available for import by other scripts:")
    print("  from implied_vol import get_bs_price   # Black-Scholes price")
    print("  from implied_vol import get_bs_vega    # BS vega (for loss fn L5)")
    print("  from implied_vol import get_iv         # Solve IV for a single option")


if __name__ == "__main__":
    main()