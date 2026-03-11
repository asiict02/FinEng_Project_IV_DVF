"""
implied_vol.py
==============
Step 2: Implied Volatility Extraction

Reads the cleaned option data produced by data_collection.py and computes
the implied volatility (IV) for each option by numerically inverting the
Black-Scholes formula.

Inputs:
    data/processed/options_final.csv   (produced by data_collection.py)

Outputs:
    data/processed/options_with_iv.csv ← used by ALL subsequent scripts

Usage:
    python implied_vol.py

All other scripts should load IV data with:
    import pandas as pd
    df = pd.read_csv("data/processed/options_with_iv.csv",
                     parse_dates=["ObsDate", "ExDt"])
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH  = "data/processed/options_final.csv"
OUTPUT_PATH = "data/processed/options_with_iv.csv"

IV_MIN   = 0.01   # 1%  — discard solutions below this (nonsensical)
IV_MAX   = 5.00   # 500% — discard solutions above this (numerical artefact)
IV_INIT  = 0.25   # 25% — initial guess passed to solver (not used by brentq)
PRICE_TOL = 1e-6  # tolerance for BS price vs market price match


# ── Black-Scholes Implementation ───────────────────────────────────────────────

def bs_d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Computes d1 in the Black-Scholes formula."""
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Computes d2 = d1 - sigma * sqrt(T)."""
    return bs_d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, option_type: str) -> float:
    """
    Black-Scholes option price for European call or put on a
    dividend-paying underlying (continuous dividend yield q).

    Parameters
    ----------
    S           : Spot price
    K           : Strike price
    T           : Time to maturity in years
    r           : Continuously compounded risk-free rate
    q           : Continuous dividend yield
    sigma       : Volatility (annualised)
    option_type : 'call' or 'put'

    Returns
    -------
    float : Option price
    """
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


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black-Scholes vega: sensitivity of option price to volatility.
    Same formula for calls and puts.
    Used for Newton-Raphson and for vega-weighted loss functions in Step 4.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, q, sigma)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def bs_delta(S: float, K: float, T: float, r: float, q: float,
             sigma: float, option_type: str) -> float:
    """
    Black-Scholes delta: dPrice/dS.
    Useful for diagnostics and hedging discussion in the report.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, q, sigma)
    if option_type == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    else:
        return float(-np.exp(-q * T) * norm.cdf(-d1))


# ── Implied Volatility Solver ──────────────────────────────────────────────────

def implied_vol_brentq(market_price: float, S: float, K: float, T: float,
                        r: float, q: float, option_type: str) -> float:
    """
    Computes implied volatility by inverting Black-Scholes using Brent's method.

    Brent's method (scipy.optimize.brentq) is preferred over Newton-Raphson
    because it is guaranteed to converge if a sign change exists in [IV_MIN, IV_MAX].
    It does not require derivatives, is numerically stable, and always brackets
    the solution.

    Returns np.nan if:
      - T <= 0 (expired option)
      - No sign change found in [IV_MIN, IV_MAX] (price outside BS range)
      - The solved IV is outside [IV_MIN, IV_MAX]
    """
    if T <= 0:
        return np.nan

    # Objective: BS_price(sigma) - market_price = 0
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, q, sigma, option_type) - market_price

    try:
        f_low  = objective(IV_MIN)
        f_high = objective(IV_MAX)

        # Brentq requires a sign change in the interval
        if f_low * f_high > 0:
            return np.nan

        iv = brentq(objective, IV_MIN, IV_MAX, xtol=PRICE_TOL, maxiter=500)

        if not (IV_MIN <= iv <= IV_MAX):
            return np.nan

        return float(iv)

    except (ValueError, RuntimeError):
        return np.nan


# ── Put-Call Parity Check ──────────────────────────────────────────────────────

def pcp_implied_vol(row_call: pd.Series, row_put: pd.Series) -> float:
    """
    Cross-check: for matched call/put pairs at the same (K, T),
    put-call parity (PCP) implies:
        C - P = S*exp(-qT) - K*exp(-rT)

    If the pair violates PCP by more than a threshold, flag both options.
    Returns the PCP violation in price units.
    """
    S, K, T, r, q = row_call["S0"], row_call["Strike"], row_call["T"], row_call["Rf"], row_call["q"]
    C = row_call["MidPrice"]
    P = row_put["MidPrice"]
    theoretical = S * np.exp(-q * T) - K * np.exp(-r * T)
    return abs((C - P) - theoretical)


# ── Main IV Computation ────────────────────────────────────────────────────────

def compute_implied_vols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the implied volatility solver row-by-row and appends:
      - IV        : implied volatility (annualised)
      - Vega      : Black-Scholes vega at the solved IV (used in Step 4)
      - Delta     : Black-Scholes delta at the solved IV
      - IV_valid  : bool flag — True if IV was successfully solved

    Rows where IV cannot be computed are retained but flagged (IV_valid=False).
    The caller decides whether to drop or keep invalid rows.
    """
    df = df.copy()

    ivs    = np.full(len(df), np.nan)
    vegas  = np.full(len(df), np.nan)
    deltas = np.full(len(df), np.nan)

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
            deltas[i] = bs_delta(row["S0"], row["Strike"], row["T"], row["Rf"], row["q"], iv, row["OptionType"])

    df["IV"]       = ivs
    df["Vega"]     = vegas
    df["Delta"]    = deltas
    df["IV_valid"] = ~np.isnan(ivs)

    return df


# ── Diagnostics ────────────────────────────────────────────────────────────────

def print_diagnostics(df_raw: pd.DataFrame, df_valid: pd.DataFrame) -> None:
    """Prints a summary of IV computation success rates."""
    total     = len(df_raw)
    solved    = df_raw["IV_valid"].sum()
    failed    = total - solved
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

    print(f"\n  Breakdown by option type:")
    for opt_type in ["call", "put"]:
        subset = df_valid[df_valid["OptionType"] == opt_type]
        print(f"    {opt_type.capitalize()}s : {len(subset)} options, "
              f"mean IV = {subset['IV'].mean():.4f}")

    print(f"\n  Breakdown by observation date:")
    for obs_date, group in df_valid.groupby("ObsDate"):
        print(f"    {str(obs_date)[:10]} : {len(group)} options, "
              f"mean IV = {group['IV'].mean():.4f}, "
              f"IV range = [{group['IV'].min():.4f}, {group['IV'].max():.4f}]")


# ── Standalone helper functions (importable by other scripts) ──────────────────

def get_bs_price(S: float, K: float, T: float, r: float, q: float,
                 sigma: float, option_type: str) -> float:
    """
    Public wrapper around bs_price().
    Import this in dvf_models.py and estimation.py to get model prices.

    Example:
        from implied_vol import get_bs_price
        price = get_bs_price(S=5000, K=5000, T=0.25, r=0.05,
                              q=0.013, sigma=0.20, option_type='call')
    """
    return bs_price(S, K, T, r, q, sigma, option_type)


def get_bs_vega(S: float, K: float, T: float, r: float, q: float,
                sigma: float) -> float:
    """
    Public wrapper around bs_vega().
    Import this in loss_functions.py for vega-weighted loss (L5).

    Example:
        from implied_vol import get_bs_vega
        vega = get_bs_vega(S=5000, K=5000, T=0.25, r=0.05, q=0.013, sigma=0.20)
    """
    return bs_vega(S, K, T, r, q, sigma)


def get_iv(market_price: float, S: float, K: float, T: float,
           r: float, q: float, option_type: str) -> float:
    """
    Public wrapper around implied_vol_brentq().
    Useful if other scripts need to compute a single IV on the fly.

    Example:
        from implied_vol import get_iv
        iv = get_iv(market_price=50.0, S=5000, K=5000, T=0.25,
                    r=0.05, q=0.013, option_type='call')
    """
    return implied_vol_brentq(market_price, S, K, T, r, q, option_type)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Step 2: Implied Volatility Extraction ===\n")

    # Load data from Step 1
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["ObsDate", "ExDt"])
    print(f"  Loaded {len(df)} options across {df['ObsDate'].nunique()} observation date(s).")

    # Compute implied volatilities
    print(f"\nSolving for implied volatility (Brent's method)...")
    print(f"  This may take a moment for large datasets...")
    df_with_iv = compute_implied_vols(df)

    # Separate valid and invalid
    df_valid = df_with_iv[df_with_iv["IV_valid"]].copy()

    # Print diagnostics
    print_diagnostics(df_with_iv, df_valid)

    # Save output — only keep options with valid IV
    df_valid.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(df_valid)} options with valid IV → {OUTPUT_PATH}")

    print("\n=== Done. Load IV data in subsequent scripts with: ===")
    print("  import pandas as pd")
    print(f'  df = pd.read_csv("{OUTPUT_PATH}", parse_dates=["ObsDate", "ExDt"])')
    print("\nFunctions available for import by other scripts:")
    print("  from implied_vol import get_bs_price   # Black-Scholes price")
    print("  from implied_vol import get_bs_vega    # BS vega (for loss fn L5)")
    print("  from implied_vol import get_iv         # Solve IV for a single option")


if __name__ == "__main__":
    main()
