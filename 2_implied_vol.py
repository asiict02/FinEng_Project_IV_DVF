"""
2_implied_vol.py — Implied Volatility Extraction
Inverts Black-Scholes (Brent's method) to compute IV, Vega and Delta per option.
Outputs: DataSet/data/processed/options_with_iv.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet")
INPUT_PATH  = os.path.join(DATASET_DIR, "data/processed/options_final.csv")
OUTPUT_PATH = os.path.join(DATASET_DIR, "data/processed/options_with_iv.csv")

IV_MIN, IV_MAX, PRICE_TOL = 0.01, 5.0, 1e-6

# ── Black-Scholes (Merton with continuous dividend yield q) ───────────────────
def _d1(S,K,T,r,q,σ): return (np.log(S/K)+(r-q+0.5*σ**2)*T)/(σ*np.sqrt(T))
def _d2(S,K,T,r,q,σ): return _d1(S,K,T,r,q,σ) - σ*np.sqrt(T)

def bs_price(S,K,T,r,q,σ,opt):
    if T<=0 or σ<=0: return 0.0
    d1,d2 = _d1(S,K,T,r,q,σ), _d2(S,K,T,r,q,σ)
    if opt=="call": return float(S*np.exp(-q*T)*norm.cdf(d1)  - K*np.exp(-r*T)*norm.cdf(d2))
    else:           return float(K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1))

def bs_vega(S,K,T,r,q,σ):
    if T<=0 or σ<=0: return 0.0
    return float(S*np.exp(-q*T)*norm.pdf(_d1(S,K,T,r,q,σ))*np.sqrt(T))

def solve_iv(price,S,K,T,r,q,opt):
    """Brent's method: find σ such that BS_price(σ) = market_price."""
    if T<=0: return np.nan
    f = lambda σ: bs_price(S,K,T,r,q,σ,opt) - price
    try:
        if f(IV_MIN)*f(IV_MAX) > 0: return np.nan
        iv = brentq(f, IV_MIN, IV_MAX, xtol=PRICE_TOL, maxiter=500)
        return float(iv) if IV_MIN<=iv<=IV_MAX else np.nan
    except: return np.nan

def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["ObsDate","ExDt"])
    print(f"Loaded {len(df)} options. Solving IV (Brent's method)...")

    ivs, vegas = np.full(len(df),np.nan), np.full(len(df),np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        iv = solve_iv(row["MidPrice"],row["S0"],row["Strike"],row["T"],row["Rf"],row["q"],row["OptionType"])
        ivs[i] = iv
        if not np.isnan(iv):
            vegas[i] = bs_vega(row["S0"],row["Strike"],row["T"],row["Rf"],row["q"],iv)

    df["IV"], df["Vega"] = ivs, vegas
    df_valid = df[~np.isnan(ivs)].copy()
    df_valid.to_csv(OUTPUT_PATH, index=False)

    print(f"✓ IV solved: {len(df_valid)}/{len(df)} ({100*len(df_valid)/len(df):.1f}%)")
    print(f"  IV mean={df_valid['IV'].mean():.4f}  std={df_valid['IV'].std():.4f}  "
          f"min={df_valid['IV'].min():.4f}  max={df_valid['IV'].max():.4f}")

if __name__ == "__main__":
    main()
