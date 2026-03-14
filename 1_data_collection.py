# DATA COLLECTION

"""
Data Collected:
  - SPX option chains (calls & puts) from historical CSV
  - S0, Rf, q, and T are all sourced directly from the data (no external fetching needed)

Outputs (saved relative to CSV_FILE_PATH):
  - data/processed/options_final.csv   <- main file for downstream scripts
  - data/raw/options_raw.csv           <- raw snapshot before cleaning
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 0. SET UP ──────────────────────────────────────────────────────────────────

# !! Each user sets their own path to the raw historical options CSV !!
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "DataSet")
SPX_FILE        = os.path.join(DATASET_DIR, "SPX1mSample.csv")

# Date range for the analysis
# Last 2 months of available data: 2022-04-01 -> 2022-05-31
START_DATE = "2022-04-01"
END_DATE   = "2022-05-31"

MIN_MATURITY_DAYS = 7      # Filter options expiring in fewer than 7 days
MAX_MATURITY_DAYS = 365    # Filter options expiring beyond 1 year
MIN_BID           = 0.05   # Drop options with near-zero bid (illiquid)
MIN_IV_FILTER     = 0.01   # Drop options where IV would be < 1% (used in implied_vol.py)

# ── 1. FILES DIRECTORY ─────────────────────────────────────────────────────────

def make_dirs():
    for folder in ["data/raw", "data/processed"]:
        os.makedirs(os.path.join(DATASET_DIR, folder), exist_ok=True)
    print("✓ Output directories ready.")

# ── 2. FILTERING DATA ──────────────────────────────────────────────────────────

KEEP_COLS = [
    "ObsDate", "ExDt", "T", "S0", "Strike",
    "Bid", "Ask", "MidPrice",
    "Rf", "q", "OptionType",
    "Moneyness",    # K / S0 — for smile plots
    "LogMoneyness", # ln(K / S0)
    "IV_data",      # IV from Refinitiv — kept for reference only, NOT used in calculations
]

def clean_options(df: pd.DataFrame) -> pd.DataFrame:
    """Applies filtering rules and computes derived columns."""
    if df.empty:
        return df

    df = df.copy()

    # Compute mid-price from bid and ask
    df["MidPrice"] = (df["Bid"] + df["Ask"]) / 2

    # ── Filters ────────────────────────────────────────────────────────────────
    # 1. Remove zero or near-zero bid (illiquid options)
    df = df[df["Bid"] >= MIN_BID]

    # 2. Remove options with zero mid-price
    df = df[df["MidPrice"] > 0]

    # 3. Remove options where bid > ask (data error)
    df = df[df["Bid"] <= df["Ask"]]

    # 4. Apply maturity filter (T is in years)
    df = df[(df["T"] >= MIN_MATURITY_DAYS / 365) & (df["T"] <= MAX_MATURITY_DAYS / 365)]

    # 5. Remove extreme moneyness (deep ITM / OTM are unreliable)
    df["Moneyness"]    = df["Strike"] / df["S0"]
    df["LogMoneyness"] = np.log(df["Strike"] / df["S0"])
    df = df[(df["Moneyness"] >= 0.7) & (df["Moneyness"] <= 1.3)]

    # 6. Drop rows with missing key fields
    df.dropna(subset=["Strike", "Bid", "Ask", "T", "S0", "Rf"], inplace=True)

    # ── Keep only the columns we need ─────────────────────────────────────────
    available_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[available_cols]

    # ── Sort ───────────────────────────────────────────────────────────────────
    df.sort_values(["ObsDate", "OptionType", "ExDt", "Strike"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# ── 3. LOAD HISTORICAL OPTIONS FROM CSV ───────────────────────────────────────

def load_from_csv() -> pd.DataFrame:
    files = glob.glob(os.path.join(DATASET_DIR, "SP_*.csv"))
    df = pd.concat([pd.read_csv(f, parse_dates=["t", "T"]) for f in files], ignore_index=True)

    # ── Rename columns to internal naming convention ───────────────────────────
    df.rename(columns={
        "t":         "ObsDate",
        "T":         "ExDt",
        "tau":       "T",        # time to maturity in years — used directly
        "K":         "Strike",
        "bid":       "Bid",
        "ask":       "Ask",
        "under_mid": "S0",
        "r":         "Rf",       # already continuously compounded — no conversion needed
        "divyield":  "q",
        "IV":        "IV_data",  # kept for reference only
    }, inplace=True)

    # ── Map opt_type: p -> put, c -> call ─────────────────────────────────────
    df["OptionType"] = df["opt_type"].map({"p": "put", "c": "call"})

    # ── Filter to analysis window ──────────────────────────────────────────────
    df["ObsDate"] = pd.to_datetime(df["ObsDate"])
    df = df[
        (df["ObsDate"] >= pd.to_datetime(START_DATE)) &
        (df["ObsDate"] <= pd.to_datetime(END_DATE))
    ]

    return clean_options(df)

# ── 4. SAVING AND STORING THE DATA ────────────────────────────────────────────

def main():
    make_dirs()
    print("\n=== Data Collection: Implied Volatility & DVF Project ===\n")
    print(f"  Analysis window: {START_DATE} -> {END_DATE}\n")

    # Load, rename, filter, and clean historical options data
    print("Step 1/1 — Loading and processing historical options data...")
    df_final = load_from_csv()

    if df_final.empty:
        print("\n⚠ No data collected. Check your CSV path and date range.")
        return

    # Save raw snapshot and final cleaned file
    path_raw = os.path.join(DATASET_DIR, "data/raw/options_raw.csv")
    path_final = os.path.join(DATASET_DIR, "data/processed/options_final.csv")

    df_final.to_csv(path_raw, index=False)
    print(f"✓ Raw snapshot saved -> {path_raw}  ({len(df_final)} rows)")

    df_final.to_csv(path_final, index=False)
    print(f"✓ Final data saved   -> {path_final}  ({len(df_final)} rows)")

    # Summary
    print("\n── Summary ───────────────────────────────────────────────")
    print(f"  Observation dates : {df_final['ObsDate'].nunique()}")
    print(f"  Unique expirations: {df_final['ExDt'].nunique()}")
    print(f"  Total options     : {len(df_final)}")
    print(f"  Calls / Puts      : {(df_final['OptionType']=='call').sum()} / {(df_final['OptionType']=='put').sum()}")
    print(f"  Maturity range    : {df_final['T'].min():.3f}y — {df_final['T'].max():.3f}y")
    print(f"  Strike range      : {df_final['Strike'].min():.0f} — {df_final['Strike'].max():.0f}")
    print(f"  Rf range          : {df_final['Rf'].min():.4f} — {df_final['Rf'].max():.4f}")
    print(f"  S0 range          : {df_final['S0'].min():.2f} — {df_final['S0'].max():.2f}")

    print("\n=== Done. Load data in other scripts with: ===")
    print("  import pandas as pd")
    print(f'  df = pd.read_csv("{path_final}", parse_dates=["ObsDate", "ExDt"])')


if __name__ == "__main__":
    main()