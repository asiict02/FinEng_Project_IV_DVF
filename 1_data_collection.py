"""
1_data_collection.py — Data Collection
Loads SPX option chain (calls & puts) from SP_*.csv files, filters and cleans.
Outputs: DataSet/data/raw/options_raw.csv
         DataSet/data/processed/options_final.csv
"""

import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet")

START_DATE        = "2022-04-01"
END_DATE          = "2022-05-31"
MIN_MATURITY_DAYS = 7
MAX_MATURITY_DAYS = 365
MIN_BID           = 0.05

def main():
    os.makedirs(os.path.join(DATASET_DIR, "data/raw"),       exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "data/processed"), exist_ok=True)

    # ── Load all SP_*.csv files ────────────────────────────────────────────────
    files = glob.glob(os.path.join(DATASET_DIR, "SP_*.csv"))
    df = pd.concat([pd.read_csv(f, parse_dates=["t","T"]) for f in files], ignore_index=True)

    # ── Rename to internal convention ─────────────────────────────────────────
    df.rename(columns={"t":"ObsDate","T":"ExDt","tau":"T","K":"Strike",
                        "bid":"Bid","ask":"Ask","under_mid":"S0",
                        "r":"Rf","divyield":"q","IV":"IV_data"}, inplace=True)
    df["OptionType"] = df["opt_type"].map({"p":"put","c":"call"})
    df["ObsDate"]    = pd.to_datetime(df["ObsDate"])

    # ── Filter to analysis window ──────────────────────────────────────────────
    df = df[(df["ObsDate"] >= START_DATE) & (df["ObsDate"] <= END_DATE)]

    # ── Clean ──────────────────────────────────────────────────────────────────
    df["MidPrice"]    = (df["Bid"] + df["Ask"]) / 2
    df["Moneyness"]   = df["Strike"] / df["S0"]
    df["LogMoneyness"]= np.log(df["Moneyness"])
    df = df[(df["Bid"] >= MIN_BID) & (df["MidPrice"] > 0) & (df["Bid"] <= df["Ask"])]
    df = df[(df["T"] >= MIN_MATURITY_DAYS/365) & (df["T"] <= MAX_MATURITY_DAYS/365)]
    df = df[(df["Moneyness"] >= 0.7) & (df["Moneyness"] <= 1.3)]
    df.dropna(subset=["Strike","Bid","Ask","T","S0","Rf"], inplace=True)

    keep = ["ObsDate","ExDt","T","S0","Strike","Bid","Ask","MidPrice",
            "Rf","q","OptionType","Moneyness","LogMoneyness","IV_data"]
    df = df[[c for c in keep if c in df.columns]]
    df.sort_values(["ObsDate","OptionType","ExDt","Strike"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    df.to_csv(os.path.join(DATASET_DIR, "data/raw/options_raw.csv"), index=False)
    df.to_csv(os.path.join(DATASET_DIR, "data/processed/options_final.csv"), index=False)

    print(f"✓ {len(df)} options saved  |  dates: {df['ObsDate'].nunique()}  "
          f"|  calls: {(df['OptionType']=='call').sum()}  puts: {(df['OptionType']=='put').sum()}")
    print(f"  Maturity: {df['T'].min():.3f}y–{df['T'].max():.3f}y  "
          f"|  Strike: {df['Strike'].min():.0f}–{df['Strike'].max():.0f}  "
          f"|  S0: {df['S0'].min():.2f}–{df['S0'].max():.2f}")

if __name__ == "__main__":
    main()
