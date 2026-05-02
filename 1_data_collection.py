"""
1_data_collection.py — Data Collection
Loads SPX option chain (calls & puts) from SP_*.csv files, files, applies
standard market filters, removes put-call parity violations, and saves
the cleaned dataset for downstream IV extraction and model fitting.

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
MIN_MATURITY_DAYS = 7      # drop options close to maturity
MAX_MATURITY_DAYS = 365    # drop options with over 1 year maturity
MIN_BID           = 0.05   # remove stale or crossed quotes
IV_MIN_FILTER     = 0.05   # drop options with pre-computed IV below this
IV_MAX_FILTER     = 0.80   # drop options with pre-computed IV above this (outliers)
PCP_TOL           = 2.0    # maximum allowed put-call parity violation in dollars


def _apply_pcp_filter(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    """
    Removes matched call-put pairs that violate put-call parity beyond PCP_TOL.
    Both legs of a violating pair are dropped since it is impossible to identify which side
    is mis-quoted. Returns the filtered DataFrame with index reset.
    """
    key = ["ObsDate", "ExDt", "T", "Strike", "S0", "Rf", "q"]
    calls = df[df["OptionType"] == "call"][key + ["MidPrice"]].copy()
    puts  = df[df["OptionType"] == "put"][key  + ["MidPrice"]].copy()

    merged = calls.merge(puts, on=key, suffixes=("_c", "_p"))
    if merged.empty:
        return df

    merged["pcp_rhs"] = (merged["S0"] * np.exp(-merged["q"] * merged["T"])
                         - merged["Strike"] * np.exp(-merged["Rf"] * merged["T"]))
    merged["pcp_err"] = (merged["MidPrice_c"] - merged["MidPrice_p"] - merged["pcp_rhs"]).abs()

    bad = merged[merged["pcp_err"] > tol][["ObsDate", "T", "Strike", "S0"]]
    n_bad = len(bad)

    if n_bad > 0:
        bad = bad.drop_duplicates()
        df = df.merge(bad.assign(_pcp_flag=True),
                      on=["ObsDate", "T", "Strike", "S0"], how="left")
        df = df[df["_pcp_flag"].isna()].drop(columns="_pcp_flag")
        print(f"  PCP filter: removed {n_bad} violating matched pairs (tol=${tol:.2f})")

    return df.reset_index(drop=True)


def main():
    """
    Loads all SP_*.csv files, standardises column names, restricts to the analysis window,
    derives MidPrice and moneyness, applies standard market filters (bid, maturity, moneyness,
    missing values), removes IV outliers, runs the PCP filter, and saves the cleaned dataset.
    """
    os.makedirs(os.path.join(DATASET_DIR, "data/raw"),       exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "data/processed"), exist_ok=True)

    # ── Load all SP_*.csv files ────────────────────────────────────────────────
    files = glob.glob(os.path.join(DATASET_DIR, "SP_*.csv"))
    if not files:
        raise FileNotFoundError(f"No SP_*.csv files found in {DATASET_DIR}")
    df = pd.concat([pd.read_csv(f, parse_dates=["t", "T"]) for f in files],
                   ignore_index=True)

    # ── Rename to internal convention ─────────────────────────────────────────
    df.rename(columns={"t": "ObsDate", "T": "ExDt", "tau": "T", "K": "Strike",
                        "bid": "Bid", "ask": "Ask", "under_mid": "S0",
                        "r": "Rf", "divyield": "q", "IV": "IV_data"}, inplace=True)
    df["OptionType"] = df["opt_type"].map({"p": "put", "c": "call"})
    df["ObsDate"]    = pd.to_datetime(df["ObsDate"])

    # ── Filter to analysis window ──────────────────────────────────────────────
    df = df[(df["ObsDate"] >= START_DATE) & (df["ObsDate"] <= END_DATE)]
    n0 = len(df)

    # ── Basic derived columns ──────────────────────────────────────────────────
    df["MidPrice"]     = (df["Bid"] + df["Ask"]) / 2
    df["Moneyness"]    = df["Strike"] / df["S0"]
    df["LogMoneyness"] = np.log(df["Moneyness"])

    # ── Warn on negative risk-free rates (keep but flag) ──────────────────────
    n_neg_rf = (df["Rf"] < 0).sum()
    if n_neg_rf > 0:
        print(f"  Warning: {n_neg_rf} rows have negative Rf "
              f"(min={df['Rf'].min():.6f}). Retained as-is.")

    # ── Standard quote / maturity / moneyness filters ─────────────────────────
    df = df[(df["Bid"] >= MIN_BID) & (df["MidPrice"] > 0) & (df["Bid"] <= df["Ask"])]
    df = df[(df["T"] >= MIN_MATURITY_DAYS / 365) & (df["T"] <= MAX_MATURITY_DAYS / 365)]
    df = df[(df["Moneyness"] >= 0.7) & (df["Moneyness"] <= 1.3)]
    df.dropna(subset=["Strike", "Bid", "Ask", "T", "S0", "Rf"], inplace=True)
    print(f"  After standard filters: {len(df)} / {n0} rows")

    # ── IV outlier filter using pre-computed IV_data ───────────────────
    if "IV_data" in df.columns:
        n_before = len(df)
        df = df[(df["IV_data"] >= IV_MIN_FILTER) & (df["IV_data"] <= IV_MAX_FILTER)]
        print(f"  IV outlier filter [{IV_MIN_FILTER}, {IV_MAX_FILTER}]: "
              f"removed {n_before - len(df)} rows")

    df = _apply_pcp_filter(df, tol=PCP_TOL)

    # ── Final column selection and sort ───────────────────────────────────────
    keep = ["ObsDate", "ExDt", "T", "S0", "Strike", "Bid", "Ask", "MidPrice",
            "Rf", "q", "OptionType", "Moneyness", "LogMoneyness", "IV_data"]
    df = df[[c for c in keep if c in df.columns]]
    df.sort_values(["ObsDate", "OptionType", "ExDt", "Strike"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    df.to_csv(os.path.join(DATASET_DIR, "data/raw/options_raw.csv"),       index=False)
    df.to_csv(os.path.join(DATASET_DIR, "data/processed/options_final.csv"), index=False)

    print(f"\n✓ {len(df)} options saved  |  dates: {df['ObsDate'].nunique()}  "
          f"|  calls: {(df['OptionType'] == 'call').sum()}  "
          f"puts: {(df['OptionType'] == 'put').sum()}")
    print(f"  Maturity: {df['T'].min():.3f}y–{df['T'].max():.3f}y  "
          f"|  Strike: {df['Strike'].min():.0f}–{df['Strike'].max():.0f}  "
          f"|  S0: {df['S0'].min():.2f}–{df['S0'].max():.2f}")
    if "IV_data" in df.columns:
        print(f"  IV_data: {df['IV_data'].min():.4f}–{df['IV_data'].max():.4f}  "
              f"mean={df['IV_data'].mean():.4f}")

if __name__ == "__main__":
    main()
