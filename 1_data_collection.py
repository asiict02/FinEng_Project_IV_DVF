"""
data_collection.py
==================
Collects all data required for the Implied Volatility & DVF project:
  - SPX option chains (calls & puts) via yfinance
  - SPX spot price at each observation date
  - Risk-free rates from the U.S. Treasury yield curve
  - Constant dividend yield for SPX

Outputs (saved to data/ folder):
  - data/processed/options_final.csv   ← main file used by all other scripts
  - data/raw/options_raw.csv           ← raw chains before filtering
  - data/risk_free/treasury_rates.csv  ← raw treasury curve

Usage:
  python data_collection.py

All other scripts should load data with:
  import pandas as pd
  df = pd.read_csv("data/processed/options_final.csv", parse_dates=["ObsDate", "ExDt"])
"""

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from io import StringIO

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

# Observation dates: spread across different market regimes.
# Edit this list to add/remove dates. Format: "YYYY-MM-DD"
# Suggestions:
#   2020-03-16 → extreme vol (COVID crash)
#   2022-06-15 → rising rate / bear market
#   2023-03-15 → banking stress (SVB)
#   2024-01-17 → normal/low vol regime
#   2024-10-16 → pre-election uncertainty
OBSERVATION_DATES = [
    "2024-01-17",
    "2024-04-17",
    "2024-07-17",
    "2024-10-16",
]

DIVIDEND_YIELD = 0.013          # Constant ~1.3% annualised for SPX
MIN_MATURITY_DAYS = 7           # Drop options expiring in fewer than 7 days
MAX_MATURITY_DAYS = 365         # Drop options expiring beyond 1 year
MIN_IV_FILTER = 0.01            # Drop options where IV would be < 1%  (applied later in implied_vol.py)
MIN_BID = 0.05                  # Drop options with near-zero bid (illiquid)
OUTPUT_DIR = "data"


# ── Directory setup ────────────────────────────────────────────────────────────

def make_dirs():
    for folder in ["data/raw", "data/processed", "data/risk_free"]:
        os.makedirs(folder, exist_ok=True)
    print("✓ Output directories ready.")


# ── 1. Treasury Yield Curve ────────────────────────────────────────────────────

# Maturity labels as they appear in the Treasury CSV, mapped to years
TREASURY_MATURITIES = {
    "1 Mo":  1/12,
    "2 Mo":  2/12,
    "3 Mo":  3/12,
    "6 Mo":  6/12,
    "1 Yr":  1.0,
    "2 Yr":  2.0,
    "3 Yr":  3.0,
    "5 Yr":  5.0,
    "7 Yr":  7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

def fetch_treasury_rates(obs_dates: list[str]) -> pd.DataFrame:
    """
    Downloads the U.S. Treasury par yield curve from treasury.gov for the
    years covering all observation dates, then returns daily rows filtered
    to those dates (or nearest available business day).

    Returns a DataFrame with columns: Date, 1 Mo, 2 Mo, ..., 30 Yr
    Yields are in percentage points (e.g. 5.25 means 5.25%).
    """
    years_needed = sorted(set(pd.to_datetime(obs_dates).year))
    all_rows = []

    for year in years_needed:
        url = (
            f"https://home.treasury.gov/resource-center/data-chart-center/"
            f"interest-rates/daily-treasury-rates.csv/{year}/all?"
            f"type=daily_treasury_yield_curve&field_tdr_date_value={year}&download=true"
        )
        print(f"  Fetching Treasury rates for {year}...")
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            df_year = pd.read_csv(StringIO(resp.text))
            all_rows.append(df_year)
        except Exception as e:
            print(f"  ⚠ Could not fetch treasury data for {year}: {e}")
            print(f"    Falling back to yfinance proxies (^IRX, ^FVX, ^TNX).")
            return fetch_treasury_rates_yfinance_fallback(obs_dates)

    df_treasury = pd.concat(all_rows, ignore_index=True)

    # Normalise the date column (treasury.gov labels it differently over years)
    date_col = [c for c in df_treasury.columns if "date" in c.lower()]
    if not date_col:
        raise ValueError("Cannot find date column in Treasury CSV.")
    df_treasury.rename(columns={date_col[0]: "Date"}, inplace=True)
    df_treasury["Date"] = pd.to_datetime(df_treasury["Date"])
    df_treasury.sort_values("Date", inplace=True)

    df_treasury.to_csv("data/risk_free/treasury_rates.csv", index=False)
    print(f"  ✓ Treasury rates saved → data/risk_free/treasury_rates.csv")
    return df_treasury


def fetch_treasury_rates_yfinance_fallback(obs_dates: list[str]) -> pd.DataFrame:
    """
    Fallback: approximate yield curve from yfinance Treasury tickers.
    Returns a DataFrame shaped to match the main treasury function output.
    """
    proxies = {
        "3 Mo": "^IRX",   # 13-week T-bill
        "5 Yr":  "^FVX",  # 5-year
        "10 Yr": "^TNX",  # 10-year
        "30 Yr": "^TYX",  # 30-year
    }
    start = min(obs_dates)
    end   = max(obs_dates)
    rows = []
    for label, ticker in proxies.items():
        data = yf.download(ticker, start=start, end=end, progress=False)
        rows.append(data["Close"].rename(label))
    df = pd.concat(rows, axis=1).reset_index().rename(columns={"index": "Date"})
    df.to_csv("data/risk_free/treasury_rates.csv", index=False)
    print("  ✓ Fallback Treasury rates saved → data/risk_free/treasury_rates.csv")
    return df


def interpolate_risk_free(T_years: float, obs_date: pd.Timestamp,
                           df_treasury: pd.DataFrame) -> float:
    """
    Given a time-to-maturity T (in years) and an observation date,
    find the nearest date row in the Treasury curve and linearly
    interpolate to get the continuously compounded risk-free rate.

    Treasury yields are par yields (%) → convert to continuous rate:
        r_continuous = ln(1 + r_par/100)
    """
    # Find nearest available date on or before obs_date
    available = df_treasury[df_treasury["Date"] <= obs_date]
    if available.empty:
        available = df_treasury  # use earliest available
    row = available.iloc[-1]

    # Build (maturity_years, yield_pct) pairs from available columns
    mat_yield_pairs = []
    for col_label, mat_yr in TREASURY_MATURITIES.items():
        if col_label in row.index and pd.notna(row[col_label]):
            mat_yield_pairs.append((mat_yr, float(row[col_label])))

    if not mat_yield_pairs:
        return 0.05  # last-resort default

    mat_yield_pairs.sort(key=lambda x: x[0])
    mats   = np.array([x[0] for x in mat_yield_pairs])
    yields = np.array([x[1] for x in mat_yield_pairs])

    # Interpolate (clamp to edges if T is outside range)
    par_yield_pct = float(np.interp(T_years, mats, yields))

    # Convert par yield (%) → continuously compounded rate
    r_continuous = np.log(1 + par_yield_pct / 100)
    return r_continuous


# ── 2. SPX Spot Price ──────────────────────────────────────────────────────────

def fetch_spot_price(obs_date: str) -> float:
    """
    Returns the SPX closing price on or just before obs_date.
    """
    target = pd.to_datetime(obs_date)
    # Download a small window around the target date
    start = (target - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end   = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data  = yf.download("^SPX", start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No SPX price data available around {obs_date}")
    # Use the last available close on or before the target
    available = data[data.index <= target]
    if available.empty:
        available = data
    price = float(available["Close"].iloc[-1])
    return price


# ── 3. Option Chain ────────────────────────────────────────────────────────────

def fetch_option_chain(obs_date: str, S0: float,
                       df_treasury: pd.DataFrame) -> pd.DataFrame:
    """
    Downloads all available SPX option expiries from yfinance and builds
    a clean DataFrame of calls and puts for the given observation date.

    Note: yfinance returns the *current* live chain. If obs_date is in the
    past, the expirations available will differ from what was available then.
    For a live/current dataset this is fine. For historical data, use CBOE
    or OptionMetrics instead and skip to the 'Load from CSV' section below.
    """
    ticker = yf.Ticker("^SPX")
    expirations = ticker.options  # tuple of expiry date strings

    target_dt = pd.to_datetime(obs_date)
    records = []

    for exp_str in expirations:
        exp_dt  = pd.to_datetime(exp_str)
        T_days  = (exp_dt - target_dt).days
        T_years = T_days / 365.0

        # Apply maturity filters
        if T_days < MIN_MATURITY_DAYS or T_days > MAX_MATURITY_DAYS:
            continue

        try:
            chain = ticker.option_chain(exp_str)
        except Exception as e:
            print(f"    ⚠ Could not fetch chain for {exp_str}: {e}")
            continue

        # Get interpolated risk-free rate for this maturity
        Rf = interpolate_risk_free(T_years, target_dt, df_treasury)

        for opt_type, df_opts in [("call", chain.calls), ("put", chain.puts)]:
            if df_opts.empty:
                continue

            df_opts = df_opts.copy()
            df_opts["OptionType"] = opt_type
            df_opts["ObsDate"]    = obs_date
            df_opts["ExDt"]       = exp_str
            df_opts["T"]          = round(T_years, 6)
            df_opts["S0"]         = S0
            df_opts["Rf"]         = round(Rf, 6)
            df_opts["q"]          = DIVIDEND_YIELD

            # Standardise column names from yfinance
            df_opts.rename(columns={
                "strike":       "Strike",
                "bid":          "Bid",
                "ask":          "Ask",
                "lastPrice":    "LastPrice",
                "volume":       "Volume",
                "openInterest": "OpenInterest",
                "impliedVolatility": "IV_yf",  # yfinance's own IV estimate (for reference)
            }, inplace=True)

            records.append(df_opts)

        time.sleep(0.1)  # be polite to the API

    if not records:
        print(f"  ⚠ No option data returned for {obs_date}.")
        return pd.DataFrame()

    df_all = pd.concat(records, ignore_index=True)
    return df_all


# ── 4. Cleaning & Filtering ────────────────────────────────────────────────────

KEEP_COLS = [
    "ObsDate", "ExDt", "T", "S0", "Strike",
    "Bid", "Ask", "MidPrice", "LastPrice",
    "Volume", "OpenInterest", "IV_yf",
    "Rf", "q", "OptionType",
    "Moneyness",   # K / S0 — useful for smile plots
    "LogMoneyness" # ln(K / S0) — standard in academic papers
]

def clean_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies filtering rules and computes derived columns.
    """
    if df.empty:
        return df

    df = df.copy()

    # Compute mid-price
    df["MidPrice"] = (df["Bid"] + df["Ask"]) / 2

    # ── Filters ────────────────────────────────────────────────────────────────
    # 1. Remove zero or near-zero bid (illiquid options)
    df = df[df["Bid"] >= MIN_BID]

    # 2. Remove options with zero mid-price
    df = df[df["MidPrice"] > 0]

    # 3. Remove options where bid > ask (data error)
    df = df[df["Bid"] <= df["Ask"]]

    # 4. Remove extreme moneyness (deep ITM / OTM are unreliable)
    df["Moneyness"]    = df["Strike"] / df["S0"]
    df["LogMoneyness"] = np.log(df["Strike"] / df["S0"])
    df = df[(df["Moneyness"] >= 0.7) & (df["Moneyness"] <= 1.3)]

    # 5. Drop rows with missing key fields
    df.dropna(subset=["Strike", "Bid", "Ask", "T", "S0", "Rf"], inplace=True)

    # ── Keep only the columns we need ─────────────────────────────────────────
    available_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[available_cols]

    # ── Sort ───────────────────────────────────────────────────────────────────
    df.sort_values(["ObsDate", "OptionType", "ExDt", "Strike"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ── 5. Load from CSV (for historical / CBOE data) ─────────────────────────────

def load_from_csv(filepath: str, df_treasury: pd.DataFrame) -> pd.DataFrame:
    """
    If you have historical data from CBOE or OptionMetrics as a CSV,
    use this function instead of fetch_option_chain().

    Expected columns in your CSV:
        ObsDate, ExDt, Strike, Bid, Ask, OptionType
        (S0, Rf, q will be added automatically)

    Usage:
        df = load_from_csv("data/raw/my_cboe_data.csv", df_treasury)
    """
    df = pd.read_csv(filepath, parse_dates=["ObsDate", "ExDt"])

    # Compute T in years
    df["T"] = (df["ExDt"] - df["ObsDate"]).dt.days / 365.0

    # Fetch spot price for each unique obs date
    spot_map = {}
    for d in df["ObsDate"].dt.strftime("%Y-%m-%d").unique():
        spot_map[d] = fetch_spot_price(d)
    df["S0"] = df["ObsDate"].dt.strftime("%Y-%m-%d").map(spot_map)

    # Interpolate risk-free rate for each row
    df["Rf"] = df.apply(
        lambda row: interpolate_risk_free(
            row["T"],
            row["ObsDate"],
            df_treasury
        ), axis=1
    )

    df["q"] = DIVIDEND_YIELD

    return clean_options(df)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    make_dirs()
    print("\n=== Data Collection: Implied Volatility & DVF Project ===\n")

    # Step 1: Treasury yield curve
    print("Step 1/3 — Fetching Treasury yield curve...")
    df_treasury = fetch_treasury_rates(OBSERVATION_DATES)

    all_raw    = []
    all_clean  = []

    # Steps 2 & 3: Spot price + option chains for each observation date
    for obs_date in OBSERVATION_DATES:
        print(f"\nStep 2-3 — Processing observation date: {obs_date}")

        # Spot price
        print(f"  Fetching SPX spot price...")
        S0 = fetch_spot_price(obs_date)
        print(f"  SPX spot on {obs_date}: {S0:.2f}")

        # Option chain
        print(f"  Fetching option chain (this may take ~30 seconds)...")
        df_raw = fetch_option_chain(obs_date, S0, df_treasury)

        if df_raw.empty:
            print(f"  ⚠ Skipping {obs_date} — no data.")
            continue

        print(f"  Raw rows fetched: {len(df_raw)}")
        all_raw.append(df_raw)

        # Clean & filter
        df_clean = clean_options(df_raw)
        print(f"  Rows after cleaning: {len(df_clean)}")
        all_clean.append(df_clean)

    # Save outputs
    if all_raw:
        df_raw_all = pd.concat(all_raw, ignore_index=True)
        df_raw_all.to_csv("data/raw/options_raw.csv", index=False)
        print(f"\n✓ Raw data saved → data/raw/options_raw.csv  ({len(df_raw_all)} rows)")

    if all_clean:
        df_final = pd.concat(all_clean, ignore_index=True)
        df_final.to_csv("data/processed/options_final.csv", index=False)
        print(f"✓ Final data saved → data/processed/options_final.csv  ({len(df_final)} rows)")

        # Summary
        print("\n── Summary ───────────────────────────────────────────────")
        print(f"  Observation dates : {df_final['ObsDate'].nunique()}")
        print(f"  Unique expirations: {df_final['ExDt'].nunique()}")
        print(f"  Total options     : {len(df_final)}")
        print(f"  Calls / Puts      : {(df_final['OptionType']=='call').sum()} / {(df_final['OptionType']=='put').sum()}")
        print(f"  Maturity range    : {df_final['T'].min():.3f}y — {df_final['T'].max():.3f}y")
        print(f"  Strike range      : {df_final['Strike'].min():.0f} — {df_final['Strike'].max():.0f}")
        print(f"  Rf range          : {df_final['Rf'].min():.4f} — {df_final['Rf'].max():.4f}")
    else:
        print("\n⚠ No data collected. Check your observation dates and internet connection.")

    print("\n=== Done. Load data in other scripts with: ===")
    print("  import pandas as pd")
    print('  df = pd.read_csv("data/processed/options_final.csv", parse_dates=["ObsDate", "ExDt"])')


if __name__ == "__main__":
    main()
