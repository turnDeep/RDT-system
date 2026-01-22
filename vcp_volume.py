import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import sys

def fetch_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def analyze_vcp_volume(ticker, target_date=None):
    print(f"--- Volume VCP Analysis for {ticker} ---")
    if target_date:
        print(f"Target Date: {target_date}")

    df = fetch_data(ticker, period="2y") # Fetch more to ensure we have enough history for the target date

    if df.empty:
        print("No data found.")
        return

    # Filter by target date if provided
    if target_date:
        target_ts = pd.Timestamp(target_date)
        df = df[df.index <= target_ts]
        if df.empty:
            print(f"No data found up to {target_date}.")
            return
        print(f"Data truncated to {df.index[-1].date()}")

    # Ensure we have enough data
    if len(df) < 50:
        print("Not enough data (need at least 50 days).")
        return

    # 1. Volume Moving Average Check
    # Vol_SMA_5 < 0.7 * Vol_SMA_50

    vol_sma_5 = df['Volume'].rolling(window=5).mean().iloc[-1]
    vol_sma_50 = df['Volume'].rolling(window=50).mean().iloc[-1]

    dry_up_condition = vol_sma_5 < (0.7 * vol_sma_50)

    print(f"Volume SMA 5: {vol_sma_5:,.0f}")
    print(f"Volume SMA 50: {vol_sma_50:,.0f}")
    print(f"Dry Up Check: {vol_sma_5:,.0f} < {0.7 * vol_sma_50:,.0f} (70% of SMA50) ? {dry_up_condition}")

    # 2. Up/Down Volume Ratio (Last 50 days)
    # Up day: Close > Previous Close
    # Down day: Close < Previous Close

    # We need the last 50 days.
    # To correctly calculate Up/Down for the last 50 days, we need 51 rows (row 0 to calculate change for row 1).
    subset_calc = df.tail(51).copy()

    if len(subset_calc) < 51:
         print("Warning: Less than 51 days of data available for Ratio calculation.")

    subset_calc['prev_close'] = subset_calc['Close'].shift(1)
    # Drop the first row as it has NaN prev_close
    subset_calc = subset_calc.iloc[1:]

    up_days = subset_calc[subset_calc['Close'] > subset_calc['prev_close']]
    down_days = subset_calc[subset_calc['Close'] < subset_calc['prev_close']]

    up_volume = up_days['Volume'].sum()
    down_volume = down_days['Volume'].sum()

    if down_volume == 0:
        ud_ratio = float('inf')
    else:
        ud_ratio = up_volume / down_volume

    accumulation_condition = ud_ratio >= 1.0
    strong_accumulation = ud_ratio >= 1.2

    print(f"Up Volume (50d): {up_volume:,.0f}")
    print(f"Down Volume (50d): {down_volume:,.0f}")
    print(f"Up/Down Volume Ratio: {ud_ratio:.2f}")
    print(f"Accumulation Check (>= 1.0): {accumulation_condition}")
    print(f"Strong Accumulation Check (>= 1.2): {strong_accumulation}")

    if dry_up_condition and accumulation_condition:
        print("Result: True (Volume VCP Pattern Detected)")
    else:
        print("Result: False")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze VCP Volume.')
    parser.add_argument('ticker', nargs='?', default="FORM", help='Ticker symbol')
    parser.add_argument('--date', help='Target date (YYYY-MM-DD)')

    args = parser.parse_args()

    analyze_vcp_volume(args.ticker, target_date=args.date)
