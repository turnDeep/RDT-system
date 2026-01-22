import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import datetime

def fetch_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    # yfinance might return MultiIndex columns if ticker is a list or just one.
    # Ensure flat index for easier access if needed, but for single ticker it's usually fine.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Drop rows with NaN if any (though yf usually handles it)
    df.dropna(inplace=True)
    return df

def find_pivots(df, order=5):
    """
    Finds pivot highs and lows.
    order=5 means looking 5 days before and 5 days after.
    """
    # iloc is used to get numpy array of prices
    highs = df['High'].values
    lows = df['Low'].values

    # argrelextrema returns indices
    high_idx = argrelextrema(highs, np.greater, order=order)[0]
    low_idx = argrelextrema(lows, np.less, order=order)[0]

    return high_idx, low_idx

def analyze_vcp_geometric(ticker):
    print(f"--- Geometric VCP Analysis for {ticker} ---")
    df = fetch_data(ticker)

    if df.empty:
        print("No data found.")
        return

    # Find pivots
    high_idx, low_idx = find_pivots(df, order=5)

    # Combine pivots into a single timeline
    # We want to identify sequences of High -> Low -> High -> Low ...
    # But VCP is specifically about the depth of correction (High to Low).

    pivots = []
    for idx in high_idx:
        pivots.append({'date': df.index[idx], 'price': df['High'].iloc[idx], 'type': 'high', 'idx': idx})
    for idx in low_idx:
        pivots.append({'date': df.index[idx], 'price': df['Low'].iloc[idx], 'type': 'low', 'idx': idx})

    # Sort by date/index
    pivots.sort(key=lambda x: x['idx'])

    # Identify contractions: High followed by Low
    contractions = []

    # Simple state machine to find pairs
    last_high = None

    for p in pivots:
        if p['type'] == 'high':
            last_high = p
        elif p['type'] == 'low' and last_high is not None:
            # We found a High -> Low pair
            # Check if Low is after High (it should be since we sorted)
            if p['idx'] > last_high['idx']:
                high_price = last_high['price']
                low_price = p['price']
                depth = (high_price - low_price) / high_price
                contractions.append({
                    'start_date': last_high['date'],
                    'end_date': p['date'],
                    'high': high_price,
                    'low': low_price,
                    'depth': depth
                })
                # Reset last_high to avoid reusing same high for multiple lows
                # (though standard zigzag usually goes H-L-H-L)
                last_high = None

    # Filter only recent contractions if we want to check "Current" status,
    # but the prompt asks to analyze the last 1 year.
    # We will look at the sequence of contractions found.

    if len(contractions) < 2:
        print("Result: False (Not enough contractions found)")
        return

    # To check VCP, we look at the last few contractions.
    # VCP pattern implies Volatility Contraction Pattern.
    # Depths should be decreasing.
    # e.g. 20% -> 10% -> 5%

    # Let's take the last sequence of contractions that are "connected"
    # (i.e., belong to the same base structure).
    # Ideally, they should be relatively close in time.
    # For this simplified script, we will just look at the last 3-4 contractions if available.

    recent_contractions = contractions[-4:] # Take up to last 4

    print(f"Found {len(recent_contractions)} recent contractions (High -> Low):")
    for i, c in enumerate(recent_contractions):
        print(f"  {i+1}. {c['start_date'].date()} -> {c['end_date'].date()}: Depth = {c['depth']:.2%}")

    if len(recent_contractions) < 2:
        print("Result: False (Not enough recent contractions)")
        return

    # Check for decreasing depth (allow some leeway or strict?)
    # The text says: "depth i < depth i-1"
    # And "Last contraction < 10%"

    depths = [c['depth'] for c in recent_contractions]

    # Check if depths are generally decreasing
    # We allow a small error margin or strict check? Text says "Strict monotonic decrease is not always required",
    # but "latest is smaller than previous" is key.

    # Check if the last one is the smallest or at least smaller than the second to last
    last_depth = depths[-1]
    prev_depth = depths[-2]

    is_tightening = last_depth < prev_depth
    is_tight_enough = last_depth < 0.10

    print(f"Tightening Check: {last_depth:.2%} < {prev_depth:.2%} ? {is_tightening}")
    print(f"Tight Enough Check: {last_depth:.2%} < 10% ? {is_tight_enough}")

    if is_tightening and is_tight_enough:
        print("Result: True (VCP Pattern Detected)")
    else:
        print("Result: False")

if __name__ == "__main__":
    analyze_vcp_geometric("FORM")
