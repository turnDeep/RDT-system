import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import datetime
import argparse

def fetch_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def find_pivots(df, order=5):
    highs = df['High'].values
    lows = df['Low'].values
    high_idx = argrelextrema(highs, np.greater, order=order)[0]
    low_idx = argrelextrema(lows, np.less, order=order)[0]
    return high_idx, low_idx

def analyze_vcp_geometric(ticker, target_date=None):
    print(f"--- Geometric VCP Analysis for {ticker} ---")
    if target_date:
        print(f"Target Date: {target_date}")

    df = fetch_data(ticker, period="2y") # Fetch more to allow cutting back

    if df.empty:
        print("No data found.")
        return

    if target_date:
        target_ts = pd.Timestamp(target_date)
        df = df[df.index <= target_ts]
        if df.empty:
            print(f"No data found up to {target_date}.")
            return
        print(f"Data truncated to {df.index[-1].date()}")

    # Find pivots
    high_idx, low_idx = find_pivots(df, order=5)

    pivots = []
    for idx in high_idx:
        pivots.append({'date': df.index[idx], 'price': df['High'].iloc[idx], 'type': 'high', 'idx': idx})
    for idx in low_idx:
        pivots.append({'date': df.index[idx], 'price': df['Low'].iloc[idx], 'type': 'low', 'idx': idx})

    pivots.sort(key=lambda x: x['idx'])

    contractions = []
    last_high = None

    for p in pivots:
        if p['type'] == 'high':
            last_high = p
        elif p['type'] == 'low' and last_high is not None:
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
                last_high = None

    if len(contractions) < 2:
        print("Result: False (Not enough contractions found)")
        return

    recent_contractions = contractions[-4:]

    print(f"Found {len(recent_contractions)} recent contractions (High -> Low):")
    for i, c in enumerate(recent_contractions):
        print(f"  {i+1}. {c['start_date'].date()} -> {c['end_date'].date()}: Depth = {c['depth']:.2%}")

    if len(recent_contractions) < 2:
        print("Result: False (Not enough recent contractions)")
        return

    depths = [c['depth'] for c in recent_contractions]
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
    parser = argparse.ArgumentParser(description='Analyze Geometric VCP.')
    parser.add_argument('ticker', nargs='?', default="FORM", help='Ticker symbol')
    parser.add_argument('--date', help='Target date (YYYY-MM-DD)')
    args = parser.parse_args()

    analyze_vcp_geometric(args.ticker, target_date=args.date)
