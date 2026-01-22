import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema
import datetime

def get_tickers_from_csv(filename='stock.csv'):
    try:
        df = pd.read_csv(filename)
        # Assume first column is tickers or a column named 'Ticker' or 'Symbol'
        if 'Ticker' in df.columns:
            return df['Ticker'].tolist()
        elif 'Symbol' in df.columns:
            return df['Symbol'].tolist()
        else:
            return df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

def analyze_geometric_history_fast(df_high, df_low):
    """
    Optimized Geometric VCP analysis.
    Iterates pivots instead of days.
    """
    highs = df_high.values
    lows = df_low.values
    dates = df_high.index
    n = len(dates)
    order = 5

    high_idxs = argrelextrema(highs, np.greater, order=order)[0]
    low_idxs = argrelextrema(lows, np.less, order=order)[0]

    pivots = []
    for idx in high_idxs:
        pivots.append({'idx': idx, 'price': highs[idx], 'type': 'high'})
    for idx in low_idxs:
        pivots.append({'idx': idx, 'price': lows[idx], 'type': 'low'})
    pivots.sort(key=lambda x: x['idx'])

    result_array = np.zeros(n, dtype=bool)

    # Iterate through pivot sequence
    # At any point P_k, the state applies from time (P_k + 5) until (P_{k+1} + 5)

    for i in range(len(pivots)):
        # Visible pivots at this point are 0..i
        # Check logic using pivots[0..i]

        current_visible = pivots[:i+1]

        # Logic check
        # Reuse check_vcp_logic but inline or simplified for speed
        # Need at least 2 contractions
        contractions = []
        last_h = None
        for p in current_visible:
            if p['type'] == 'high':
                last_h = p
            elif p['type'] == 'low' and last_h:
                if p['idx'] > last_h['idx']:
                    depth = (last_h['price'] - p['price']) / last_h['price']
                    contractions.append(depth)
                    last_h = None

        valid = False
        if len(contractions) >= 2:
            recent = contractions[-4:]
            if len(recent) >= 2:
                last_depth = recent[-1]
                prev_depth = recent[-2]
                if last_depth < prev_depth and last_depth < 0.10:
                    valid = True

        if valid:
            # Mark days where this state is active
            # Active from confirmation of current pivot (idx + 5)
            # Until confirmation of NEXT pivot (next_idx + 5)

            start_day_idx = pivots[i]['idx'] + order

            if i + 1 < len(pivots):
                end_day_idx = pivots[i+1]['idx'] + order
            else:
                end_day_idx = n # Until end of data

            # Bounds check
            if start_day_idx < n:
                if end_day_idx > n: end_day_idx = n
                result_array[start_day_idx:end_day_idx] = True

    return pd.Series(result_array, index=dates)

def main():
    tickers = get_tickers_from_csv()
    if not tickers:
        print("No tickers found.")
        return

    print(f"Total tickers: {len(tickers)}. Processing in batches...")

    start_date = (pd.Timestamp.now() - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=365)

    BATCH_SIZE = 500

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        print(f"Processing batch {i} to {i+len(batch)}...")

        try:
            data = yf.download(batch, start=start_date, group_by='ticker', progress=False, auto_adjust=True)
        except Exception as e:
            print(f"Batch download failed: {e}")
            continue

        if data.empty: continue

        if len(batch) == 1:
             data.columns = pd.MultiIndex.from_product([data.columns, batch])

        # Volume Vectorized
        data = data.swaplevel(0, 1, axis=1)
        data.sort_index(axis=1, level=0, inplace=True)

        try:
            vol_df = data['Volume']
            close_df = data['Close']
            high_df = data['High']
            low_df = data['Low']
        except KeyError:
             # Data might be malformed or all failed
             continue

        sma5 = vol_df.rolling(5).mean()
        sma50 = vol_df.rolling(50).mean()
        cond_dry_up = sma5 < (0.7 * sma50)

        diff = close_df.diff()
        up_vol = vol_df.where(diff > 0, 0)
        down_vol = vol_df.where(diff < 0, 0)
        roll_up = up_vol.rolling(50).sum()
        roll_down = down_vol.rolling(50).sum()
        ratio = roll_up / roll_down.replace(0, np.nan)
        cond_accum = ratio >= 1.0

        mask_volume = cond_dry_up & cond_accum

        # Geometric Optimized
        mask_geometric = pd.DataFrame(False, index=data.index, columns=close_df.columns)

        for t in batch:
            if t not in close_df.columns: continue

            t_high = high_df[t].dropna()
            t_low = low_df[t].dropna()
            if len(t_high) < 50: continue

            geo_series = analyze_geometric_history_fast(t_high, t_low)
            mask_geometric[t] = geo_series

        final_mask = mask_volume & mask_geometric
        final_mask = final_mask[final_mask.index >= cutoff_date]

        results = final_mask.stack()
        hits = results[results]

        for (date, ticker), _ in hits.items():
            print(f"[{date.date()}] {ticker}: Geometric VCP + Volume VCP Detected")

if __name__ == "__main__":
    main()
