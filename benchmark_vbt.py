import vectorbt as vbt
import numpy as np
import pandas as pd
import time
import argparse
import yfinance as yf

# --- Mock Data Generation for Benchmarking ---
def generate_data(n_rows, n_cols):
    print(f"Generating data: {n_rows} days, {n_cols} tickers...")
    index = pd.date_range(start="2020-01-01", periods=n_rows)
    # Random walk
    np.random.seed(42)
    close = np.random.normal(0, 1, size=(n_rows, n_cols)).cumsum(axis=0) + 100
    volume = np.random.randint(1000, 100000, size=(n_rows, n_cols))

    # Create DataFrame (single level columns for simplicity in basic pandas, or MultiIndex for structure)
    # vbt likes separate DataFrames for Open, High, Low, Close, Volume
    # We will simulate Close and Volume for Volume VCP check
    close_df = pd.DataFrame(close, index=index, columns=[f"T_{i}" for i in range(n_cols)])
    volume_df = pd.DataFrame(volume, index=index, columns=[f"T_{i}" for i in range(n_cols)])

    return close_df, volume_df

# --- Iterative approach (mimicking loop in scan_volume_generic) ---
def analyze_volume_vcp_iterative(close_df, volume_df):
    start_time = time.time()
    results = {}

    # Iterate over each ticker (column)
    for col in close_df.columns:
        c = close_df[col]
        v = volume_df[col]

        # Calculate indicators iteratively per ticker
        vol_sma_5 = v.rolling(window=5).mean()
        vol_sma_50 = v.rolling(window=50).mean()

        # Dry up condition
        dry_up = vol_sma_5 < (0.7 * vol_sma_50)

        # Accumulation Ratio (rolling 50 days)
        # This is expensive to do purely rolling with accumulation logic in pandas without apply or loop
        # We will use a simplified approach: just calc for the *last* day to match original script logic?
        # But for benchmark we want 1 year analysis. Original script checked ONE day.
        # The prompt asks "perform 1 year analysis fast".
        # So we should calculate the condition for ALL days in the year.

        # Up/Down Volume Ratio Rolling
        # Create Up/Down volume series
        price_change = c.diff()
        up_vol = v.where(price_change > 0, 0)
        down_vol = v.where(price_change < 0, 0)

        roll_up = up_vol.rolling(50).sum()
        roll_down = down_vol.rolling(50).sum()
        ratio = roll_up / roll_down

        accum = ratio >= 1.0

        final_signal = dry_up & accum
        results[col] = final_signal.sum() # Count hits

    end_time = time.time()
    return end_time - start_time

# --- VectorBT approach ---
def analyze_volume_vcp_vectorized(close_df, volume_df):
    start_time = time.time()

    # vbt can handle the entire DataFrame at once
    # 1. Volume SMAs
    # vbt.MA.run usually takes 1d or 2d array.

    # We can just use pandas vectorized operations on the whole DF?
    # Yes, pandas is already vectorized. But vbt offers indicator factories that might be faster or cache.
    # Let's stick to pure pandas vectorization first as "Vectorized approach" because vbt builds on it.
    # Or use vbt indicators.

    # Using vbt.MA
    # vbt MA run returns a wrapper object or DataFrame with specific index/columns
    # We should ensure alignment.
    sma5_obj = vbt.MA.run(volume_df, 5)
    sma50_obj = vbt.MA.run(volume_df, 50)

    sma5 = sma5_obj.ma
    sma50 = sma50_obj.ma

    # Explicit alignment not usually needed if index/columns match, but vbt might modify them if multiindex
    # However, simple DataFrames should be fine. The error suggests mismatch.
    # Let's verify shape/columns or force alignment.
    # Or just use pandas direct rolling for fairness if vbt.MA is adding complex indexing.
    # But for vbt benchmark, we should use vbt.
    # The error "Can only compare identically-labeled..." implies maybe vbt added parameter levels to columns?

    # Check if columns have MultiIndex now
    # sma5.columns -> might be (window, ticker)

    # Let's stick to pandas for the comparison if vbt wrapper complicates the simple 1-1 comparison without config.
    # OR fix it by extracting values or using vbt comparison.

    # Just using pandas for SMA here ensures "Vectorized" comparison is valid for the logic itself.
    sma5 = volume_df.rolling(5).mean()
    sma50 = volume_df.rolling(50).mean()

    dry_up = sma5 < (0.7 * sma50)

    # 2. Accumulation Ratio
    # vbt doesn't have a built-in "Up/Down Volume Ratio" indicator likely, so we build it using vbt expressions or pandas

    # Calculate price change across all columns at once
    # close_df.vbt...

    # Using pandas underlying vectorization (which vbt relies on)
    price_change = close_df.diff()

    # We need to broadcast 0 where condition not met
    # numpy where is fast
    v_vals = volume_df.values
    p_diff_vals = price_change.values

    up_vol_vals = np.where(p_diff_vals > 0, v_vals, 0)
    down_vol_vals = np.where(p_diff_vals < 0, v_vals, 0)

    # Rolling sum. vbt has fast rolling functions?
    # vbt.rolling_sum? No, but pandas rolling is okay.
    # Actually vbt has `vbt.base.array_wrapper.ArrayWrapper` etc but simpler to use vbt generic indicators or just pandas for this part if vbt doesn't have specific one.
    # However, to be "vectorbt style", we might define a custom indicator.

    # Let's use `vbt.IndicatorFactory` to create a reusable, fully vectorized indicator.

    def accum_ratio_func(close, volume, window):
        # This function receives 2D arrays if run on multiple columns
        diff = np.diff(close, axis=0, prepend=np.nan)
        up_vol = np.where(diff > 0, volume, 0)
        down_vol = np.where(diff < 0, volume, 0)

        # We need rolling sum.
        # Efficient rolling sum in numpy is tricky without stride_tricks or pandas.
        # We can use pd.DataFrame(up_vol).rolling... but that converts back to pandas.
        # vbt generic run?

        # For fairness, let's just use pandas rolling on the whole DF inside this function
        # assuming inputs are pandas Series/DFs or we wrap them.

        # Actually, let's just do it outside for the benchmark, matching logic.
        pass

    # Re-doing vectorized with pandas (since that IS the vectorized way in python usually)
    # vbt optimizes Strategy testing (signals -> entries -> exits).
    # Here we are just detecting signals.

    # If we treat Up Volume and Down Volume as indicators
    # We can use vbt.Builder to make an indicator?
    # Or just calc it.

    up_vol_df = pd.DataFrame(up_vol_vals, index=close_df.index, columns=close_df.columns)
    down_vol_df = pd.DataFrame(down_vol_vals, index=close_df.index, columns=close_df.columns)

    roll_up = up_vol_df.rolling(50).sum()
    roll_down = down_vol_df.rolling(50).sum()

    ratio = roll_up / roll_down
    accum = ratio >= 1.0

    final_signal = dry_up & accum

    # vbt allows analyzing these signals (Entries/Exits) easily
    # e.g., vbt.Portfolio.from_signals(close_df, entries=final_signal, ...)
    # But for just DETECTION speed, we measured the calc time.

    count = final_signal.sum().sum()

    end_time = time.time()
    return end_time - start_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=int, default=100) # 100 tickers
    parser.add_argument('--days', type=int, default=365)    # 1 year
    args = parser.parse_args()

    close, volume = generate_data(args.days, args.tickers)

    print(f"--- Benchmarking Volume VCP Logic ({args.tickers} tickers, {args.days} days) ---")

    t_iter = analyze_volume_vcp_iterative(close, volume)
    print(f"Iterative Loop Time: {t_iter:.4f} sec")

    t_vect = analyze_volume_vcp_vectorized(close, volume)
    print(f"Vectorized (Pandas/VBT) Time: {t_vect:.4f} sec")

    speedup = t_iter / t_vect if t_vect > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
