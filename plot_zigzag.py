import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import argparse
import os

def fetch_data(ticker, start_date, end_date):
    # Fetch a bit more data to ensure pivots at edges are caught
    pad_start = pd.Timestamp(start_date) - pd.Timedelta(days=30)
    pad_end = pd.Timestamp(end_date) + pd.Timedelta(days=30)

    df = yf.download(ticker, start=pad_start, end=pad_end, progress=False, auto_adjust=True)
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

def plot_zigzag(ticker, start_date, end_date, output_file="zigzag_plot.png"):
    df = fetch_data(ticker, start_date, end_date)

    # Filter for the plot range
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    plot_df = df.loc[mask]

    if plot_df.empty:
        print("No data found for the specified range.")
        return

    # Find pivots on the full data to avoid edge effects, then filter
    high_idx, low_idx = find_pivots(df, order=5)

    pivots = []
    for idx in high_idx:
        date = df.index[idx]
        if pd.Timestamp(start_date) <= date <= pd.Timestamp(end_date):
            pivots.append({'date': date, 'price': df['High'].iloc[idx], 'type': 'high'})

    for idx in low_idx:
        date = df.index[idx]
        if pd.Timestamp(start_date) <= date <= pd.Timestamp(end_date):
            pivots.append({'date': date, 'price': df['Low'].iloc[idx], 'type': 'low'})

    pivots.sort(key=lambda x: x['date'])

    # Create plot
    plt.figure(figsize=(14, 7))
    plt.plot(plot_df.index, plot_df['Close'], label='Close Price', color='lightgray', alpha=0.7)
    plt.plot(plot_df.index, plot_df['High'], color='gray', linestyle=':', alpha=0.3)
    plt.plot(plot_df.index, plot_df['Low'], color='gray', linestyle=':', alpha=0.3)

    # Draw ZigZag lines and annotate
    last_p = None

    # Filter to ensure we alternate High -> Low properly for drawing
    # or just connect them sequentially as they appear?
    # VCP is High -> Low.

    cleaned_pivots = []
    # Simple logic: if multiple highs, take highest. If multiple lows, take lowest?
    # Or just plot all found local extrema.
    # Let's plot all found extrema connected with lines.

    pivot_dates = [p['date'] for p in pivots]
    pivot_prices = [p['price'] for p in pivots]

    plt.plot(pivot_dates, pivot_prices, color='blue', linestyle='-', marker='o', markersize=4, label='ZigZag')

    # Calculate drops (High -> Low)
    for i in range(len(pivots) - 1):
        p1 = pivots[i]
        p2 = pivots[i+1]

        if p1['type'] == 'high' and p2['type'] == 'low':
            # Calculate depth
            high_price = p1['price']
            low_price = p2['price']
            depth = (high_price - low_price) / high_price

            # Draw annotation
            mid_date = p1['date'] + (p2['date'] - p1['date']) / 2
            mid_price = (high_price + low_price) / 2

            plt.annotate(f"-{depth:.1%}",
                         xy=(mid_date, mid_price),
                         xytext=(0, -10), textcoords='offset points',
                         ha='center', color='red', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))

    plt.title(f"{ticker} ZigZag Analysis ({start_date} to {end_date})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ZigZag and Drawdowns.')
    parser.add_argument('ticker', help='Ticker symbol')
    parser.add_argument('start_date', help='Start Date (YYYY-MM-DD)')
    parser.add_argument('end_date', help='End Date (YYYY-MM-DD)')
    parser.add_argument('--output', default='zigzag_plot.png', help='Output filename')

    args = parser.parse_args()

    plot_zigzag(args.ticker, args.start_date, args.end_date, args.output)
