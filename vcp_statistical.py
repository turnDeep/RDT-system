import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def analyze_vcp_statistical(ticker):
    print(f"--- Statistical VCP Analysis for {ticker} ---")
    df = fetch_data(ticker)

    if df.empty:
        print("No data found.")
        return

    # Calculate percent change
    df['pct_change'] = df['Close'].pct_change()

    # Calculate rolling standard deviations
    # Long term: 50 days
    # Short term: 10 days

    # We use .shift(1) implicitly when using rolling on current data to include "today"?
    # Usually we want the volatility leading up to today.

    df['std_50'] = df['pct_change'].rolling(window=50).std()
    df['std_10'] = df['pct_change'].rolling(window=10).std()

    # Get latest values
    latest = df.iloc[-1]

    std_50 = latest['std_50']
    std_10 = latest['std_10']

    # We can also look at the average of std_50 over time vs current std_10 as per some interpretations,
    # but the text says:
    # "Volatility Ratio: Standard deviation of price changes for the last 10 days divided by standard deviation for the last 50 days."
    # And "short_term_vol < (long_term_vol * 0.5)"

    # Text snippet B says:
    # long_term_vol = df['Close'].pct_change().rolling(50).std().mean()  <-- MEAN of the rolling std?
    # short_term_vol = df['Close'].pct_change().tail(window).std()

    # Let's align with snippet B provided in the text.

    # Re-calculating based on snippet B logic
    # "long_term_vol = df['Close'].pct_change().rolling(50).std().mean()"
    # This implies the average volatility over the entire loaded period (1y)? Or just recent?
    # Usually "Long Term Volatility" is a single number representing the baseline.
    # I will interpret it as the mean of the 50-day rolling std over the last year.

    long_term_vol = df['pct_change'].rolling(window=50).std().mean()

    # "short_term_vol = df['Close'].pct_change().tail(window).std()"
    # Window is 10
    short_term_vol = df['pct_change'].tail(10).std()

    if pd.isna(long_term_vol) or pd.isna(short_term_vol):
        print("Not enough data to calculate volatility.")
        return

    vol_ratio = short_term_vol / long_term_vol

    print(f"Long-term Volatility (Avg 50d Std): {long_term_vol:.4f}")
    print(f"Short-term Volatility (Last 10d Std): {short_term_vol:.4f}")
    print(f"Volatility Ratio: {vol_ratio:.2f}")

    # Condition: Ratio < 0.5 (or significantly lower)
    is_contracted = short_term_vol < (long_term_vol * 0.5)

    print(f"Contraction Check: {short_term_vol:.4f} < {long_term_vol * 0.5:.4f} ? {is_contracted}")

    if is_contracted:
        print("Result: True (Statistical Contraction Detected)")
    else:
        print("Result: False")

if __name__ == "__main__":
    analyze_vcp_statistical("FORM")
