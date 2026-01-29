import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from data_fetcher import RDTDataFetcher
import os
import numpy as np

class RDTChartGenerator:
    def __init__(self):
        self.fetcher = RDTDataFetcher()
        self.data_folder = "data"

    def load_pickle_data(self, filename):
        path = os.path.join(self.data_folder, filename)
        if os.path.exists(path):
            return pd.read_pickle(path)
        return None

    def generate_chart(self, ticker, output_filename=None):
        print(f"Generating chart for {ticker}...")

        # 1. Load Data
        zone_rs_data = self.load_pickle_data("zone_rs_weekly.pkl")
        rs_perc_data = self.load_pickle_data("rs_percentile_histogram_weekly.pkl")
        rs_vol_data = self.load_pickle_data("rs_volatility_adjusted_weekly.pkl")
        rti_data = self.load_pickle_data("rti_weekly.pkl")
        atr_ts_data = self.load_pickle_data("atr_trailing_stop_weekly.pkl")

        # Load Raw Price (Daily) and Resample to Weekly for Main Chart
        price_pkl = self.load_pickle_data("price_data_ohlcv.pkl")

        if price_pkl is None:
            print("Error: price_data_ohlcv.pkl not found.")
            return

        # Extract Ticker Data
        if isinstance(price_pkl.columns, pd.MultiIndex):
            try:
                # Level 0: Close, High, etc. Level 1: Ticker
                df = pd.DataFrame({
                    'Open': price_pkl['Open'][ticker],
                    'High': price_pkl['High'][ticker],
                    'Low': price_pkl['Low'][ticker],
                    'Close': price_pkl['Close'][ticker],
                    'Volume': price_pkl['Volume'][ticker]
                })
            except KeyError:
                print(f"Error: {ticker} not found in price data.")
                return
        else:
            # Single ticker case
            df = price_pkl.copy()

        # Resample to Weekly (W-FRI)
        df.index = pd.to_datetime(df.index)
        weekly_agg = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        df_weekly = df.resample('W-FRI').agg(weekly_agg).dropna()

        # Slicing: Last 2 Years approx (104 weeks)
        if len(df_weekly) > 104:
            df_weekly = df_weekly.iloc[-104:]

        plot_df = df_weekly.copy()
        valid_idx = plot_df.index

        # --- Prepare Additional Plots (apds) ---
        apds = []

        # 1. ATR Trailing Stop (Panel 0)
        # Requirement: Colored Bars (Green, Blue, Yellow, Red) based on Trend State
        # Trails: Fast (Blue/Yellow?), Slow (Green/Red?).
        # Fill: Green (Fast > Slow), Red (Slow > Fast).

        main_plot_type = 'candle' # Default if no ATR data

        if atr_ts_data:
            try:
                fast_trail = atr_ts_data["Fast_Trail"][ticker].reindex(valid_idx)
                slow_trail = atr_ts_data["Slow_Trail"][ticker].reindex(valid_idx)
                states = atr_ts_data["Trend_State"][ticker].reindex(valid_idx)

                # 1. Trails
                # Colors based on prompt/image (Slow Trail Green/Red)
                # We'll use: Fast (Blue), Slow (Red) for simplicity unless strict requirement.
                # Actually, Image 9340 shows "Slow Trail" with color Green and Red.
                # This implies the line changes color.
                # Let's try to implement color change for Slow Trail if possible?
                # Or just use Red for Slow, Blue for Fast.
                # Fill logic: Green (Bull), Red (Bear).

                apds.append(mpf.make_addplot(fast_trail, panel=0, color='blue', width=1.0))

                # Fill logic
                v_fast = fast_trail.values
                v_slow = slow_trail.values
                mask_valid = ~np.isnan(v_fast) & ~np.isnan(v_slow)

                where_bull = np.zeros_like(v_fast, dtype=bool)
                where_bull[mask_valid] = v_fast[mask_valid] > v_slow[mask_valid]

                where_bear = np.zeros_like(v_fast, dtype=bool)
                where_bear[mask_valid] = v_fast[mask_valid] <= v_slow[mask_valid]

                # Slow Trail with Fill
                # We can't change line color easily without splitting, but simple Red is standard for TS.
                apds.append(mpf.make_addplot(
                    slow_trail, panel=0, color='red', width=1.5,
                    fill_between=dict(y1=v_fast, y2=v_slow, where=where_bull, color='green', alpha=0.1)
                ))
                 # Add another for Bear fill
                apds.append(mpf.make_addplot(
                    slow_trail, panel=0, color='red', width=1.5,
                    fill_between=dict(y1=v_fast, y2=v_slow, where=where_bear, color='red', alpha=0.1)
                ))

                # 2. Colored Candles
                # We need to suppress main plot candles and add our own.
                main_plot_type = 'line' # We will make it invisible later

                # Create 4 DataFrames for states
                # 0: Red, 1: Yellow, 2: Blue, 3: Green
                df_red = plot_df.copy()
                df_red[states != 0] = np.nan

                df_yellow = plot_df.copy()
                df_yellow[states != 1] = np.nan

                df_blue = plot_df.copy()
                df_blue[states != 2] = np.nan

                df_green = plot_df.copy()
                df_green[states != 3] = np.nan

                # Add candle plots
                # Note: 'type' argument in make_addplot is supported in recent mplfinance versions.
                # Ensure we have OHLC data in these DFs (we copied plot_df).

                # Colors
                # Red, Yellow, Blue, Green (Lime)
                apds.append(mpf.make_addplot(df_red, type='candle', panel=0, color='red'))
                apds.append(mpf.make_addplot(df_yellow, type='candle', panel=0, color='yellow'))
                apds.append(mpf.make_addplot(df_blue, type='candle', panel=0, color='#1e90ff')) # DodgerBlue
                apds.append(mpf.make_addplot(df_green, type='candle', panel=0, color='lime'))

            except KeyError:
                print(f"Warning: {ticker} not found in ATR TS data.")
                main_plot_type = 'candle'

        # 2. Zone RS (Panel 2)
        if zone_rs_data:
            try:
                ratio = zone_rs_data["Ratio"][ticker].reindex(valid_idx)
                momentum = zone_rs_data["Momentum"][ticker].reindex(valid_idx)
                zones = zone_rs_data["Zone"][ticker].reindex(valid_idx)

                y_min = min(ratio.min(), momentum.min())
                y_max = max(ratio.max(), momentum.max())
                y_min -= abs(y_min) * 0.1
                y_max += abs(y_max) * 0.1

                mask_dead = zones == 0
                mask_lift = zones == 1
                mask_drift = zones == 2
                mask_power = zones == 3

                height = y_max - y_min
                def create_bar_series(mask):
                    s = pd.Series(np.nan, index=valid_idx)
                    s[mask] = height
                    return s

                apds.append(mpf.make_addplot(create_bar_series(mask_dead), type='bar', panel=2, color='red', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(create_bar_series(mask_lift), type='bar', panel=2, color='blue', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(create_bar_series(mask_drift), type='bar', panel=2, color='yellow', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(create_bar_series(mask_power), type='bar', panel=2, color='green', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))

                apds.append(mpf.make_addplot(ratio, panel=2, color='blue', ylabel='Zone RS'))
                apds.append(mpf.make_addplot(momentum, panel=2, color='orange', secondary_y=False))

            except KeyError:
                pass

        # 3. Historical Percentile (Panel 3)
        if rs_perc_data:
            try:
                perc = rs_perc_data["Percentile_1M"][ticker].reindex(valid_idx)
                colors = []
                for v in perc:
                    if pd.isna(v): colors.append('white')
                    elif v < 10: colors.append('#d86eef')
                    elif v < 30: colors.append('#d3eeff')
                    elif v < 50: colors.append('#4e7eff')
                    elif v < 70: colors.append('#96d7ff')
                    elif v < 85: colors.append('#80cfff')
                    elif v < 95: colors.append('#1eaaff')
                    else: colors.append('#30b0ff')

                apds.append(mpf.make_addplot(perc, type='bar', panel=3, color=colors, ylabel='Hist %'))
            except KeyError:
                pass

        # 4. Volatility Adjusted RS (Panel 4)
        if rs_vol_data:
            try:
                rs_val = rs_vol_data["RS_Values"][ticker].reindex(valid_idx)
                rs_ma = rs_vol_data["RS_MA"][ticker].reindex(valid_idx)

                rs_pos = rs_val.apply(lambda x: x if x >= 0 else np.nan)
                rs_neg = rs_val.apply(lambda x: x if x <= 0 else np.nan)

                apds.append(mpf.make_addplot(rs_pos, panel=4, color='blue', width=1.5, ylabel='Vol Adj RS'))
                apds.append(mpf.make_addplot(rs_neg, panel=4, color='fuchsia', width=1.5))

                if not rs_ma.isna().all():
                    ma_diff = rs_ma.diff()
                    ma_rising = rs_ma.copy()
                    ma_falling = rs_ma.copy()

                    ma_rising_mask = ma_diff >= 0
                    ma_falling_mask = ma_diff < 0

                    ma_rising[~ma_rising_mask] = np.nan
                    ma_falling[~ma_falling_mask] = np.nan

                    apds.append(mpf.make_addplot(ma_rising, panel=4, color='blue', width=1.5))
                    apds.append(mpf.make_addplot(ma_falling, panel=4, color='fuchsia', width=1.5))

                # Fills
                v_rs = rs_val.values
                v_ma = rs_ma.values
                v_zero = np.zeros_like(v_rs)
                mask_valid_rs = ~np.isnan(v_rs)
                mask_valid_ma = ~np.isnan(v_ma)

                where_rs_pos = np.zeros_like(v_rs, dtype=bool)
                where_rs_pos[mask_valid_rs] = v_rs[mask_valid_rs] > 0
                where_rs_neg = np.zeros_like(v_rs, dtype=bool)
                where_rs_neg[mask_valid_rs] = v_rs[mask_valid_rs] <= 0

                apds.append(mpf.make_addplot(rs_val, panel=4, color='blue', alpha=0, fill_between=dict(y1=v_rs, y2=v_zero, where=where_rs_pos, color='#0084ff', alpha=0.2), secondary_y=False))
                apds.append(mpf.make_addplot(rs_val, panel=4, color='pink', alpha=0, fill_between=dict(y1=v_rs, y2=v_zero, where=where_rs_neg, color='#ff52c8', alpha=0.2), secondary_y=False))

                mask_both = mask_valid_rs & mask_valid_ma
                where_rs_gt_ma = np.zeros_like(v_rs, dtype=bool)
                where_rs_gt_ma[mask_both] = v_rs[mask_both] > v_ma[mask_both]
                where_rs_lt_ma = np.zeros_like(v_rs, dtype=bool)
                where_rs_lt_ma[mask_both] = v_rs[mask_both] <= v_ma[mask_both]

                apds.append(mpf.make_addplot(rs_val, panel=4, color='blue', alpha=0, fill_between=dict(y1=v_rs, y2=v_ma, where=where_rs_gt_ma, color='#0084ff', alpha=0.2), secondary_y=False))
                apds.append(mpf.make_addplot(rs_val, panel=4, color='pink', alpha=0, fill_between=dict(y1=v_rs, y2=v_ma, where=where_rs_lt_ma, color='#ff52c8', alpha=0.2), secondary_y=False))

            except KeyError:
                pass

        # 5. RTI (Panel 5)
        if rti_data:
            try:
                rti_val = rti_data["RTI_Values"][ticker].reindex(valid_idx)
                rti_sig = rti_data["RTI_Signals"][ticker].reindex(valid_idx)

                apds.append(mpf.make_addplot(rti_val, panel=5, color='blue', ylabel='RTI', width=2.0))

                exp_dots = rti_val.copy()
                exp_dots[:] = np.nan
                mask_exp = (rti_sig == 3)
                exp_dots[mask_exp] = rti_val[mask_exp]

                if mask_exp.any() and not exp_dots.isna().all():
                     apds.append(mpf.make_addplot(exp_dots, panel=5, type='scatter', markersize=30, color='green', marker='^'))

                dots = rti_val.copy()
                dots[:] = np.nan
                mask_dot = (rti_sig == 2)
                dots[mask_dot] = rti_val[mask_dot]

                if not dots.isna().all():
                    apds.append(mpf.make_addplot(dots, panel=5, type='scatter', markersize=20, color='orange'))

            except KeyError:
                pass

        # 6. Plotting
        if output_filename is None:
            output_filename = f"{ticker}_weekly_chart.png"

        rc_params = {'axes.grid': True, 'grid.linestyle': ':'}
        s = mpf.make_mpf_style(base_mpf_style='yahoo', rc=rc_params)

        # Main Plot Logic
        # If we use main_plot_type='line', we can hide it by setting linecolor='None' (needs testing) or alpha=0?
        # mpf.plot doesn't support alpha for line easily in kwargs.
        # But we can set `linecolor='white'` if background is white?
        # 'yahoo' style has white background.

        # Another option: Use plot_df but with all NaNs for Close?
        # Then axis range won't be calculated correctly.

        # Best bet: Use 'line' with very thin width or white color?
        # But grid lines might be covered? No, line is on top.

        kwargs = {}
        if main_plot_type == 'line':
             # Attempt to make invisible
             # Note: passing `update_width_config` helps
             kwargs['linecolor'] = 'white' # Hacky, assumes white background
             # kwargs['linewidths'] = 0 # might work

        try:
            fig, axes = mpf.plot(
                plot_df,
                type=main_plot_type,
                style=s,
                addplot=apds,
                volume=True,
                volume_panel=1,
                panel_ratios=(4, 1, 1, 1, 1, 1),
                returnfig=True,
                figsize=(12, 16),
                tight_layout=True,
                title=f"{ticker} Weekly Analysis",
                **kwargs
            )

            fig.savefig(output_filename, bbox_inches='tight')
            print(f"Chart saved to {output_filename}")
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting {ticker}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("-o", "--output", type=str, help="Output filename")
    args = parser.parse_args()

    generator = RDTChartGenerator()
    generator.generate_chart(args.ticker, args.output)
