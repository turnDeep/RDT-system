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
                # Handle MultiIndex safely
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

        # 1. ATR Trailing Stop (Main Chart Overlay)
        if atr_ts_data:
            try:
                fast_trail = atr_ts_data["Fast_Trail"][ticker].reindex(valid_idx)
                slow_trail = atr_ts_data["Slow_Trail"][ticker].reindex(valid_idx)
                trend_state = atr_ts_data["Trend_State"][ticker].reindex(valid_idx)
                signals = atr_ts_data["Signals"][ticker].reindex(valid_idx)

                # --- 1.1 Colored Candles ---
                # State: 0=Red, 1=Yellow, 2=Blue, 3=Green
                ohlc_green = plot_df.copy()
                ohlc_blue = plot_df.copy()
                ohlc_yellow = plot_df.copy()
                ohlc_red = plot_df.copy()

                # Masking
                # We set values to NaN where state doesn't match
                ohlc_green.loc[trend_state != 3] = np.nan
                ohlc_blue.loc[trend_state != 2] = np.nan
                ohlc_yellow.loc[trend_state != 1] = np.nan
                ohlc_red.loc[trend_state != 0] = np.nan

                # Add Candle Plots (Panel 0)
                # Note: 'type' argument in make_addplot works for 'candle' in recent versions
                apds.append(mpf.make_addplot(ohlc_green, type='candle', panel=0, color='lime'))
                apds.append(mpf.make_addplot(ohlc_blue, type='candle', panel=0, color='blue'))
                apds.append(mpf.make_addplot(ohlc_yellow, type='candle', panel=0, color='yellow'))
                apds.append(mpf.make_addplot(ohlc_red, type='candle', panel=0, color='red'))

                # --- 1.2 Trail Lines (Split Colors) ---
                # Fast Trail: Blue if Fast > Slow (Bullish context for Fast), Yellow if Fast <= Slow
                # Slow Trail: Green if Fast > Slow, Red if Fast <= Slow

                # Logic check from user request:
                # "Fast Trail is Blue/Yellow? User image shows Fast Trail (thin) and Slow Trail (thick)?"
                # Pine Script:
                # TS1 (Fast): color=Trail1 > Trail2 ? color.blue : color.yellow
                # TS2 (Slow): color=Trail1 > Trail2 ? color.green : color.red

                cond_bull = fast_trail > slow_trail

                fast_blue = fast_trail.copy()
                fast_yellow = fast_trail.copy()
                fast_blue[~cond_bull] = np.nan
                fast_yellow[cond_bull] = np.nan

                slow_green = slow_trail.copy()
                slow_red = slow_trail.copy()
                slow_green[~cond_bull] = np.nan
                slow_red[cond_bull] = np.nan

                # Add Lines
                apds.append(mpf.make_addplot(fast_blue, panel=0, color='blue', width=1.5))
                apds.append(mpf.make_addplot(fast_yellow, panel=0, color='yellow', width=1.5))
                apds.append(mpf.make_addplot(slow_green, panel=0, color='green', width=2.0))
                apds.append(mpf.make_addplot(slow_red, panel=0, color='red', width=2.0))

                # --- 1.3 Fill ---
                # Fill between Fast and Slow
                # Greenish if Bull (Fast > Slow), Reddish if Bear

                # Use invisible line to drive fill
                # We need numpy arrays for 'where'
                v_fast = fast_trail.values
                v_slow = slow_trail.values

                mask_valid = (~np.isnan(v_fast)) & (~np.isnan(v_slow))
                where_bull = np.zeros_like(v_fast, dtype=bool)
                where_bull[mask_valid] = v_fast[mask_valid] > v_slow[mask_valid]

                where_bear = np.zeros_like(v_fast, dtype=bool)
                where_bear[mask_valid] = v_fast[mask_valid] <= v_slow[mask_valid]

                # Fill Bull (Green)
                apds.append(mpf.make_addplot(
                    fast_trail, panel=0, color='green', alpha=0,
                    fill_between=dict(y1=v_fast, y2=v_slow, where=where_bull, color='green', alpha=0.1),
                    secondary_y=False
                ))
                 # Fill Bear (Red)
                apds.append(mpf.make_addplot(
                    fast_trail, panel=0, color='red', alpha=0,
                    fill_between=dict(y1=v_fast, y2=v_slow, where=where_bear, color='red', alpha=0.1),
                    secondary_y=False
                ))

                # --- 1.4 Signals ---
                # Buy (1), Sell (-1)
                # Markers: Buy (Arrow Up, Green), Sell (Arrow Down, Red)

                # Create series with NaN everywhere except signal points
                # Position markers slightly below Low (Buy) or above High (Sell)

                buy_sig = plot_df['Low'].copy() * 0.95 # Shift down
                sell_sig = plot_df['High'].copy() * 1.05 # Shift up

                buy_sig[:] = np.nan
                sell_sig[:] = np.nan

                buy_mask = signals == 1
                sell_mask = signals == -1

                buy_sig[buy_mask] = plot_df['Low'][buy_mask] * 0.98
                sell_sig[sell_mask] = plot_df['High'][sell_mask] * 1.02

                if buy_mask.any():
                    apds.append(mpf.make_addplot(buy_sig, type='scatter', panel=0, markersize=100, marker='^', color='green'))
                if sell_mask.any():
                    apds.append(mpf.make_addplot(sell_sig, type='scatter', panel=0, markersize=100, marker='v', color='red'))


            except KeyError:
                print(f"Warning: {ticker} not found in ATR TS data.")

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

                bar_dead = create_bar_series(mask_dead)
                bar_lift = create_bar_series(mask_lift)
                bar_drift = create_bar_series(mask_drift)
                bar_power = create_bar_series(mask_power)

                apds.append(mpf.make_addplot(bar_dead, type='bar', panel=2, color='red', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(bar_lift, type='bar', panel=2, color='blue', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(bar_drift, type='bar', panel=2, color='yellow', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(bar_power, type='bar', panel=2, color='green', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))

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

                v_rs = rs_val.values
                v_ma = rs_ma.values
                v_zero = np.zeros_like(v_rs)
                mask_valid_rs = ~np.isnan(v_rs)
                mask_valid_ma = ~np.isnan(v_ma)

                where_rs_pos = np.zeros_like(v_rs, dtype=bool)
                where_rs_pos[mask_valid_rs] = v_rs[mask_valid_rs] > 0
                where_rs_neg = np.zeros_like(v_rs, dtype=bool)
                where_rs_neg[mask_valid_rs] = v_rs[mask_valid_rs] <= 0

                apds.append(mpf.make_addplot(
                    rs_val, panel=4, color='blue', alpha=0,
                    fill_between=dict(y1=v_rs, y2=v_zero, where=where_rs_pos, color='#0084ff', alpha=0.2),
                    secondary_y=False
                ))
                apds.append(mpf.make_addplot(
                    rs_val, panel=4, color='pink', alpha=0,
                    fill_between=dict(y1=v_rs, y2=v_zero, where=where_rs_neg, color='#ff52c8', alpha=0.2),
                    secondary_y=False
                ))

                mask_both = mask_valid_rs & mask_valid_ma
                where_rs_gt_ma = np.zeros_like(v_rs, dtype=bool)
                where_rs_gt_ma[mask_both] = v_rs[mask_both] > v_ma[mask_both]
                where_rs_lt_ma = np.zeros_like(v_rs, dtype=bool)
                where_rs_lt_ma[mask_both] = v_rs[mask_both] <= v_ma[mask_both]

                apds.append(mpf.make_addplot(
                    rs_val, panel=4, color='blue', alpha=0,
                    fill_between=dict(y1=v_rs, y2=v_ma, where=where_rs_gt_ma, color='#0084ff', alpha=0.2),
                    secondary_y=False
                ))
                apds.append(mpf.make_addplot(
                    rs_val, panel=4, color='pink', alpha=0,
                    fill_between=dict(y1=v_rs, y2=v_ma, where=where_rs_lt_ma, color='#ff52c8', alpha=0.2),
                    secondary_y=False
                ))

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

        # Plot Type: 'line' for main plot (invisible), but we add 'candle' addplots
        # Note: If we use type='line' with alpha=0, we lose volume bars on panel 1 unless we add them explicitly or let mpf handle volume.
        # But volume=True with type='line' usually works.
        # To make line invisible, we set line width to 0? Or color to clear?
        # mpf doesn't support 'none' type.
        # We'll use type='line', color='white' (or background color) or alpha=0?
        # A trick is using line plot with NaNs? No.
        # Let's try type='line', line_width=0.

        fig, axes = mpf.plot(
            plot_df,
            type='candle', # Main plot (covered by addplots)
            style=s,
            addplot=apds,
            volume=True,
            volume_panel=1,
            panel_ratios=(4, 1, 1, 1, 1, 1),
            returnfig=True,
            figsize=(12, 16),
            tight_layout=True,
            title=f"{ticker} Weekly Analysis",
            show_nontrading=False
        )

        fig.savefig(output_filename, bbox_inches='tight')
        print(f"Chart saved to {output_filename}")
        plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("-o", "--output", type=str, help="Output filename")
    args = parser.parse_args()

    generator = RDTChartGenerator()
    generator.generate_chart(args.ticker, args.output)
