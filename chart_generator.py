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
        # We assume the calculate scripts have run and populated the PKLs.
        # But for a specific ticker argument, we might need to fetch fresh if not present?
        # The prompt implies "verify with HYMC", assuming data exists or we make it exist.
        # Let's assume the calculate scripts handle the universe. If HYMC isn't in stock.csv, it won't be in PKLs.
        # Ideally, we should fetch HYMC daily data first if missing, then recalc?
        # For simplicity in this tool context, we assume data is available or we rely on what's in the PKLs.

        # Load Weekly Indicators
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
                # Level 0: Close, High, etc. Level 1: Ticker
                # We want columns: Open, High, Low, Close, Volume
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

                # Colors based on state (Green if Fast > Slow, Red else)
                # Fill between is tricky with make_addplot directly, but we can plot lines.
                # Pine: fill(TS1, TS2, Bull ? green : red)
                # mpf.make_addplot has 'fill_between' argument in newer versions or use separate collection.
                # Simple approach: Plot lines.

                apds.append(mpf.make_addplot(fast_trail, panel=0, color='blue', width=1.0)) # Fast
                apds.append(mpf.make_addplot(slow_trail, panel=0, color='red', width=1.5)) # Slow

                # We can replicate "Bull" fill by filling where Fast > Slow
                # dict(y1=..., y2=..., where=..., color=...)
                # mpf currently supports fill_between as a dict or collection.
                # Let's stick to lines to avoid complex fill logic breakage, or try specific fill if robust.

            except KeyError:
                print(f"Warning: {ticker} not found in ATR TS data.")

        # 2. Zone RS (Panel 2)
        # Ratio (Blue), Momentum (Orange)
        if zone_rs_data:
            try:
                ratio = zone_rs_data["Ratio"][ticker].reindex(valid_idx)
                momentum = zone_rs_data["Momentum"][ticker].reindex(valid_idx)
                zones = zone_rs_data["Zone"][ticker].reindex(valid_idx)

                # Determine scale for fill
                y_min = min(ratio.min(), momentum.min())
                y_max = max(ratio.max(), momentum.max())
                # Add buffer
                y_min -= abs(y_min) * 0.1
                y_max += abs(y_max) * 0.1

                # Create constants for fill
                y1 = pd.Series(y_min, index=valid_idx)
                y2 = pd.Series(y_max, index=valid_idx)

                # Masks
                mask_dead = zones == 0
                mask_lift = zones == 1
                mask_drift = zones == 2
                mask_power = zones == 3

                # Background Fills (using fill_between logic via addplot lines + fill_between)
                # Note: mplfinance 0.12.7+ supports fill_between in make_addplot via `type='line'` and `fill_between` kwarg?
                # Actually, `fill_between` argument in make_addplot expects a dict or value.
                # Let's use the `fill_between` argument of `make_addplot` which accepts a dict.

                # Dead (Red)
                # We plot invisible lines and fill between them?
                # No, we can attach fill_between to the ratio plot? No, that fills to Y=0 or Y=y2.
                # We want to fill the whole panel based on X-axis condition.
                # Efficient way: Plot a constant line at y_min, and fill to y_max where condition met.

                # Background Fills using Type='bar' to avoid gaps
                # We calculate a full-height bar for each zone
                height = y_max - y_min

                def create_bar_series(mask):
                    s = pd.Series(np.nan, index=valid_idx)
                    s[mask] = height
                    return s

                bar_dead = create_bar_series(mask_dead)
                bar_lift = create_bar_series(mask_lift)
                bar_drift = create_bar_series(mask_drift)
                bar_power = create_bar_series(mask_power)

                # Plot bars filling the background
                # Width=1.0 ensures continuous fill
                # Alpha=0.15 slightly higher to see yellow
                apds.append(mpf.make_addplot(bar_dead, type='bar', panel=2, color='red', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(bar_lift, type='bar', panel=2, color='blue', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(bar_drift, type='bar', panel=2, color='yellow', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))
                apds.append(mpf.make_addplot(bar_power, type='bar', panel=2, color='green', alpha=0.15, width=1.0, bottom=y_min, secondary_y=False))

                apds.append(mpf.make_addplot(ratio, panel=2, color='blue', ylabel='Zone RS'))
                apds.append(mpf.make_addplot(momentum, panel=2, color='orange', secondary_y=False))

            except KeyError:
                pass

        # 3. Historical Percentile (Panel 3)
        # 1M Mode Histogram
        if rs_perc_data:
            try:
                # Use 1M mode
                perc = rs_perc_data["Percentile_1M"][ticker].reindex(valid_idx)

                # Color coding (Pine logic: <10, <30, <50, <70, <85, <95, >95)
                # Creating a list of colors
                colors = []
                for v in perc:
                    if pd.isna(v): colors.append('white')
                    elif v < 10: colors.append('#d86eef') # Very Low (Purple)
                    elif v < 30: colors.append('#d3eeff') # Low (Light Blue)
                    elif v < 50: colors.append('#4e7eff') # Mod Low
                    elif v < 70: colors.append('#96d7ff') # Mod
                    elif v < 85: colors.append('#80cfff') # Mod High
                    elif v < 95: colors.append('#1eaaff') # High
                    else: colors.append('#30b0ff') # Very High

                apds.append(mpf.make_addplot(perc, type='bar', panel=3, color=colors, ylabel='Hist %'))

            except KeyError:
                pass

        # 4. Volatility Adjusted RS (Panel 4)
        if rs_vol_data:
            try:
                rs_val = rs_vol_data["RS_Values"][ticker].reindex(valid_idx)
                rs_ma = rs_vol_data["RS_MA"][ticker].reindex(valid_idx)

                # --- RS Line (Blue if > 0, Fuchsia if < 0) ---
                # To ensure continuous line without gaps, we plot the full line in Fuchsia (Base)
                # and overlay the Positive part in Blue.
                apds.append(mpf.make_addplot(rs_val, panel=4, color='#ff52c8', ylabel='Vol Adj RS', width=1.5)) # Base (Fuchsia)

                rs_pos = rs_val.copy()
                rs_pos[rs_pos <= 0] = np.nan
                if not rs_pos.isna().all():
                    apds.append(mpf.make_addplot(rs_pos, panel=4, color='#0084ff', width=1.5)) # Overlay (Blue)

                # --- MA Line (Blue if Rising, Fuchsia if Falling) ---
                if not rs_ma.isna().all():
                    ma_diff = rs_ma.diff()

                    # Base Line: Fuchsia (Falling/Default)
                    apds.append(mpf.make_addplot(rs_ma, panel=4, color='#ff52c8', width=1.0))

                    # Overlay Rising: Blue
                    # We want to color segments where slope is positive.
                    # ma_diff > 0.
                    ma_rising = rs_ma.copy()
                    # Mask where NOT rising.
                    # Note: We keep points where diff > 0.
                    ma_rising[ma_diff <= 0] = np.nan

                    if not ma_rising.isna().all():
                        apds.append(mpf.make_addplot(ma_rising, panel=4, color='#0084ff', width=1.0))

                # --- Fills ---
                # Anchor: Zero Line (Invisible)
                zero_line = pd.Series(0, index=valid_idx)

                # Fill 1: RS vs 0
                # Blue where RS > 0
                apds.append(mpf.make_addplot(zero_line, panel=4, color='white', alpha=0, secondary_y=False,
                                            fill_between=dict(y1=rs_val.values, y2=0, where=rs_val.values > 0, color='#0084ff', alpha=0.2)))
                # Fuchsia where RS < 0
                apds.append(mpf.make_addplot(zero_line, panel=4, color='white', alpha=0, secondary_y=False,
                                            fill_between=dict(y1=rs_val.values, y2=0, where=rs_val.values < 0, color='#ff52c8', alpha=0.2)))

                # Fill 2: RS vs MA
                # Blue where RS > MA
                apds.append(mpf.make_addplot(zero_line, panel=4, color='white', alpha=0, secondary_y=False,
                                            fill_between=dict(y1=rs_val.values, y2=rs_ma.values, where=rs_val.values > rs_ma.values, color='#0084ff', alpha=0.2)))
                # Fuchsia where RS < MA
                apds.append(mpf.make_addplot(zero_line, panel=4, color='white', alpha=0, secondary_y=False,
                                            fill_between=dict(y1=rs_val.values, y2=rs_ma.values, where=rs_val.values < rs_ma.values, color='#ff52c8', alpha=0.2)))

            except KeyError:
                pass

        # 5. RTI (Panel 5)
        if rti_data:
            try:
                rti_val = rti_data["RTI_Values"][ticker].reindex(valid_idx)
                rti_sig = rti_data["RTI_Signals"][ticker].reindex(valid_idx)

                # Line Color: Green if Expansion (Sig=3), Blue otherwise
                # Split into two lines for coloring
                rti_exp = rti_val.copy()
                rti_norm = rti_val.copy()

                # Masking is tricky for connected lines.
                # Simpler: Plot base blue line, and overlay green line where sig==3.
                # But green line needs to connect?
                # Let's plot main line as blue, and markers for expansion? Or just simple blue line for now to avoid crash.
                # Or use scatter for expansion points?
                # User asked for line color.
                # Workaround: Scatter for specific points or just single color.
                # Let's use single color 'blue' for the line, and add markers for expansion.

                apds.append(mpf.make_addplot(rti_val, panel=5, color='blue', ylabel='RTI', width=2.0))

                # Expansion Markers (Green)
                exp_dots = rti_val.copy()
                exp_dots[:] = np.nan
                mask_exp = (rti_sig == 3)
                exp_dots[mask_exp] = rti_val[mask_exp]

                # Ensure we have data before plotting
                if mask_exp.any() and not exp_dots.isna().all():
                     apds.append(mpf.make_addplot(exp_dots, panel=5, type='scatter', markersize=30, color='green', marker='^'))

                # Dots: Orange if Sig=2 (Consecutive Tight)
                dots = rti_val.copy()
                dots[:] = np.nan
                mask_dot = (rti_sig == 2)
                dots[mask_dot] = rti_val[mask_dot]

                if not dots.isna().all():
                    apds.append(mpf.make_addplot(dots, panel=5, type='scatter', markersize=20, color='orange'))

                # Zones (0-5 Red, 5-20 Green)
                # Hard to fill specific y-ranges in panels in mpf easily without complex returnfig hacking.
                # We will draw lines at 5, 20, 100.

            except KeyError:
                pass

        # 6. Plotting
        if output_filename is None:
            output_filename = f"{ticker}_weekly_chart.png"

        # Style
        # Custom style to match aesthetics
        # Panels: 0=Main, 1=Vol, 2=Zone, 3=Perc, 4=VolAdj, 5=RTI
        # Ratios: Main=3, others=1

        rc_params = {'axes.grid': True, 'grid.linestyle': ':'}
        s = mpf.make_mpf_style(base_mpf_style='yahoo', rc=rc_params)

        fig, axes = mpf.plot(
            plot_df,
            type='candle',
            style=s,
            addplot=apds,
            volume=True,
            volume_panel=1,
            panel_ratios=(4, 1, 1, 1, 1, 1),
            returnfig=True,
            figsize=(12, 16),
            tight_layout=True,
            title=f"{ticker} Weekly Analysis"
        )

        # Save
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
