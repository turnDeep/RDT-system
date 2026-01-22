import pandas as pd
from vcp_volume import analyze_vcp_volume
import contextlib
import io
import sys
import argparse

def check_volume_vcp(ticker, date_str):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            analyze_vcp_volume(ticker, target_date=date_str)
        except Exception as e:
            return False, str(e)

    output = f.getvalue()
    result = "Result: True" in output
    return result, output

def main():
    parser = argparse.ArgumentParser(description='Scan Volume VCP for a range.')
    parser.add_argument('ticker', help='Ticker symbol')
    parser.add_argument('start_date', help='Start Date (YYYY-MM-DD)')
    parser.add_argument('end_date', help='End Date (YYYY-MM-DD)')

    args = parser.parse_args()

    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date

    print(f"Scanning {ticker} from {start_date} to {end_date} for Volume VCP...")

    dates = pd.bdate_range(start=start_date, end=end_date)

    detected_dates = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        is_detected, output = check_volume_vcp(ticker, date_str)
        if is_detected:
            print(f"[FOUND] {date_str}")
            detected_dates.append(date_str)
        else:
            pass

    print(f"\nScan complete. Found {len(detected_dates)} days matching criteria.")
    if detected_dates:
        print("Dates:", detected_dates)

if __name__ == "__main__":
    main()
