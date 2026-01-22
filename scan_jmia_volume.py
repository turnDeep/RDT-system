import pandas as pd
from vcp_volume import analyze_vcp_volume
import contextlib
import io
import sys

# Define a function that returns boolean instead of printing,
# or capture the output to parse result.
# Since I can't easily change the return type of the imported function without editing it significantly,
# I will capture stdout.

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
    ticker = "JMIA"
    start_date = "2025-07-01"
    end_date = "2025-10-31"

    print(f"Scanning {ticker} from {start_date} to {end_date} for Volume VCP...")

    # Generate business days
    dates = pd.bdate_range(start=start_date, end=end_date)

    detected_dates = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        is_detected, output = check_volume_vcp(ticker, date_str)
        if is_detected:
            print(f"[FOUND] {date_str}")
            detected_dates.append(date_str)
        else:
            # print(f"[...] {date_str}", end='\r')
            pass

    print(f"\nScan complete. Found {len(detected_dates)} days matching criteria.")
    if detected_dates:
        print("Dates:", detected_dates)

if __name__ == "__main__":
    main()
