import argparse
import sys
from pathlib import Path

import pandas as pd

from data_sweep.clean import clean
from data_sweep.profile import profile
from data_sweep.report import report


def main():
    parser = argparse.ArgumentParser(
        prog="data-sweep",
        description="Scan a CSV, fix basic data quality issues, and explain what changed.",
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.5,
        help="Drop a column if the fraction of missing values exceeds this (default: 0.5).",
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    findings = profile(df, args.missing_threshold)
    cleaned_df = clean(df, args.missing_threshold)
    report_text = report(findings, df.shape, cleaned_df.shape)

    stem = Path(args.input_csv).stem
    cleaned_path = f"cleaned_{stem}.csv"
    report_path = f"report_{stem}.md"

    cleaned_df.to_csv(cleaned_path, index=False)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Scanned {args.input_csv}: {len(findings)} issue(s) found.")
    print(f"Wrote {cleaned_path} and {report_path}.")


if __name__ == "__main__":
    main()
