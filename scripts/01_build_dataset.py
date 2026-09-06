#!/usr/bin/env python3
"""Rebuild the vendored price panel from the upstream raw dataset.

The repository ships ``data/processed/sp500_prices.csv.gz`` already, so you
do not need to run this to reproduce the results. It exists so the vendored
file is not a black box: run it and you get the same panel back.

Usage
-----
    python scripts/01_build_dataset.py --raw path/to/all_stocks_5yr.csv

If ``--raw`` is omitted the script clones the upstream mirror into a
temporary directory (requires network access to github.com).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustcov import clean_price_panel, load_price_panel  # noqa: E402

MIRROR = "https://github.com/orest-d/sp500exp.git"
RAW_NAME = "all_stocks_5yr.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data/processed/sp500_prices.csv.gz"


def fetch_raw() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="sp500-"))
    print(f"cloning {MIRROR} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", MIRROR, str(tmp / "src")],
        check=True,
    )
    return tmp / "src" / RAW_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=None, help="path to the raw CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    raw = args.raw or fetch_raw()

    panel = load_price_panel(raw, require_full_history=True)
    print(f"balanced panel: {panel.shape[0]} dates x {panel.shape[1]} tickers")
    print(f"  {panel.index[0].date()} to {panel.index[-1].date()}")

    repaired, report = clean_price_panel(panel)
    print(f"cleaning: {report}")
    for ticker, date in report.bad_prints_fixed:
        print(f"  fixed print   {ticker:6s} {date}")
    for ticker, date, move in report.dropped_tickers:
        print(f"  dropped       {ticker:6s} {date}  first flagged move {move:+.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    repaired.round(6).to_csv(args.output, compression="gzip")
    size_mb = args.output.stat().st_size / 1e6
    print(f"\nwrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
