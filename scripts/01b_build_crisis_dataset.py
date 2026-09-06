#!/usr/bin/env python3
"""Vendor a second dataset spanning the 2008 financial crisis.

The main dataset (``01_build_dataset.py``) only covers August 2012 to August
2017 -- a calm bull market. That leaves an obvious question unanswered: does
any of this hold up during an actual crash? This script builds a second,
independent panel that does span one: 23 large, liquid blue-chip stocks with
complete daily history from January 2006 to December 2017, which covers the
NBER-dated 2007-12 to 2009-06 recession in full.

The tradeoff is universe size. Requiring genuinely complete history back to
2006 -- surviving IPOs, spinoffs, and data gaps -- leaves only 23 names
(several 2006-era constituents such as AAPL, GOOGL, AMZN and MSFT are
excluded here only because of short gaps elsewhere in their history, not
because of the crisis). See ``results/CRISIS_ROBUSTNESS.md`` for what that
means for interpreting the results: at p=23 this universe is far from the
p~n regime that makes the main study's estimators diverge, and that itself
turns out to be the first finding.

Usage
-----
    python scripts/01b_build_crisis_dataset.py [--raw path/to/all_stocks.csv]

Without --raw, clones the upstream mirror (network access to github.com).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustcov import clean_price_panel, load_price_panel  # noqa: E402

MIRROR = "https://github.com/mayur-tikar/DJIA-30-Stock-Time-Series-Dataset.git"
RAW_NAME = "all_stocks_2006-01-01_to_2018-01-01.csv"
OUTPUT = (
    Path(__file__).resolve().parents[1] / "data/processed/blue_chip_2006_2018.csv.gz"
)


def fetch_raw() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="crisis-"))
    print(f"cloning {MIRROR} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", MIRROR, str(tmp / "src")],
        check=True,
    )
    return tmp / "src" / RAW_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    raw = args.raw or fetch_raw()

    panel = load_price_panel(raw, require_full_history=True)
    print(f"balanced panel: {panel.shape[0]} dates x {panel.shape[1]} tickers")
    print(f"  {panel.index[0].date()} to {panel.index[-1].date()}")
    print(f"  tickers: {', '.join(sorted(panel.columns))}")

    cleaned, report = clean_price_panel(panel)
    print(f"cleaning: {report}")
    for ticker, date in report.bad_prints_fixed:
        print(f"  fixed print   {ticker:6s} {date}")
    for ticker, date, move in report.dropped_tickers:
        print(f"  dropped       {ticker:6s} {date}  first flagged move {move:+.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.round(6).to_csv(args.output, compression="gzip")
    size_kb = args.output.stat().st_size / 1e3
    print(f"\nwrote {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
