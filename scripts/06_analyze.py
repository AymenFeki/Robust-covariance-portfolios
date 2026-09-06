#!/usr/bin/env python3
"""Run the full pipeline -- clean, estimate, backtest -- on ANY price panel.

Every other script in this repo is wired to one of the two vendored datasets
(the module-level ``PANEL = ...`` constant at the top of each). This one
takes a file path and a set of parameters instead, so the library can be
pointed at a dataset it has never seen.

    python scripts/06_analyze.py --prices my_prices.csv
    python scripts/06_analyze.py --prices crypto.csv --skip-cleaning \
        --periods-per-year 365
    python scripts/06_analyze.py --prices weekly_fx.csv --periods-per-year 52 \
        --window 52 --step 4
    python scripts/06_analyze.py --prices sp500.csv --estimator mp-clipped \
        --portfolio gmv -o out.csv

Input format: a CSV with ``Date``, ``Close`` and ``Name`` columns (long
format -- one row per date/ticker), or a wide panel with dates as the first
column and one column per asset. Both plain and gzipped files work.

IMPORTANT -- the cleaning defaults (``--threshold 0.35``, tuned to what a
daily large-cap US equity does) are NOT universal. They will misfire on
crypto (routine >35% daily moves are normal, not data errors) and on data
coarser or finer than daily. Read the printed cleaning report before trusting
it, retune ``--threshold``, or pass ``--skip-cleaning`` if your source is
already clean. See docs/METHODOLOGY.md and data/README.md for the reasoning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustcov import (  # noqa: E402
    clean_price_panel,
    condition_number,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_identity,
    load_price_panel,
    marchenko_pastur_clipped,
    pca_factor_model,
    sample_covariance,
    to_log_returns,
    walk_forward,
)
from robustcov.portfolios import (  # noqa: E402
    equal_risk_contribution,
    equal_weight,
    global_minimum_variance,
    inverse_variance,
    min_variance_long_only,
)

PORTFOLIO_CHOICES = {
    "gmv": global_minimum_variance,
    "long-only": min_variance_long_only,
    "erc": equal_risk_contribution,
    "inverse-var": inverse_variance,
    "equal-weight": equal_weight,
}

COLUMNS = [
    "ann_vol_pct", "ann_return_pct", "sharpe", "max_drawdown_pct",
    "avg_turnover", "avg_leverage", "avg_max_weight", "avg_short_pct",
]


def build_estimators(n_factors: int) -> dict:
    """Same five estimators as the rest of the repo, with a user-set k."""
    return {
        "Sample": lambda X: (sample_covariance(X), np.nan),
        "LW-identity": ledoit_wolf_identity,
        "LW-constcorr": ledoit_wolf_constant_correlation,
        "MP-clipped": marchenko_pastur_clipped,
        f"PCA-{n_factors}factor": lambda X: pca_factor_model(X, n_factors=n_factors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prices", type=Path, required=True, help="path to a price CSV"
    )
    parser.add_argument(
        "--allow-partial-history", action="store_true",
        help="keep tickers with gaps instead of requiring complete history",
    )
    parser.add_argument("--skip-cleaning", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--reversal-ratio", type=float, default=0.7)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument(
        "--periods-per-year", type=int, default=252,
        help="252 for daily, 52 for weekly, 12 for monthly, 365 for crypto",
    )
    parser.add_argument(
        "--estimator", choices=["all", "sample", "lw-identity", "lw-constcorr",
                                 "mp-clipped", "pca"], default="all",
    )
    parser.add_argument("--n-factors", type=int, default=5)
    parser.add_argument(
        "--portfolio", choices=list(PORTFOLIO_CHOICES), default="long-only",
        help="'gmv' is unconstrained and can short heavily -- see the main "
             "README for why 'long-only' is the safer default on unfamiliar data",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"loading {args.prices} ...")
    panel = load_price_panel(
        args.prices, require_full_history=not args.allow_partial_history
    )
    print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets, "
          f"{panel.index[0].date()} to {panel.index[-1].date()}")

    if args.skip_cleaning:
        print("--skip-cleaning set: using prices as-is")
    else:
        panel, report = clean_price_panel(
            panel, threshold=args.threshold, reversal_ratio=args.reversal_ratio
        )
        print(f"cleaning: {report}")
        for ticker, date, move in report.dropped_tickers[:10]:
            print(f"  dropped  {ticker:8s} {date}  first flagged move {move:+.2f}")
        if len(report.dropped_tickers) > 10:
            print(f"  ... and {len(report.dropped_tickers) - 10} more -- rerun with "
                  f"a higher --threshold if these look like real moves, not "
                  f"data errors")

    returns = to_log_returns(panel)
    n, p = returns.shape
    q = p / args.window
    print(f"\nreturns: {n} periods x {p} assets  |  q = p/window = {q:.3f}", end="")
    if q > 0.5:
        print("  <-- p is comparable to the window; sample covariance will be "
              "unreliable (see the main README) -- prefer LW-constcorr, "
              "MP-clipped or PCA here")
    else:
        print()

    if n <= args.window:
        print(f"\nERROR: only {n} return periods but --window {args.window}. "
              f"Reduce --window or provide more history.", file=sys.stderr)
        sys.exit(1)

    first_window = returns.iloc[: args.window].to_numpy()
    sample_cond = condition_number(sample_covariance(first_window))
    singular_note = "  (singular)" if not np.isfinite(sample_cond) else ""
    print(f"condition number of the raw sample covariance on the first "
          f"window: {sample_cond:.3e}{singular_note}")

    estimators = build_estimators(args.n_factors)
    if args.estimator != "all":
        key = {
            "sample": "Sample", "lw-identity": "LW-identity",
            "lw-constcorr": "LW-constcorr", "mp-clipped": "MP-clipped",
            "pca": f"PCA-{args.n_factors}factor",
        }[args.estimator]
        estimators = {key: estimators[key]}

    portfolio = PORTFOLIO_CHOICES[args.portfolio]
    print(f"\nportfolio rule: {args.portfolio}  |  window={args.window} "
          f"step={args.step}  cost={args.cost_bps}bps\n")

    rows = {}
    for name, estimator in estimators.items():
        result = walk_forward(returns, estimator, portfolio=portfolio,
                               window=args.window, step=args.step)
        rows[name] = result.summary(cost_bps=args.cost_bps,
                                     periods_per_year=args.periods_per_year)
    table = pd.DataFrame(rows).T
    print(table[COLUMNS].round(2).to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
