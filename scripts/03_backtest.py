#!/usr/bin/env python3
"""Run the walk-forward comparison and write the results tables.

    python scripts/03_backtest.py [--window 250] [--step 21] [--cost-bps 10]

Writes ``results/backtest_<window>d.csv`` and a markdown summary to
``results/RESULTS.md``.
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
    ESTIMATORS,
    condition_number,
    equal_weight,
    global_minimum_variance,
    load_price_panel,
    min_variance_long_only,
    summarise,
    to_log_returns,
    walk_forward,
)

PANEL = ROOT / "data/processed/sp500_prices.csv.gz"

COLUMNS = [
    "ann_vol_pct",
    "ann_return_pct",
    "sharpe",
    "max_drawdown_pct",
    "avg_turnover",
    "avg_leverage",
    "avg_max_weight",
    "avg_short_pct",
]


def benchmark_row(returns: pd.DataFrame, window: int, step: int) -> dict[str, float]:
    """1/N held on the same rebalance schedule, as a reference point."""
    result = walk_forward(
        returns,
        estimator=lambda X: (np.cov(X, rowvar=False), np.nan),
        portfolio=equal_weight,
        window=window,
        step=step,
    )
    return result.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    args = parser.parse_args()

    prices = load_price_panel(PANEL, require_full_history=False)
    returns = to_log_returns(prices)
    n_obs, n_assets = returns.shape
    print(f"panel: {n_obs} days x {n_assets} assets")
    print(f"window {args.window}d -> p/n = {n_assets / args.window:.2f}\n")

    for label, portfolio in [
        ("Unconstrained GMV", global_minimum_variance),
        ("Long-only minimum variance", min_variance_long_only),
    ]:
        table = summarise(
            returns,
            portfolio=portfolio,
            estimators=ESTIMATORS,
            window=args.window,
            step=args.step,
            cost_bps=args.cost_bps,
        )
        table.loc["EqualWeight (1/N)"] = benchmark_row(returns, args.window, args.step)
        print(f"=== {label} ===")
        print(table[COLUMNS].round(2).to_string(), "\n")

        out = ROOT / f"results/{label.split()[0].lower()}_{args.window}d.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out)

    print("=== conditioning of each estimate (first window) ===")
    first = returns.iloc[: args.window].to_numpy()
    for name, estimator in ESTIMATORS.items():
        sigma, diagnostic = estimator(first)
        eig = np.linalg.eigvalsh(sigma)
        print(
            f"  {name:14s} cond {condition_number(sigma):.3e}"
            f"  min eig {eig.min():+.2e}  diagnostic {diagnostic:.3f}"
        )
    rank = np.linalg.matrix_rank(np.cov(first, rowvar=False))
    print(f"\n  sample covariance rank {rank} of {n_assets} -> "
          f"{'singular' if rank < n_assets else 'full rank'}")


if __name__ == "__main__":
    main()
