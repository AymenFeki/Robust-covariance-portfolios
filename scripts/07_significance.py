#!/usr/bin/env python3
"""How much of the headline comparison is signal, and how much is noise?

    python scripts/07_significance.py [--window 250] [--step 21] [--n-boot 2000]

Every number in ``results/RESULTS.md`` is a point estimate from one realised
five-year path of the market. This script adds what that table is missing:

1. **Bootstrap confidence intervals** on annualised volatility and Sharpe
   ratio for each estimator, via a circular block bootstrap of the realised
   out-of-sample return series (``robustcov.inference.block_bootstrap_ci``).
   This answers "how much would this number move if history had gone a
   little differently", holding the estimation procedure fixed.
2. **The deflated Sharpe ratio** (Bailey & Lopez de Prado, 2014) for the
   best-performing estimator, correcting for exactly the selection bias this
   project's own README flags in its "Threats to validity" table: five
   estimators were compared on one dataset, so the best one's Sharpe ratio
   is inflated simply by virtue of being the best of five noisy draws, even
   before asking whether it is individually significant.

Writes ``results/significance_<window>d.csv`` for each portfolio rule.
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
    deflated_sharpe_ratio_from_returns,
    global_minimum_variance,
    load_price_panel,
    min_variance_long_only,
    sharpe_ratio,
    sharpe_ratio_ci,
    to_log_returns,
    walk_forward,
)
from robustcov.inference import annualized_vol_ci  # noqa: E402

PANEL = ROOT / "data/processed/sp500_prices.csv.gz"


def significance_table(
    returns: pd.DataFrame,
    portfolio,
    window: int,
    step: int,
    n_boot: int,
) -> pd.DataFrame:
    """Bootstrap CIs for every estimator, plus the deflated Sharpe ratio for
    whichever one comes out on top."""
    rows = {}
    period_returns = {}
    for name, estimator in ESTIMATORS.items():
        result = walk_forward(
            returns, estimator, portfolio=portfolio, window=window, step=step
        )
        r = result.returns.to_numpy()
        period_returns[name] = r

        sharpe_pt, sharpe_lo, sharpe_hi = sharpe_ratio_ci(r, n_boot=n_boot, seed=0)
        vol_pt, vol_lo, vol_hi = annualized_vol_ci(r, n_boot=n_boot, seed=0)
        rows[name] = {
            "sharpe": sharpe_pt,
            "sharpe_ci_low": sharpe_lo,
            "sharpe_ci_high": sharpe_hi,
            "ann_vol_pct": vol_pt,
            "ann_vol_pct_ci_low": vol_lo,
            "ann_vol_pct_ci_high": vol_hi,
        }
    table = pd.DataFrame(rows).T

    best_name = table["sharpe"].idxmax()
    all_trial_sharpes = np.array(
        [sharpe_ratio(r, periods_per_year=1) for r in period_returns.values()]
    )
    dsr = deflated_sharpe_ratio_from_returns(
        period_returns[best_name],
        n_trials=len(ESTIMATORS),
        all_trial_sharpes=all_trial_sharpes,
        periods_per_year=252,
    )
    table["deflated_sharpe_ratio"] = np.nan
    table.loc[best_name, "deflated_sharpe_ratio"] = dsr["deflated_sharpe_ratio"]
    return table, best_name, dsr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--n-boot", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices = load_price_panel(PANEL, require_full_history=False)
    returns = to_log_returns(prices)
    n_obs, n_assets = returns.shape
    print(f"panel: {n_obs} days x {n_assets} assets, window={args.window} "
          f"step={args.step} n_boot={args.n_boot}\n")

    for label, portfolio in [
        ("Unconstrained GMV", global_minimum_variance),
        ("Long-only minimum variance", min_variance_long_only),
    ]:
        table, best_name, dsr = significance_table(
            returns, portfolio, args.window, args.step, args.n_boot
        )
        print(f"=== {label}: bootstrap 95% CIs "
              f"(block bootstrap, {args.n_boot} resamples) ===")
        print(table[["sharpe", "sharpe_ci_low", "sharpe_ci_high",
                      "ann_vol_pct", "ann_vol_pct_ci_low", "ann_vol_pct_ci_high"]]
              .round(3).to_string())
        print(f"\nBest by Sharpe ratio: {best_name} "
              f"(SR={dsr['sharpe_annualized']:.3f})")
        benchmark_ann = dsr["benchmark_sharpe_per_period"] * np.sqrt(252)
        print(f"  benchmark Sharpe expected from the best of {dsr['n_trials']} "
              f"noisy trials by chance: {benchmark_ann:.3f} (annualised)")
        print(f"  return skew={dsr['skew']:.3f}  kurtosis={dsr['kurtosis']:.3f}")
        print(f"  deflated Sharpe ratio: {dsr['deflated_sharpe_ratio']:.3f}")
        if dsr["deflated_sharpe_ratio"] >= 0.95:
            verdict = "likely a real edge, not just the best of five noisy draws"
        elif dsr["deflated_sharpe_ratio"] >= 0.5:
            verdict = ("directionally positive but not distinguishable from "
                       "chance at conventional confidence")
        else:
            verdict = "no better than the best of five random strategies by chance"
        print(f"  verdict: {verdict}\n")

        fname = f"results/significance_{label.split()[0].lower()}_{args.window}d.csv"
        out = ROOT / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out)
        print(f"wrote {out}\n")


if __name__ == "__main__":
    main()
