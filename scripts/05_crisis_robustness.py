#!/usr/bin/env python3
"""Robustness check: a second dataset, spanning the 2008 crisis.

Answers two distinct questions on 23 blue-chip stocks, January 2006 to
December 2017 (data/processed/blue_chip_2006_2018.csv.gz):

1. At a realistic ~1-year estimation window, does the estimator ranking from
   the main study survive an actual crash (NBER recession: Dec 2007 to
   Jun 2009), or was the main result an artefact of one calm bull market?

2. Independently of (1): does the phase transition from
   ``04_window_sweep.py`` -- the sample estimator's blow-up exactly where
   p/n = 1 -- reproduce on a completely different dataset, universe size,
   and market regime? This dataset has p=23 instead of 428, so the window
   that puts q=1 is 23 trading days instead of 428.

    python scripts/05_crisis_robustness.py

Writes ``results/figures/crisis_phase_transition.png`` and
``results/crisis_summary.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import INK_SECONDARY, MUTED, SERIES, apply_style, declutter  # noqa: E402
from robustcov import (  # noqa: E402
    ESTIMATORS,
    load_price_panel,
    sample_covariance,
    to_log_returns,
    walk_forward,
)
from robustcov.portfolios import global_minimum_variance  # noqa: E402

PANEL = ROOT / "data/processed/blue_chip_2006_2018.csv.gz"
CRISIS_START, CRISIS_END = "2007-12-01", "2009-06-30"  # NBER-dated recession
SWEEP_WINDOWS = [15, 18, 21, 23, 26, 30, 40, 60, 120, 250]

ORDER = [
    "Sample (textbook inverse)",
    "Sample (pseudo-inverse)",
    "LW-identity",
    "LW-constcorr",
    "MP-clipped",
    "PCA-5factor",
]
COLOURS = dict(zip(ORDER, SERIES, strict=True))


def textbook_gmv(sigma: np.ndarray) -> np.ndarray:
    """Same uninstrumented ``inv(S)`` formula as in 04_window_sweep.py."""
    ones = np.ones(len(sigma))
    with np.errstate(all="ignore"):
        try:
            raw = np.linalg.inv(sigma) @ ones
        except np.linalg.LinAlgError:
            return np.full(len(sigma), np.nan)
    total = ones @ raw
    if not np.isfinite(total) or abs(total) < 1e-300:
        return np.full(len(sigma), np.nan)
    return raw / total


def question_1_does_the_ranking_survive(returns: pd.DataFrame) -> pd.DataFrame:
    """Standard 250-day/21-day walk-forward, split into crisis vs calm."""
    rows = {}
    for name, estimator in ESTIMATORS.items():
        result = walk_forward(
            returns, estimator, portfolio=global_minimum_variance, window=250, step=21
        )
        crisis = (result.returns.index >= CRISIS_START) & (
            result.returns.index <= CRISIS_END
        )
        rows[name] = {
            "full_period_vol_pct": result.returns.std() * np.sqrt(252) * 100,
            "crisis_vol_pct": result.returns[crisis].std() * np.sqrt(252) * 100,
            "calm_vol_pct": result.returns[~crisis].std() * np.sqrt(252) * 100,
            "avg_leverage": result.weights.abs().sum(axis=1).mean(),
        }
    return pd.DataFrame(rows).T


def question_2_does_the_phase_transition_replicate(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Same sweep as 04_window_sweep.py, on a dataset where p=23 makes the
    q=1 crossover fall at a 23-day window instead of a 428-day one."""
    n_assets = returns.shape[1]
    rows = []
    for window in SWEEP_WINDOWS:
        for name, estimator in ESTIMATORS.items():
            label = "Sample (pseudo-inverse)" if name == "Sample" else name
            summary = walk_forward(returns, estimator, window=window, step=10).summary()
            rows.append({"window": window, "estimator": label, **summary})
        summary = walk_forward(
            returns,
            lambda X: (sample_covariance(X), np.nan),
            portfolio=textbook_gmv,
            window=window,
            step=10,
        ).summary()
        rows.append({"window": window, "estimator": ORDER[0], **summary})
    frame = pd.DataFrame(rows)
    frame["q"] = n_assets / frame["window"]
    frame["n_assets"] = n_assets
    return frame


def plot(frame: pd.DataFrame) -> Path:
    """Linear axis over the regularised estimators only.

    Both sample-based variants blow up by one to two orders of magnitude
    right at q=1 (2,315% and 839% respectively -- see the annotation) and
    would otherwise crush every other line flat at the bottom of the axis,
    exactly the problem the main window-sweep figure already had to solve.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    plotted = ORDER[2:]  # exclude both sample variants from the drawn lines
    blown = frame[frame.estimator.isin(ORDER[:2])]

    ends = {}
    for name in plotted:
        sub = frame[frame.estimator == name].sort_values("window")
        ax.plot(
            sub["window"], sub["ann_vol_pct"], color=COLOURS[name],
            marker="o", markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=name,
        )
        ends[name] = float(sub["ann_vol_pct"].iloc[-1])

    lo = min(frame[frame.estimator.isin(plotted)]["ann_vol_pct"]) * 0.97
    hi = max(frame[frame.estimator.isin(plotted)]["ann_vol_pct"]) * 1.05
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_xticks(SWEEP_WINDOWS)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(SWEEP_WINDOWS[0] * 0.85, SWEEP_WINDOWS[-1] * 1.9)
    ax.set_xlabel("Estimation window (trading days)")
    ax.set_ylabel("Out-of-sample annualised volatility (%)")

    n_assets = int(frame["n_assets"].iloc[0])
    ax.axvline(n_assets, color=MUTED, linewidth=1.0, linestyle=":")
    ax.text(
        n_assets * 1.06, lo + (hi - lo) * 0.04, f"$p = n = {n_assets}$",
        color=INK_SECONDARY, fontsize=8.5, va="bottom", ha="left",
    )

    names = list(ends)
    positions = declutter(np.array([ends[n] for n in names]), min_gap=(hi - lo) * 0.05)
    for name, y in zip(names, positions, strict=True):
        ax.text(
            SWEEP_WINDOWS[-1] * 1.15, y, name,
            color=COLOURS[name], fontsize=8.5, va="center", ha="left",
        )

    lines = []
    for label in ORDER[:2]:
        rows = blown[blown.estimator == label]
        peak = rows.loc[rows["ann_vol_pct"].idxmax()]
        lines.append(
            f"{label} peaks at {peak['ann_vol_pct']:,.0f}% "
            f"(window={int(peak['window'])}, q={peak['q']:.2f})"
        )
    ax.annotate(
        "off the top of this axis:\n" + "\n".join(lines),
        xy=(0.03, 0.97), xycoords="axes fraction",
        color=MUTED, fontsize=8.5, va="top", ha="left",
    )

    fig.suptitle(
        "The same p=n blow-up, replicated on a different dataset,\n"
        "different universe size, and a real financial crisis",
        fontsize=11.5, y=0.99,
    )
    ax.set_title(
        "23 blue-chip stocks, 2006-2017 (spans the 2008 crisis) -- vs. "
        "428 S&P 500 names, 2012-2017, in the main study",
        fontsize=9.5, color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    fig.legend(
        *ax.get_legend_handles_labels(), loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.0), columnspacing=1.6,
    )

    out = ROOT / "results/figures/crisis_phase_transition.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def main() -> None:
    returns = to_log_returns(load_price_panel(PANEL, require_full_history=False))
    n_assets = returns.shape[1]
    print(f"panel: {returns.shape[0]} days x {n_assets} assets, "
          f"{returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"crisis window: {CRISIS_START} to {CRISIS_END} (NBER-dated recession)\n")

    print("=== Q1: does the estimator ranking survive the 2008 crisis? ===")
    print("(250-day window -> q = p/n = "
          f"{n_assets/250:.3f}, far below the p~n danger zone)\n")
    q1 = question_1_does_the_ranking_survive(returns)
    print(q1.round(2).to_string(), "\n")

    print("=== Q2: does the phase transition replicate at q=1 on this dataset? ===")
    q2 = question_2_does_the_phase_transition_replicate(returns)
    pivot = q2.pivot(index="estimator", columns="window", values="ann_vol_pct")
    print(pivot.reindex(ORDER).round(2).to_string())

    out_csv = ROOT / "results/crisis_summary.csv"
    q1.to_csv(out_csv)
    print(f"\nwrote {out_csv}")
    print(f"wrote {plot(q2)}")


if __name__ == "__main__":
    main()
