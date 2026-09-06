#!/usr/bin/env python3
"""Sweep the estimation window and plot how each estimator degrades.

Varying the window length varies ``q = p / n`` -- the single parameter that
governs how noisy the sample covariance is. This is the project's headline
result: the sample estimator's out-of-sample risk peaks where ``q`` is near
1, exactly where random matrix theory says the smallest eigenvalues collapse
toward zero, while every regularised estimator stays flat.

    python scripts/04_window_sweep.py

Writes ``results/window_sweep.csv`` and ``results/figures/window_sweep.png``.
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

from _style import (  # noqa: E402
    INK_SECONDARY,
    MUTED,
    SERIES,
    apply_style,
    declutter,
)
from robustcov import (  # noqa: E402
    ESTIMATORS,
    load_price_panel,
    sample_covariance,
    to_log_returns,
    walk_forward,
)
from robustcov.backtest import walk_forward as _wf  # noqa: E402

PANEL = ROOT / "data/processed/sp500_prices.csv.gz"
WINDOWS = [150, 250, 375, 500, 750]

#: Fixed slot order -- never cycled, never reassigned when a series is hidden.
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
    """``inv(S) 1 / (1' inv(S) 1)`` with no numerical safeguards at all.

    What you get if you code the formula straight out of the textbook. When
    ``S`` is singular this is meaningless, and the point of including it is
    to show exactly how meaningless.
    """
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


def run() -> pd.DataFrame:
    returns = to_log_returns(load_price_panel(PANEL, require_full_history=False))
    n_assets = returns.shape[1]
    rows = []

    for window in WINDOWS:
        for name, estimator in ESTIMATORS.items():
            label = "Sample (pseudo-inverse)" if name == "Sample" else name
            summary = walk_forward(returns, estimator, window=window).summary()
            rows.append({"window": window, "estimator": label, **summary})

        summary = _wf(
            returns,
            lambda X: (sample_covariance(X), np.nan),
            portfolio=textbook_gmv,
            window=window,
        ).summary()
        rows.append(
            {"window": window, "estimator": "Sample (textbook inverse)", **summary}
        )

    frame = pd.DataFrame(rows)
    frame["q"] = n_assets / frame["window"]
    frame["n_assets"] = n_assets
    return frame


def plot(frame: pd.DataFrame) -> Path:
    """Two linear panels comparing the usable estimators.

    The textbook-inverse case is deliberately kept off the axes: at 460-546%
    volatility it would compress every real difference into a single line at
    the bottom of a log scale. It is reported as an annotation instead, which
    is the honest way to show an outlier that is three orders of magnitude
    away from everything else.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))

    plotted = ORDER[1:]  # everything except the textbook inverse
    blown = frame[frame.estimator == ORDER[0]]

    panels = [
        (axes[0], "ann_vol_pct", "Out-of-sample annualised volatility (%)",
         "Realised risk of the resulting portfolio"),
        (axes[1], "avg_leverage", "Average gross leverage, $\\sum|w_i|$",
         "How large the offsetting positions get"),
    ]

    for ax, column, ylabel, title in panels:
        ends = {}
        for name in plotted:
            sub = frame[frame.estimator == name].sort_values("window")
            ax.plot(
                sub["window"],
                sub[column],
                color=COLOURS[name],
                marker="o",
                markeredgecolor="#fcfcfb",
                markeredgewidth=0.8,
                label=name,
            )
            ends[name] = float(sub[column].iloc[-1])

        lo = min(frame[frame.estimator.isin(plotted)][column]) * 0.92
        hi = max(frame[frame.estimator.isin(plotted)][column]) * 1.06
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Estimation window (trading days)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(WINDOWS)
        ax.set_xlim(WINDOWS[0] - 25, WINDOWS[-1] + 285)

        names = list(ends)
        positions = declutter(
            np.array([ends[n] for n in names]), min_gap=(hi - lo) * 0.062
        )
        for name, y in zip(names, positions, strict=True):
            ax.text(
                WINDOWS[-1] + 22, y, name,
                color=COLOURS[name], fontsize=8, va="center", ha="left",
            )

        worst = blown[column].max()
        unit = "%" if column == "ann_vol_pct" else "x"
        ax.annotate(
            f"textbook $S^{{-1}}$ peaks at {worst:,.0f}{unit},\n"
            f"off the top of this axis",
            xy=(WINDOWS[0] + 8, hi * 0.985),
            color=MUTED, fontsize=8, va="top", ha="left",
        )

    n_assets = int(frame["n_assets"].iloc[0])
    for ax in axes:
        low, high = ax.get_ylim()
        ax.axvline(n_assets, color=MUTED, linewidth=1.0, linestyle=":")
        ax.text(
            n_assets + 12,
            low + (high - low) * 0.03,
            "$p = n$",
            color=INK_SECONDARY, fontsize=8, va="bottom", ha="left",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.012), columnspacing=1.8,
    )
    fig.suptitle(
        "The sample covariance fails worst where the number of assets meets the "
        "number of observations",
        fontsize=11, y=0.985,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))

    out = ROOT / "results/figures/window_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def main() -> None:
    frame = run()
    table = frame.pivot(index="estimator", columns="window", values="ann_vol_pct")
    print("Out-of-sample annualised volatility (%), by estimation window")
    print(table.reindex(ORDER).round(2).to_string(), "\n")
    print("Average gross leverage")
    print(
        frame.pivot(index="estimator", columns="window", values="avg_leverage")
        .reindex(ORDER)
        .round(2)
        .to_string()
    )

    csv = ROOT / "results/window_sweep.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv, index=False)
    print(f"\nwrote {csv}")
    print(f"wrote {plot(frame)}")


if __name__ == "__main__":
    main()
