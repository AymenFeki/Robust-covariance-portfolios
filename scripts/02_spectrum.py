#!/usr/bin/env python3
"""Compare the empirical eigenvalue spectrum with the Marchenko-Pastur law.

This is the diagnostic that motivates everything else in the project: it
shows directly how much of the sample correlation matrix is noise.

    python scripts/02_spectrum.py [--window 250]

Writes ``results/figures/spectrum.png`` and prints the numbers behind it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import (  # noqa: E402
    CRITICAL,
    INK_SECONDARY,
    MUTED,
    SEQUENTIAL,
    SURFACE,
    apply_style,
)
from robustcov import (  # noqa: E402
    load_price_panel,
    marchenko_pastur_density,
    to_log_returns,
)

PANEL = ROOT / "data/processed/sp500_prices.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=250)
    args = parser.parse_args()

    returns = to_log_returns(load_price_panel(PANEL, require_full_history=False))
    window = returns.iloc[: args.window].to_numpy()
    n, p = window.shape
    q = p / n

    centred = window - window.mean(axis=0)
    standardised = centred / centred.std(axis=0, ddof=1)
    correlation = standardised.T @ standardised / n
    eigenvalues = np.linalg.eigvalsh(correlation)

    lower = (1 - np.sqrt(q)) ** 2
    upper = (1 + np.sqrt(q)) ** 2

    n_zero = int((eigenvalues < 1e-10).sum())
    expected_zero = p * (1 - 1 / q) if q > 1 else 0.0
    n_signal = int((eigenvalues > upper).sum())
    variance_in_signal = eigenvalues[eigenvalues > upper].sum() / eigenvalues.sum()

    print(f"window {n} days, {p} assets, q = p/n = {q:.3f}")
    print(f"MP noise band: [{lower:.3f}, {upper:.3f}]")
    print(f"zero eigenvalues: {n_zero} observed vs {expected_zero:.1f} predicted "
          f"by the MP atom at 0 (mass 1 - 1/q)")
    print(f"eigenvalues above the upper edge: {n_signal} of {p} "
          f"({n_signal / p:.1%} of directions)")
    print(f"share of total variance in those {n_signal}: {variance_in_signal:.1%}")
    print(f"largest eigenvalue: {eigenvalues.max():.1f} "
          f"({eigenvalues.max() / p:.1%} of total variance -- the market factor)")

    # ---------------------------------------------------------------- plot
    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    nonzero = eigenvalues[eigenvalues > 1e-10]
    bins = np.linspace(0, 6.4, 72)
    ax.hist(
        nonzero,
        bins=bins,
        density=False,
        weights=np.full(len(nonzero), 1.0 / p),
        color=SEQUENTIAL,
        alpha=0.85,
        edgecolor=SURFACE,
        linewidth=0.4,
        label="Empirical eigenvalues",
    )

    grid = np.linspace(max(lower, 1e-6), upper, 600)
    width = bins[1] - bins[0]
    ax.plot(
        grid,
        marchenko_pastur_density(grid, q) * width,
        color=CRITICAL,
        linewidth=2.0,
        label="Marchenko-Pastur (pure noise)",
    )

    top = ax.get_ylim()[1]
    ax.axvline(upper, color=INK_SECONDARY, linewidth=1.2, linestyle="--")
    ax.annotate(
        f"noise edge $(1+\\sqrt{{q}})^2$ = {upper:.2f}",
        xy=(upper, top * 0.50),
        xytext=(upper - 1.75, top * 0.62),
        color=INK_SECONDARY,
        fontsize=8.5,
        ha="left",
        arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, linewidth=0.9),
    )
    ax.annotate(
        f"only {n_signal} eigenvalues clear the edge,\nand they carry "
        f"{variance_in_signal:.0%} of total variance\n"
        f"(largest = {eigenvalues.max():.0f}: the market factor)",
        xy=(2.55, top * 0.26),
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    ax.annotate(
        f"a further {n_zero} eigenvalues are exactly zero:\n"
        f"with $p > n$ the matrix is singular and\ncannot be inverted at all",
        xy=(0.62, top * 0.74),
        color=MUTED,
        fontsize=8.5,
    )

    ax.set_xlim(0, 6.4)
    ax.set_xlabel("Eigenvalue of the sample correlation matrix")
    ax.set_ylabel("Share of eigenvalues per bin")
    ax.set_title(
        f"Most of the sample correlation matrix is indistinguishable from noise "
        f"($p$ = {p}, $n$ = {n})"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

    out = ROOT / "results/figures/spectrum.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
