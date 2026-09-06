"""Loading and cleaning the price panel.

The raw dataset is daily closing prices for S&P 500 constituents, and it is
dirty in ways that will quietly wreck a covariance study. Inspecting every
daily move larger than 35% in the raw panel turns up three distinct causes:

1. **Bad prints** -- one wrong closing price, showing up as a spike that
   reverses completely the next day.
2. **Unadjusted corporate actions** -- splits and spinoffs, showing up as a
   permanent level shift (DVA's 2-for-1 in September 2013, BAX's Baxalta
   spinoff in July 2015).
3. **Corrupted segments** -- stretches where the source stitched two series
   together. HPQ across October 2015 alternates between roughly 12 and 28 on
   consecutive days, because HP Inc. and Hewlett Packard Enterprise rows are
   interleaved after the separation.

Only the first has an unambiguous fix. The second and third look identical
to a genuine crash from prices alone: CHK fell 33% on 8 February 2016 on
bankruptcy reports, which is a real return that happens to sit at exactly
the ratio a 3-for-2 split would produce.

So this module does not guess. Isolated reversing spikes are interpolated,
because that inference is safe. Anything else large and persistent means the
ticker cannot be trusted without a corporate-actions feed, and the ticker is
**dropped from the universe and reported by name**. Roughly 4% of the
universe goes this way -- a price worth paying to avoid either fabricating
returns or leaving 20-sigma artefacts in the covariance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_price_panel", "clean_price_panel", "to_log_returns", "CleaningReport"]


@dataclass
class CleaningReport:
    """Audit trail for :func:`clean_price_panel`.

    Every modification and every exclusion is recorded by ticker and date,
    so the cleaning step can be checked rather than trusted.
    """

    bad_prints_fixed: list[tuple[str, str]] = field(default_factory=list)
    dropped_tickers: list[tuple[str, str, float]] = field(default_factory=list)
    n_input: int = 0
    n_output: int = 0

    def __str__(self) -> str:
        return (
            f"{len(self.bad_prints_fixed)} bad print(s) interpolated, "
            f"{len(self.dropped_tickers)} ticker(s) dropped, "
            f"{self.n_output} of {self.n_input} tickers retained"
        )


def load_price_panel(
    path: str | Path,
    require_full_history: bool = True,
) -> pd.DataFrame:
    """Read the long-format raw file into a wide date-by-ticker price panel.

    Parameters
    ----------
    path
        CSV with ``Date``, ``Close`` and ``Name`` columns, or a pre-built
        wide panel (dates as the index, one column per ticker). Both plain
        and gzipped files are accepted.
    require_full_history
        Keep only tickers observed on every date. This trades universe size
        for a strictly balanced panel, which every estimator here assumes.
        The cost is survivorship bias -- see ``docs/METHODOLOGY.md``.
    """
    path = Path(path)
    frame = pd.read_csv(path, parse_dates=[0])

    if {"Name", "Close"}.issubset(frame.columns):
        date_col = frame.columns[0]
        panel = frame.pivot(index=date_col, columns="Name", values="Close")
    else:
        panel = frame.set_index(frame.columns[0])

    panel = panel.sort_index()
    panel.index.name = "date"
    # Some mirrors prefix a handful of tickers with their exchange.
    panel.columns = [str(c).split(":")[-1] for c in panel.columns]
    panel.columns.name = "ticker"

    if require_full_history:
        complete = panel.notna().sum() == len(panel)
        panel = panel.loc[:, complete]

    return panel.astype(float)


def clean_price_panel(
    panel: pd.DataFrame,
    threshold: float = 0.35,
    reversal_ratio: float = 0.7,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Interpolate isolated bad prints; drop tickers with untrustworthy moves.

    Parameters
    ----------
    panel
        Wide price panel from :func:`load_price_panel`.
    threshold
        Absolute log return flagged for inspection. ``0.35`` is roughly a 42%
        one-day move. Large-cap US equities essentially never do this on
        genuine news -- in this panel only one flagged move (CHK, February
        2016) turns out to be real -- while it sits below the smallest common
        split ratio, so real splits are all caught.
    reversal_ratio
        A flagged move counts as a bad print only when the very next day
        moves back by at least this fraction of it in the opposite direction.

    Returns
    -------
    (clean_panel, report)

    Notes
    -----
    A repaired bad print is replaced by the geometric mean of its neighbours,
    which leaves the surrounding price level untouched. Tickers are dropped
    rather than back-adjusted because a permanent level shift is
    observationally identical to a genuine crash, and inventing the wrong
    correction is worse than losing the name: a fabricated zero return
    understates that asset's variance in every window that contains it.
    """
    cleaned = panel.copy()
    report = CleaningReport(n_input=panel.shape[1])
    drop: list[str] = []

    for ticker in panel.columns:
        prices = panel[ticker].to_numpy(dtype=float).copy()
        repaired_here: list[tuple[str, str]] = []
        offending: int | None = None

        # Walk forward one day at a time, recomputing each move from the
        # prices as they stand. A repaired spike must not be re-flagged on
        # the following day, which is what happens if the moves are computed
        # once up front: a single bad print shows up as two large returns.
        t = 1
        while t < len(prices):
            move = np.log(prices[t] / prices[t - 1])
            if abs(move) <= threshold:
                t += 1
                continue

            if t + 1 >= len(prices):  # nothing after it to corroborate
                offending = t
                break

            next_move = np.log(prices[t + 1] / prices[t])
            reverses = (
                np.sign(next_move) != np.sign(move)
                and abs(next_move) > threshold * reversal_ratio
            )
            if not reverses:  # a permanent level shift; cause unknowable
                offending = t
                break

            prices[t] = np.exp((np.log(prices[t - 1]) + np.log(prices[t + 1])) / 2.0)
            repaired_here.append((ticker, str(panel.index[t].date())))
            t += 1

        if offending is not None:
            drop.append(ticker)
            report.dropped_tickers.append(
                (
                    ticker,
                    str(panel.index[offending].date()),
                    float(np.log(prices[offending] / prices[offending - 1])),
                )
            )
            continue

        report.bad_prints_fixed.extend(repaired_here)
        cleaned[ticker] = prices

    cleaned = cleaned.drop(columns=drop)
    report.n_output = cleaned.shape[1]
    return cleaned, report


def to_log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, first row dropped.

    Log returns are used because they aggregate additively over time, which
    keeps the multi-day holding-period arithmetic in the backtest exact.
    """
    return np.log(panel).diff().dropna(how="all")
