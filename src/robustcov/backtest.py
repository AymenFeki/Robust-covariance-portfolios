"""Walk-forward backtesting engine.

The single most important property of this module is that no weight is ever
computed from data that would not have been available at the time it is
applied. Concretely, the covariance used for the holding period starting at
index ``t`` is estimated strictly from rows ``[t - window, t)``, and the
returns it is scored against are rows ``[t, t + step)``.

That sounds obvious and is the most common way a backtest silently lies.
``tests/test_data_and_backtest.py`` enforces it mechanically: perturbing any return
at or after the rebalance date must leave the weights bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimators import ESTIMATORS
from .portfolios import global_minimum_variance

__all__ = ["BacktestResult", "walk_forward", "summarise"]

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Per-strategy output of a walk-forward run."""

    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    diagnostics: pd.Series

    def summary(
        self, cost_bps: float = 0.0, periods_per_year: int = TRADING_DAYS
    ) -> dict[str, float]:
        """Headline statistics, optionally net of linear transaction costs.

        ``cost_bps`` is charged on one-way traded notional at each rebalance,
        so a turnover of 2.0 (a full round-trip of the book) at 10 bps costs
        20 bps of net asset value.

        ``periods_per_year`` controls annualisation and defaults to 252
        trading days. Every script and test in this repo uses daily data, so
        the default is what they all rely on -- but the return series itself
        is frequency-agnostic, so weekly data should pass 52 here and monthly
        data 12, or the reported annualised figures will be wrong by a factor
        of roughly sqrt(252/actual).
        """
        gross = self.returns
        net_mean = gross.mean()
        if cost_bps > 0 and len(self.turnover):
            total_cost = self.turnover.sum() * cost_bps / 1e4
            net_mean = net_mean - total_cost / len(gross)

        vol = gross.std(ddof=1)
        scale = np.sqrt(periods_per_year)
        return {
            "ann_return_pct": float(net_mean * periods_per_year * 100),
            "ann_vol_pct": float(vol * scale * 100),
            "sharpe": (float(net_mean / vol * scale) if vol > 0 else np.nan),
            "max_drawdown_pct": float(_max_drawdown(gross) * 100),
            "avg_turnover": float(self.turnover.mean()) if len(self.turnover) else 0.0,
            "avg_leverage": float(self.weights.abs().sum(axis=1).mean()),
            "avg_max_weight": float(self.weights.abs().max(axis=1).mean()),
            "avg_short_pct": float(
                (self.weights[self.weights < 0].sum(axis=1).abs().mean()) * 100
            ),
            "diagnostic": float(self.diagnostics.mean()),
        }


def _max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative log-return path."""
    equity = returns.cumsum()
    return float((equity - equity.cummax()).min())


def walk_forward(
    returns: pd.DataFrame,
    estimator,
    portfolio=global_minimum_variance,
    window: int = 250,
    step: int = 21,
) -> BacktestResult:
    """Run one estimator/portfolio pair over the panel.

    Parameters
    ----------
    returns
        Date-indexed daily returns, one column per asset.
    estimator
        Callable mapping an ``(n, p)`` array to ``(covariance, diagnostic)``.
    portfolio
        Callable mapping a covariance matrix to weights.
    window
        Length of the rolling estimation window in trading days. The ratio
        ``p / window`` is the quantity that governs how badly the sample
        estimator fails.
    step
        Trading days between rebalances. Weights are held fixed in between.
    """
    values = returns.to_numpy(dtype=float)
    n_obs, n_assets = values.shape
    if window >= n_obs:
        raise ValueError(f"window {window} exceeds available history {n_obs}")

    out_returns: list[float] = []
    out_dates: list[pd.Timestamp] = []
    weight_rows: list[np.ndarray] = []
    weight_dates: list[pd.Timestamp] = []
    turnovers: list[float] = []
    diagnostics: list[float] = []
    previous: np.ndarray | None = None

    for start in range(window, n_obs, step):
        # --- information set: strictly before `start` -----------------
        train = values[start - window : start]
        sigma, diagnostic = estimator(train)
        weights = portfolio(sigma)

        # --- applied to the future, never seen above ------------------
        holding = values[start : min(start + step, n_obs)]
        if not len(holding):
            break
        out_returns.extend(holding @ weights)
        out_dates.extend(returns.index[start : start + len(holding)])

        weight_rows.append(weights)
        weight_dates.append(returns.index[start])
        diagnostics.append(diagnostic)
        if previous is not None:
            turnovers.append(float(np.abs(weights - previous).sum()))
        previous = weights

    return BacktestResult(
        returns=pd.Series(
            out_returns, index=pd.DatetimeIndex(out_dates), name="return"
        ),
        weights=pd.DataFrame(weight_rows, index=weight_dates, columns=returns.columns),
        turnover=pd.Series(turnovers, index=weight_dates[1:], name="turnover"),
        diagnostics=pd.Series(diagnostics, index=weight_dates, name="diagnostic"),
    )


def summarise(
    returns: pd.DataFrame,
    portfolio=global_minimum_variance,
    estimators: dict | None = None,
    window: int = 250,
    step: int = 21,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Run every estimator and collect the summary statistics into a table."""
    estimators = estimators or ESTIMATORS
    rows = {}
    for name, estimator in estimators.items():
        result = walk_forward(
            returns, estimator, portfolio=portfolio, window=window, step=step
        )
        rows[name] = result.summary(cost_bps=cost_bps)
    return pd.DataFrame(rows).T
