"""Tests for data cleaning and for the backtest's information set.

The look-ahead test is the important one here. Every other bug in a backtest
produces a plausible-looking result; look-ahead produces a *good* one, which
is why it survives review. So it is checked mechanically rather than by
reading the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robustcov.backtest import walk_forward
from robustcov.data import clean_price_panel, load_price_panel, to_log_returns
from robustcov.estimators import ledoit_wolf_identity, sample_covariance
from robustcov.portfolios import global_minimum_variance


def _price_series(n_days: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))


def _panel(**columns: np.ndarray) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    index = pd.bdate_range("2020-01-01", periods=length)
    return pd.DataFrame(columns, index=index)


# ------------------------------------------------------------------- cleaning


def test_clean_panel_leaves_well_behaved_series_alone():
    panel = _panel(AAA=_price_series(), BBB=_price_series(seed=1))
    cleaned, report = clean_price_panel(panel)
    assert list(cleaned.columns) == ["AAA", "BBB"]
    assert not report.dropped_tickers
    assert not report.bad_prints_fixed
    assert np.allclose(cleaned.to_numpy(), panel.to_numpy())


def test_unadjusted_split_causes_the_ticker_to_be_dropped():
    """A permanent level shift is indistinguishable from a crash, so the
    ticker is excluded rather than silently 'repaired'."""
    prices = _price_series()
    prices[150:] /= 2.0  # a 2-for-1 split the vendor never adjusted
    panel = _panel(AAA=prices, BBB=_price_series(seed=1))

    cleaned, report = clean_price_panel(panel)
    assert "AAA" not in cleaned.columns
    assert "BBB" in cleaned.columns
    assert [t for t, _, _ in report.dropped_tickers] == ["AAA"]
    assert report.n_input == 2 and report.n_output == 1


def test_isolated_bad_print_is_interpolated_not_dropped():
    """A spike that fully reverses the next day is unambiguous, so it is
    repaired and the ticker is kept."""
    prices = _price_series()
    original = prices[150]
    prices[150] *= 2.5
    panel = _panel(AAA=prices)

    cleaned, report = clean_price_panel(panel)
    assert "AAA" in cleaned.columns
    assert len(report.bad_prints_fixed) == 1
    assert report.bad_prints_fixed[0][0] == "AAA"
    # Repaired back to roughly where it belonged.
    assert cleaned["AAA"].iloc[150] == pytest.approx(original, rel=0.05)


def test_cleaning_report_never_credits_a_fix_on_a_dropped_ticker():
    prices = _price_series()
    prices[100] *= 2.5      # bad print, repairable
    prices[200:] /= 2.0     # later split, not repairable
    cleaned, report = clean_price_panel(_panel(AAA=prices))
    assert "AAA" not in cleaned.columns
    assert not report.bad_prints_fixed


def test_load_price_panel_drops_incomplete_histories(tmp_path):
    panel = _panel(AAA=_price_series(50), BBB=_price_series(50, seed=1))
    panel.iloc[:10, 1] = np.nan
    path = tmp_path / "panel.csv"
    panel.to_csv(path)

    assert list(load_price_panel(path).columns) == ["AAA"]
    assert list(load_price_panel(path, require_full_history=False).columns) == [
        "AAA",
        "BBB",
    ]


def test_log_returns_have_one_fewer_row():
    panel = _panel(AAA=_price_series(50))
    assert len(to_log_returns(panel)) == 49


# ---------------------------------------------------------------- no look-ahead


@pytest.fixture
def returns() -> pd.DataFrame:
    """Returns with real factor structure, not pure noise.

    This matters: on independent noise the Ledoit-Wolf estimate shrinks all
    the way to a scaled identity, whose minimum-variance portfolio is 1/N no
    matter what the data says. A look-ahead test run on that data passes
    trivially, because the weights do not depend on *any* returns, future or
    past. Factor structure keeps the estimator responsive so the test has
    something to detect.
    """
    rng = np.random.default_rng(7)
    n, p = 400, 15
    market = rng.normal(0, 0.01, (n, 1))
    data = market @ rng.uniform(0.5, 1.5, (1, p))
    data += rng.normal(0, 0.006, (n, p))
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(data, index=index, columns=[f"A{i}" for i in range(p)])


def test_weights_do_not_depend_on_future_returns(returns):
    """Corrupt everything from the second rebalance onward. The first set of
    weights must be bit-identical, because it could not have seen any of it."""
    window, step = 100, 20
    base = walk_forward(returns, ledoit_wolf_identity, window=window, step=step)

    tampered = returns.copy()
    tampered.iloc[window + step :] *= 100.0
    after = walk_forward(tampered, ledoit_wolf_identity, window=window, step=step)

    np.testing.assert_array_equal(
        base.weights.iloc[0].to_numpy(), after.weights.iloc[0].to_numpy()
    )


def test_every_rebalance_uses_only_its_own_trailing_window(returns):
    """Stronger version: perturbing the data one day *before* a rebalance must
    change its weights, and one day *after* must not."""
    window, step = 100, 20
    base = walk_forward(returns, ledoit_wolf_identity, window=window, step=step)

    before = returns.copy()
    before.iloc[window - 1] += 0.5
    after = returns.copy()
    after.iloc[window] += 0.5

    changed = walk_forward(before, ledoit_wolf_identity, window=window, step=step)
    unchanged = walk_forward(after, ledoit_wolf_identity, window=window, step=step)

    assert not np.allclose(
        base.weights.iloc[0].to_numpy(), changed.weights.iloc[0].to_numpy()
    )
    np.testing.assert_array_equal(
        base.weights.iloc[0].to_numpy(), unchanged.weights.iloc[0].to_numpy()
    )


# --------------------------------------------------------------- backtest shape


def test_backtest_covers_the_whole_out_of_sample_period(returns):
    window, step = 100, 20
    result = walk_forward(returns, ledoit_wolf_identity, window=window, step=step)
    assert len(result.returns) == len(returns) - window
    assert result.returns.index[0] == returns.index[window]
    assert result.returns.index[-1] == returns.index[-1]


def test_turnover_has_one_fewer_entry_than_rebalances(returns):
    result = walk_forward(returns, ledoit_wolf_identity, window=100, step=20)
    assert len(result.turnover) == len(result.weights) - 1
    assert (result.turnover >= 0).all()


def test_turnover_is_zero_for_a_constant_strategy(returns):
    result = walk_forward(
        returns,
        lambda X: (sample_covariance(X), np.nan),
        portfolio=lambda S: np.ones(len(S)) / len(S),
        window=100,
        step=20,
    )
    assert result.turnover.abs().max() == pytest.approx(0.0, abs=1e-12)


def test_transaction_costs_reduce_returns(returns):
    result = walk_forward(returns, ledoit_wolf_identity, window=100, step=20)
    assert result.summary(cost_bps=25)["ann_return_pct"] < (
        result.summary(cost_bps=0)["ann_return_pct"]
    )


def test_window_longer_than_history_is_rejected(returns):
    with pytest.raises(ValueError):
        walk_forward(returns, ledoit_wolf_identity, window=len(returns) + 1)


def test_summary_reports_shorting_for_unconstrained_gmv(returns):
    result = walk_forward(
        returns, ledoit_wolf_identity, portfolio=global_minimum_variance, window=100
    )
    summary = result.summary()
    assert summary["avg_leverage"] >= 1.0
    assert summary["ann_vol_pct"] > 0
