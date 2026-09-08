"""Tests for bootstrap confidence intervals and the deflated Sharpe ratio.

The bootstrap tests check statistical properties (coverage, reproducibility,
sensitivity to the actual data) rather than exact numbers, since the whole
point of the tool is that its output is itself a random variable. The
deflated-Sharpe tests check the closed-form pieces against known reference
values and known monotonicity properties -- if any of those breaks, the
formula was mistyped.
"""

from __future__ import annotations

import numpy as np
import pytest

from robustcov.inference import (
    _norm_cdf,
    _norm_ppf,
    annualized_vol_ci,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    expected_max_sharpe,
    sharpe_ratio,
    sharpe_ratio_ci,
    skew_kurtosis,
)

# --------------------------------------------------------------- normal CDF/PPF


def test_norm_cdf_at_known_quantiles():
    """Textbook values: Phi(0)=0.5, Phi(1.96)=0.975, Phi(-1.96)=0.025."""
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)
    assert _norm_cdf(1.959963986120195) == pytest.approx(0.975, abs=1e-6)
    assert _norm_cdf(-1.959963986120195) == pytest.approx(0.025, abs=1e-6)


def test_norm_ppf_is_the_inverse_of_norm_cdf():
    for p in [0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999]:
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_norm_cdf_handles_arrays():
    out = _norm_cdf(np.array([0.0, 1.0, -1.0]))
    assert out.shape == (3,)
    assert out[0] == pytest.approx(0.5)


# --------------------------------------------------------------- skew/kurtosis


def test_skew_kurtosis_matches_scipy_on_a_skewed_sample():
    from scipy.stats import kurtosis as sp_kurtosis
    from scipy.stats import skew as sp_skew

    rng = np.random.default_rng(0)
    x = rng.exponential(scale=2.0, size=4000)  # genuinely skewed, unlike Gaussian
    skew, kurtosis = skew_kurtosis(x)
    assert skew == pytest.approx(sp_skew(x), abs=1e-9)
    assert kurtosis == pytest.approx(sp_kurtosis(x) + 3, abs=1e-9)


def test_skew_kurtosis_near_gaussian_values_for_normal_data():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(20000)
    skew, kurtosis = skew_kurtosis(x)
    assert skew == pytest.approx(0.0, abs=0.1)
    assert kurtosis == pytest.approx(3.0, abs=0.2)


def test_skew_kurtosis_constant_series_does_not_divide_by_zero():
    x = np.full(50, 0.01)
    skew, kurtosis = skew_kurtosis(x)
    assert skew == 0.0
    assert kurtosis == 3.0


# --------------------------------------------------------------- block bootstrap


@pytest.fixture
def iid_returns():
    rng = np.random.default_rng(2)
    return rng.normal(0.0006, 0.011, 1000)


def test_bootstrap_point_estimate_matches_direct_computation(iid_returns):
    point, _, _ = block_bootstrap_ci(iid_returns, lambda r: r.mean(), n_boot=200)
    assert point == pytest.approx(iid_returns.mean())


def test_bootstrap_ci_brackets_point_estimate(iid_returns):
    point, lower, upper = sharpe_ratio_ci(iid_returns, n_boot=1000, seed=5)
    assert lower <= point <= upper


def test_bootstrap_ci_is_reproducible_with_a_fixed_seed(iid_returns):
    a = block_bootstrap_ci(iid_returns, lambda r: r.std(), seed=42)
    b = block_bootstrap_ci(iid_returns, lambda r: r.std(), seed=42)
    assert a == b


def test_bootstrap_ci_widens_with_fewer_observations():
    """A statistic estimated from less data should have a wider CI."""
    rng = np.random.default_rng(3)
    long_series = rng.normal(0.0005, 0.01, 2000)
    short_series = long_series[:100]

    _, lo_long, hi_long = sharpe_ratio_ci(long_series, n_boot=1000, seed=9)
    _, lo_short, hi_short = sharpe_ratio_ci(short_series, n_boot=1000, seed=9)

    assert (hi_short - lo_short) > (hi_long - lo_long)


def test_bootstrap_ci_has_roughly_correct_coverage():
    """Simulate many synthetic 'backtests' from a known distribution and check
    that a nominal 90% CI for the mean actually contains the true mean about
    90% of the time -- the property a confidence interval is supposed to have.
    """
    rng = np.random.default_rng(11)
    true_mean = 0.0005
    hits = 0
    trials = 60
    for _ in range(trials):
        sample = rng.normal(true_mean, 0.01, 300)
        _, lo, hi = block_bootstrap_ci(
            sample, lambda r: r.mean(), n_boot=300, ci=0.90, seed=None
        )
        hits += lo <= true_mean <= hi
    # Not exactly 90% with only 60 trials -- allow a wide margin so this
    # isn't flaky, while still catching a badly broken implementation
    # (e.g. one that returns a near-zero-width or wildly miscentred interval).
    assert hits / trials >= 0.70


def test_bootstrap_rejects_too_few_observations():
    with pytest.raises(ValueError):
        block_bootstrap_ci(np.array([0.01, 0.02, -0.01]), lambda r: r.mean())


def test_annualized_vol_ci_is_positive_and_ordered(iid_returns):
    point, lower, upper = annualized_vol_ci(iid_returns, n_boot=500, seed=4)
    assert 0 < lower <= point <= upper


def test_sharpe_ratio_zero_volatility_is_nan():
    assert np.isnan(sharpe_ratio(np.zeros(50)))


# --------------------------------------------------------------- expected max Sharpe


def test_expected_max_sharpe_increases_with_number_of_trials():
    values = [expected_max_sharpe(0.0004, n) for n in [2, 5, 10, 50, 200]]
    assert values == sorted(values)


def test_expected_max_sharpe_scales_with_sqrt_variance():
    low = expected_max_sharpe(0.0001, 10)
    high = expected_max_sharpe(0.0004, 10)
    assert high == pytest.approx(2 * low, rel=1e-6)  # sqrt(0.0004/0.0001) = 2


def test_expected_max_sharpe_rejects_single_trial():
    with pytest.raises(ValueError):
        expected_max_sharpe(0.0004, 1)


# --------------------------------------------------------------- deflated Sharpe ratio


def test_dsr_is_one_half_when_observed_equals_benchmark():
    """No edge over the benchmark at all -> exactly 50/50."""
    for sr in [0.0, 0.02, 0.05, -0.01]:
        assert deflated_sharpe_ratio(sr, sr, n_obs=1000, skew=0.0, kurtosis=3.0) == (
            pytest.approx(0.5)
        )


def test_dsr_increases_with_observed_sharpe():
    values = [
        deflated_sharpe_ratio(sr, 0.0, n_obs=1000, skew=0.0, kurtosis=3.0)
        for sr in [-0.05, 0.0, 0.02, 0.05, 0.1]
    ]
    assert values == sorted(values)


def test_dsr_increases_with_sample_size_for_a_fixed_positive_edge():
    """More observations of the same true edge -> more confidence it's real."""
    values = [
        deflated_sharpe_ratio(0.03, 0.0, n_obs=n, skew=0.0, kurtosis=3.0)
        for n in [50, 250, 1000, 5000]
    ]
    assert values == sorted(values)


def test_dsr_penalises_more_trials_for_the_same_observed_sharpe():
    """The more strategies you tried, the less impressive the same Sharpe is."""
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0008, 0.01, 1000)
    few_trials = deflated_sharpe_ratio_from_returns(returns, n_trials=2)
    many_trials = deflated_sharpe_ratio_from_returns(returns, n_trials=50)
    assert many_trials["deflated_sharpe_ratio"] < few_trials["deflated_sharpe_ratio"]
    assert many_trials["benchmark_sharpe_per_period"] > (
        few_trials["benchmark_sharpe_per_period"]
    )


def test_dsr_matches_lo_2002_variance_under_gaussian_assumptions():
    """With skew=0 and kurtosis=3 (Gaussian), the formula's variance term
    reduces to Lo (2002)'s correction for the Sharpe-ratio estimator's own
    variance, `1 + SR^2/2` -- not plain `1`, since the Sharpe ratio's
    sampling variance depends on the true Sharpe ratio itself (dividing by
    an estimated standard deviation adds variance beyond the mean's own).
    """
    sr, n_obs = 0.04, 500
    dsr = deflated_sharpe_ratio(sr, 0.0, n_obs=n_obs, skew=0.0, kurtosis=3.0)
    expected = _norm_cdf(sr * np.sqrt(n_obs - 1) / np.sqrt(1 + 0.5 * sr**2))
    assert dsr == pytest.approx(expected, abs=1e-9)


def test_dsr_handles_pathological_denominator_gracefully():
    """Extreme skew relative to the Sharpe ratio can make the variance term
    negative (``1 - skew*SR + ...`` with a large positive ``skew*SR``);
    this must not raise or return NaN."""
    result = deflated_sharpe_ratio(5.0, 0.0, n_obs=100, skew=50.0, kurtosis=3.0)
    assert result == 0.5


def test_dsr_rejects_too_few_observations():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.05, 0.0, n_obs=1, skew=0.0, kurtosis=3.0)


def test_dsr_from_returns_end_to_end_is_internally_consistent():
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0006, 0.009, 800)
    result = deflated_sharpe_ratio_from_returns(
        returns, n_trials=5, periods_per_year=252
    )

    assert result["n_obs"] == 800
    assert result["n_trials"] == 5
    assert result["sharpe_annualized"] == pytest.approx(
        result["sharpe_per_period"] * np.sqrt(252)
    )
    assert 0.0 <= result["deflated_sharpe_ratio"] <= 1.0

    direct = deflated_sharpe_ratio(
        result["sharpe_per_period"],
        result["benchmark_sharpe_per_period"],
        n_obs=800,
        skew=result["skew"],
        kurtosis=result["kurtosis"],
    )
    assert direct == pytest.approx(result["deflated_sharpe_ratio"])


def test_dsr_from_returns_single_trial_tests_against_zero():
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0006, 0.009, 500)
    result = deflated_sharpe_ratio_from_returns(returns, n_trials=1)
    assert result["benchmark_sharpe_per_period"] == 0.0
