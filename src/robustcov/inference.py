"""Statistical inference on backtest performance: how much of it is noise?

Every number in ``BacktestResult.summary()`` is a point estimate computed
from one realised path of returns. Two questions that raises, unaddressed
elsewhere in this repo, are answered here:

1. **How uncertain is any one statistic?** A Sharpe ratio of 0.75 computed
   from 1,000-odd daily returns has a standard error large enough that a
   second, equally valid realisation of the same market could easily have
   produced 0.5 or 1.0. :func:`block_bootstrap_ci` answers this by
   resampling the realised return path.
2. **Given that five estimators were compared and the best one picked, how
   much of its apparent edge is just the best draw among five noisy
   draws?** :func:`deflated_sharpe_ratio` (Bailey & Lopez de Prado, 2014)
   answers this by working out what Sharpe ratio the *best of N* strategies
   would achieve by pure chance, and asking whether the observed Sharpe
   clears that bar.

Both tools work directly on a realised return series and are agnostic to
where it came from -- they do not know or care that it was produced by a
walk-forward backtest over a rolling covariance estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

__all__ = [
    "block_bootstrap_ci",
    "sharpe_ratio",
    "sharpe_ratio_ci",
    "annualized_return_ci",
    "annualized_vol_ci",
    "skew_kurtosis",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
]

_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio of a per-period return series. No risk-free rate."""
    vol = returns.std(ddof=1)
    if vol <= 0:
        return float("nan")
    return float(returns.mean() / vol * np.sqrt(periods_per_year))


def _circular_block_bootstrap_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """One resample's worth of indices, via the circular block bootstrap.

    Draws blocks of ``block_size`` *consecutive* (wrapping around the end of
    the series) observations until the resample reaches length ``n``, then
    truncates. Preserving short-range runs like this is what a plain iid
    bootstrap cannot do: daily returns are approximately uncorrelated in
    mean but not in variance (volatility clustering), so resampling single
    days independently would understate the true sampling variance of any
    statistic that depends on the return path rather than just its
    marginal distribution.

    Reference: Kunsch (1989), "The Jackknife and the Bootstrap for General
    Stationary Observations". The wrap-around (circular) variant avoids the
    boundary bias of the plain moving-block bootstrap, at the cost of
    occasionally splicing the series' end to its start inside one block --
    negligible when ``block_size << n``.
    """
    n_blocks = -(-n // block_size)  # ceil
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_size)
    idx = (starts[:, None] + offsets[None, :]) % n
    return idx.reshape(-1)[:n]


def block_bootstrap_ci(
    returns: pd.Series | np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    block_size: int | None = None,
    ci: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for ``stat_fn(returns)``.

    Parameters
    ----------
    returns
        The realised per-period return series (e.g. one strategy's
        out-of-sample daily returns from :func:`~robustcov.backtest.walk_forward`).
    stat_fn
        Any function of a 1-D array of returns, e.g. :func:`sharpe_ratio`,
        ``lambda r: r.mean() * 252``, or a drawdown statistic.
    n_boot
        Number of bootstrap resamples. 2000 gives percentile estimates
        stable to about +/-1 percentage point at the 2.5/97.5 tails.
    block_size
        Length of the resampling blocks. Defaults to ``round(n ** (1/3))``,
        the standard rule-of-thumb rate for block bootstraps of dependent
        data (block length should grow like ``n^{1/3}`` for the bootstrap
        to be consistent); pass an explicit value -- e.g. the backtest's
        rebalance ``step`` -- if you have a more specific reason to believe
        dependence extends further.
    ci
        Confidence level, e.g. 0.95 for a 95% interval.

    Returns
    -------
    ``(point_estimate, lower, upper)``. The point estimate is ``stat_fn``
    applied to the *original* (non-bootstrapped) series, not the mean of the
    bootstrap replicates -- the two agree in the limit but not exactly.

    This resamples the realised return path, not the estimation procedure
    that produced it: it quantifies "how much would this statistic vary
    across alternative histories the market could plausibly have drawn",
    not "how much would the portfolio weights vary under a different
    covariance estimate". The latter is a different, harder question this
    function does not answer.
    """
    values = np.asarray(returns, dtype=float)
    n = len(values)
    if n < 8:
        raise ValueError(f"need at least 8 observations to bootstrap, got {n}")

    if block_size is None:
        block_size = max(1, round(n ** (1 / 3)))
    block_size = min(block_size, n)

    point = float(stat_fn(values))

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circular_block_bootstrap_indices(n, block_size, rng)
        replicates[b] = stat_fn(values[idx])

    replicates = replicates[np.isfinite(replicates)]
    alpha = 1.0 - ci
    lower, upper = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return point, float(lower), float(upper)


def sharpe_ratio_ci(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
    n_boot: int = 2000,
    block_size: int | None = None,
    ci: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Convenience wrapper: :func:`block_bootstrap_ci` for the Sharpe ratio."""
    return block_bootstrap_ci(
        returns,
        lambda r: sharpe_ratio(r, periods_per_year=periods_per_year),
        n_boot=n_boot,
        block_size=block_size,
        ci=ci,
        seed=seed,
    )


def annualized_return_ci(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
    n_boot: int = 2000,
    block_size: int | None = None,
    ci: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """:func:`block_bootstrap_ci` for annualised mean return (percent)."""
    return block_bootstrap_ci(
        returns,
        lambda r: r.mean() * periods_per_year * 100,
        n_boot=n_boot,
        block_size=block_size,
        ci=ci,
        seed=seed,
    )


def annualized_vol_ci(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
    n_boot: int = 2000,
    block_size: int | None = None,
    ci: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """:func:`block_bootstrap_ci` for annualised volatility (percent)."""
    return block_bootstrap_ci(
        returns,
        lambda r: r.std(ddof=1) * np.sqrt(periods_per_year) * 100,
        n_boot=n_boot,
        block_size=block_size,
        ci=ci,
        seed=seed,
    )


def skew_kurtosis(returns: np.ndarray) -> tuple[float, float]:
    """Sample skewness and kurtosis, method-of-moments (population) definition.

    Returns ``(skewness, kurtosis)`` with kurtosis on the *raw* scale, where
    3.0 is the Gaussian value (not "excess kurtosis", which would be 0.0 for
    a Gaussian) -- this is the convention :func:`deflated_sharpe_ratio`
    expects, matching Bailey & Lopez de Prado's original formula.
    """
    values = np.asarray(returns, dtype=float)
    mean = values.mean()
    centered = values - mean
    m2 = np.mean(centered**2)
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)
    if m2 <= 0:
        return 0.0, 3.0
    skew = m3 / m2**1.5
    kurtosis = m4 / m2**2
    return float(skew), float(kurtosis)


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF without a scipy dependency."""
    arr = np.atleast_1d(np.asarray(x, dtype=float)) / np.sqrt(2.0)
    erf_vals = np.array([math.erf(v) for v in arr], dtype=float)
    result = 0.5 * (1.0 + erf_vals)
    return float(result[0]) if np.ndim(x) == 0 else result


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF), Acklam's rational approximation.

    Accurate to about 1e-9 over ``(0, 1)`` -- ample for the deflated Sharpe
    ratio, which only ever calls this at ``1 - 1/N`` for small integer
    ``N``. Avoids adding scipy as a runtime dependency for one function.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")

    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = np.sqrt(-2 * np.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """Expected Sharpe ratio of the best of ``n_trials`` strategies, under the
    null that all of them have true Sharpe ratio zero.

    This is the benchmark the deflated Sharpe ratio measures against: if you
    try enough strategies, the best one will look good even if every single
    one is worthless, purely from picking the max of ``n_trials`` noisy
    draws. Bailey & Lopez de Prado (2014), extending the classical extreme
    value approximation for the maximum of correlated Gaussians:

        E[max SR] ~= sqrt(V) * (
            (1-gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))
        )

    where ``V`` is the cross-sectional variance of the ``N`` trials' Sharpe
    ratios and ``gamma`` is the Euler-Mascheroni constant. ``sharpe_variance``
    must be supplied by the caller -- estimating it from only 5 trials (as
    in this repo's main study) is itself imprecise, which is exactly the
    kind of caveat this whole calculation is meant to make explicit rather
    than hide.
    """
    if n_trials < 2:
        raise ValueError("expected_max_sharpe needs at least 2 trials")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance cannot be negative")

    term1 = (1 - _EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n_trials)
    term2 = _EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * (term1 + term2))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probability the true Sharpe ratio exceeds ``benchmark_sharpe``.

    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting and Non-Normality". Standard
    Sharpe-ratio inference assumes returns are iid Gaussian; both
    assumptions are usually false for financial returns (fat tails,
    skew, volatility clustering), and the classical standard error is
    additionally too optimistic once ``benchmark_sharpe`` itself was chosen
    by picking the best of several trials (see :func:`expected_max_sharpe`).

    The test statistic

        z = (SR_hat - SR_0) * sqrt(T - 1)
            / sqrt(1 - skew*SR_hat + (kurtosis - 1)/4 * SR_hat^2)

    reduces, when ``skew = 0`` and ``kurtosis = 3`` (Gaussian), to Lo
    (2002)'s asymptotic variance for the Sharpe-ratio estimator itself --
    ``1 + SR_hat^2 / 2``, not plain ``1`` -- since dividing by an estimated
    standard deviation adds variance beyond what the mean alone would have.
    ``DSR = Phi(z)`` is then the probability,
    under the null that the true Sharpe ratio equals ``benchmark_sharpe``,
    of observing something at least this good -- i.e. how confident you can
    be that the strategy's edge is real once selection and non-normality are
    both priced in. A DSR near 0.5 means "indistinguishable from the noise
    floor set by however many strategies were tried"; above roughly 0.95 is
    the conventional threshold for "probably real".

    Parameters
    ----------
    observed_sharpe
        The (non-deflated) Sharpe ratio of the strategy being tested, on
        the same period basis as everything else here (i.e. whatever units
        ``skew``/``kurtosis`` and ``n_obs`` were computed in -- annualising
        the Sharpe ratio alone without correspondingly rescaling ``skew``
        and ``kurtosis`` would silently give the wrong answer, so this
        function is meant to be called with figures all on the *raw
        per-period* return scale; see :func:`deflated_sharpe_ratio_from_returns`
        for a wrapper that gets this right automatically).
    benchmark_sharpe
        The null to test against: pass 0.0 to test "is this Sharpe ratio
        distinguishable from zero at all", or :func:`expected_max_sharpe`'s
        output to test "is this Sharpe ratio distinguishable from what the
        best of N random strategies would achieve by chance".
    n_obs
        Number of return observations the Sharpe ratio was computed from.
    skew, kurtosis
        From :func:`skew_kurtosis` on the same return series. ``kurtosis``
        is on the raw scale (3.0 = Gaussian), matching that function.
    """
    if n_obs < 2:
        raise ValueError("need at least 2 observations")
    denom = 1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2
    if denom <= 0:
        # Pathological input (extreme skew/kurtosis relative to the Sharpe
        # ratio) that would otherwise divide by a negative number under the
        # square root; treat as maximal uncertainty rather than raising.
        return 0.5
    z = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(_norm_cdf(z))


def deflated_sharpe_ratio_from_returns(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    all_trial_sharpes: np.ndarray | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """End-to-end deflated Sharpe ratio for one strategy's return series.

    Computes the per-period Sharpe ratio, skew and kurtosis internally (so
    they are guaranteed to be on consistent units, see the note in
    :func:`deflated_sharpe_ratio`), works out the benchmark Sharpe ratio
    expected from the best of ``n_trials`` strategies, and returns both the
    raw and deflated figures together.

    Parameters
    ----------
    returns
        Per-period returns of the strategy actually being evaluated
        (typically the best performer among the trials).
    n_trials
        How many strategies were compared before this one was selected
        (e.g. 5, if five covariance estimators were backtested and this is
        the best of them). Passing 1 skips the multiple-testing correction
        entirely (tests this specific Sharpe ratio against zero).
    all_trial_sharpes
        Per-period (not annualised) Sharpe ratios of all ``n_trials``
        strategies, used to estimate the cross-sectional variance that
        :func:`expected_max_sharpe` needs. If omitted, falls back to
        treating ``observed_sharpe`` as if it were typical of the trials
        (a conservative approximation only used when the other trials'
        return series are unavailable).
    periods_per_year
        For reporting the annualised Sharpe ratio only -- all internal
        calculations use the per-period figures.
    """
    values = np.asarray(returns, dtype=float)
    observed_sharpe = sharpe_ratio(values, periods_per_year=1)
    skew, kurtosis = skew_kurtosis(values)

    if n_trials <= 1:
        benchmark = 0.0
    else:
        if all_trial_sharpes is not None and len(all_trial_sharpes) >= 2:
            sharpe_variance = float(np.var(all_trial_sharpes, ddof=1))
        else:
            sharpe_variance = observed_sharpe**2
        benchmark = expected_max_sharpe(sharpe_variance, n_trials)

    dsr = deflated_sharpe_ratio(
        observed_sharpe, benchmark, n_obs=len(values), skew=skew, kurtosis=kurtosis
    )
    return {
        "sharpe_per_period": observed_sharpe,
        "sharpe_annualized": observed_sharpe * np.sqrt(periods_per_year),
        "benchmark_sharpe_per_period": benchmark,
        "skew": skew,
        "kurtosis": kurtosis,
        "n_trials": n_trials,
        "n_obs": len(values),
        "deflated_sharpe_ratio": dsr,
    }
