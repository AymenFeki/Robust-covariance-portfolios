"""Tests for the portfolio construction rules.

Each optimiser is checked against a case with a known closed-form answer, and
the iterative ones are additionally checked against the optimality conditions
they are supposed to satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest

from robustcov.portfolios import (
    _project_simplex,
    equal_risk_contribution,
    equal_weight,
    global_minimum_variance,
    inverse_variance,
    min_variance_long_only,
    risk_contributions,
)


@pytest.fixture
def sigma() -> np.ndarray:
    rng = np.random.default_rng(0)
    n, p = 500, 12
    data = rng.standard_normal((n, 1)) @ rng.uniform(0.5, 1.5, (1, p))
    data += 0.6 * rng.standard_normal((n, p))
    return np.cov(data, rowvar=False)


@pytest.fixture
def diagonal_sigma() -> np.ndarray:
    return np.diag([0.04, 0.01, 0.09, 0.16, 0.25])


# ------------------------------------------------------------------- the basics


@pytest.mark.parametrize(
    "rule",
    [
        equal_weight,
        inverse_variance,
        global_minimum_variance,
        min_variance_long_only,
        equal_risk_contribution,
    ],
)
def test_weights_sum_to_one(rule, sigma):
    assert rule(sigma).sum() == pytest.approx(1.0, abs=1e-9)


def test_gmv_reduces_to_inverse_variance_when_uncorrelated(diagonal_sigma):
    """With no correlations, minimum variance weights each asset by 1/variance."""
    assert np.allclose(
        global_minimum_variance(diagonal_sigma), inverse_variance(diagonal_sigma)
    )


def test_gmv_beats_every_other_rule_in_sample(sigma):
    """It is the in-sample variance minimiser by construction -- if this
    fails, the optimiser is broken."""
    optimal = global_minimum_variance(sigma)
    best = optimal @ sigma @ optimal
    for rule in (equal_weight, inverse_variance, equal_risk_contribution):
        w = rule(sigma)
        assert w @ sigma @ w >= best - 1e-12


def test_gmv_survives_a_singular_covariance():
    """A rank-deficient input must not produce NaNs or blow up."""
    rng = np.random.default_rng(4)
    data = rng.standard_normal((20, 40))
    singular = np.cov(data, rowvar=False)
    weights = global_minimum_variance(singular)
    assert np.all(np.isfinite(weights))
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


# -------------------------------------------------------------- simplex & FISTA


def test_simplex_projection_is_a_projection():
    rng = np.random.default_rng(5)
    for _ in range(50):
        v = rng.normal(size=20) * 3
        w = _project_simplex(v)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(w >= -1e-15)


def test_simplex_projection_leaves_points_already_inside_untouched():
    w = np.array([0.2, 0.3, 0.5])
    assert np.allclose(_project_simplex(w), w)


def test_simplex_projection_matches_brute_force():
    """Check against a direct numerical minimisation of ||w - v||^2."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(6)
    v = rng.normal(size=6)
    fast = _project_simplex(v)
    slow = minimize(
        lambda w: np.sum((w - v) ** 2),
        x0=np.ones(6) / 6,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        bounds=[(0, None)] * 6,
        tol=1e-12,
    ).x
    assert np.allclose(fast, slow, atol=1e-6)


def test_long_only_respects_its_constraints(sigma):
    w = min_variance_long_only(sigma)
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_long_only_matches_a_general_purpose_solver(sigma):
    """The hand-rolled projected-gradient solver must agree with SLSQP."""
    from scipy.optimize import minimize

    p = len(sigma)
    ours = min_variance_long_only(sigma)
    theirs = minimize(
        lambda w: w @ sigma @ w,
        x0=np.ones(p) / p,
        jac=lambda w: 2 * sigma @ w,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        bounds=[(0, None)] * p,
        tol=1e-14,
        options={"maxiter": 2000},
    ).x
    assert ours @ sigma @ ours == pytest.approx(theirs @ sigma @ theirs, rel=1e-6)


def test_long_only_equals_unconstrained_when_no_shorts_are_wanted(diagonal_sigma):
    """With a diagonal covariance the unconstrained optimum is already
    non-negative, so the constraint should not bind."""
    assert np.allclose(
        min_variance_long_only(diagonal_sigma),
        global_minimum_variance(diagonal_sigma),
        atol=1e-7,
    )


# ------------------------------------------------------------------ risk parity


def test_risk_contributions_sum_to_portfolio_volatility(sigma):
    """Euler's theorem: the parts must add up to the whole."""
    w = equal_weight(sigma)
    total = np.sqrt(w @ sigma @ w)
    assert risk_contributions(w, sigma).sum() == pytest.approx(total, rel=1e-12)


def test_erc_equalises_risk_contributions(sigma):
    w = equal_risk_contribution(sigma)
    contributions = risk_contributions(w, sigma)
    shares = contributions / contributions.sum()
    assert np.allclose(shares, 1.0 / len(sigma), atol=1e-6)


def test_erc_is_long_only(sigma):
    assert np.all(equal_risk_contribution(sigma) > 0)


def test_erc_reduces_to_inverse_volatility_when_uncorrelated(diagonal_sigma):
    """With zero correlations, equal risk contribution means weights
    proportional to 1/volatility."""
    expected = 1.0 / np.sqrt(np.diag(diagonal_sigma))
    expected /= expected.sum()
    assert np.allclose(equal_risk_contribution(diagonal_sigma), expected, atol=1e-8)


def test_erc_sits_between_equal_weight_and_minimum_variance(sigma):
    """Risk parity is the usual middle ground: less concentrated than
    minimum variance, less naive than 1/N."""
    variances = {
        name: w @ sigma @ w
        for name, w in {
            "gmv": min_variance_long_only(sigma),
            "erc": equal_risk_contribution(sigma),
            "ew": equal_weight(sigma),
        }.items()
    }
    assert variances["gmv"] <= variances["erc"] <= variances["ew"]
