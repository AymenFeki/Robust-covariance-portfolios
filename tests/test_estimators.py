"""Tests for the covariance estimators.

These check mathematical properties that must hold exactly or to a stated
tolerance -- positive semi-definiteness, trace and diagonal preservation,
shrinkage bounds, and the random-matrix edge behaviour on data whose true
structure is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from robustcov.estimators import (
    condition_number,
    is_psd,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_identity,
    marchenko_pastur_clipped,
    marchenko_pastur_density,
    pca_factor_model,
    sample_covariance,
)


@pytest.fixture
def noise() -> np.ndarray:
    """Pure independent noise: 150 observations of 80 unit-variance series."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((150, 80))


@pytest.fixture
def factor_panel() -> np.ndarray:
    """Returns with one strong common factor plus idiosyncratic noise."""
    rng = np.random.default_rng(1)
    n, p = 400, 60
    market = rng.standard_normal((n, 1))
    betas = rng.uniform(0.6, 1.4, size=(1, p))
    return market @ betas + 0.5 * rng.standard_normal((n, p))


@pytest.fixture
def wide_panel() -> np.ndarray:
    """More assets than observations: the singular regime."""
    rng = np.random.default_rng(2)
    return rng.standard_normal((60, 120))


# --------------------------------------------------------------------- basics


def test_sample_covariance_matches_numpy(noise):
    assert np.allclose(sample_covariance(noise), np.cov(noise, rowvar=False))


def test_sample_covariance_is_singular_when_p_exceeds_n(wide_panel):
    n, p = wide_panel.shape
    sigma = sample_covariance(wide_panel)
    assert np.linalg.matrix_rank(sigma) < p
    assert condition_number(sigma) == float("inf")


def test_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        sample_covariance(np.arange(10.0))


@pytest.mark.parametrize(
    "estimator",
    [
        ledoit_wolf_identity,
        ledoit_wolf_constant_correlation,
        marchenko_pastur_clipped,
        pca_factor_model,
    ],
)
def test_every_regularised_estimator_is_psd_and_invertible(estimator, wide_panel):
    """The whole point: usable even when the sample estimate is singular."""
    sigma, _ = estimator(wide_panel)
    assert sigma.shape == (wide_panel.shape[1],) * 2
    assert np.allclose(sigma, sigma.T)
    assert is_psd(sigma)
    assert condition_number(sigma) < 1e8


# ------------------------------------------------------------------- shrinkage


def test_lw_identity_shrinkage_is_a_valid_intensity(noise):
    _, intensity = ledoit_wolf_identity(noise)
    assert 0.0 <= intensity <= 1.0


def test_lw_identity_preserves_the_trace(noise):
    """Both the target and the sample estimate carry the same total variance,
    so any convex combination of them must too."""
    sigma, _ = ledoit_wolf_identity(noise)
    n = len(noise)
    mle = sample_covariance(noise) * (n - 1) / n
    assert np.trace(sigma) == pytest.approx(np.trace(mle), rel=1e-12)


def test_lw_constant_correlation_preserves_the_diagonal(noise):
    """The target keeps each asset's own variance; only correlations shrink."""
    sigma, _ = ledoit_wolf_constant_correlation(noise)
    n = len(noise)
    mle = sample_covariance(noise) * (n - 1) / n
    assert np.allclose(np.diag(sigma), np.diag(mle), rtol=1e-12)


def test_shrinkage_is_stronger_on_noisier_data():
    """With fewer observations per asset the sample estimate is less
    trustworthy, so the optimal intensity must rise.

    The panel needs genuine factor structure for this to be meaningful: on
    independent unit-variance noise the identity target is exactly correct,
    so the intensity saturates at 1.0 whatever the sample size.
    """
    rng = np.random.default_rng(9)
    n, p = 2000, 40
    market = rng.standard_normal((n, 1))
    long_panel = market @ rng.uniform(0.5, 1.5, (1, p))
    long_panel += 0.6 * rng.standard_normal((n, p))
    short_panel = long_panel[:100]

    _, weak = ledoit_wolf_identity(long_panel)
    _, strong = ledoit_wolf_identity(short_panel)
    assert 0.0 < weak < strong < 1.0


def test_shrinkage_lies_between_sample_and_target(noise):
    sigma, intensity = ledoit_wolf_identity(noise)
    n, p = noise.shape
    mle = sample_covariance(noise) * (n - 1) / n
    target = (np.trace(mle) / p) * np.eye(p)
    assert np.allclose(sigma, intensity * target + (1 - intensity) * mle)


# ------------------------------------------------------------- random matrices


def test_mp_density_integrates_to_one():
    """Sanity check on the theoretical density used to draw the noise band."""
    q = 0.5
    lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    grid = np.linspace(lo, hi, 200_001)
    assert np.trapezoid(marchenko_pastur_density(grid, q), grid) == pytest.approx(
        1.0, abs=2e-3
    )


def test_mp_density_is_zero_outside_its_support():
    q = 0.5
    outside = np.array([0.01, 0.05, 5.0, 20.0])
    assert np.all(marchenko_pastur_density(outside, q) == 0.0)


def test_mp_clipping_finds_almost_no_signal_in_pure_noise(noise):
    """If the data really is independent noise, essentially every eigenvalue
    should fall inside the theoretical band."""
    _, n_signal = marchenko_pastur_clipped(noise)
    assert n_signal <= 2  # the implementation always keeps at least one


def test_mp_clipping_finds_the_planted_factor(factor_panel):
    """One strong common factor should stand clear above the noise edge."""
    _, n_signal = marchenko_pastur_clipped(factor_panel)
    assert n_signal == 1


def test_mp_clipping_preserves_asset_variances(factor_panel):
    """Clipping filters correlations; each asset's own variance is untouched."""
    sigma, _ = marchenko_pastur_clipped(factor_panel)
    assert np.allclose(np.diag(sigma), factor_panel.var(axis=0, ddof=1), rtol=1e-10)


def test_mp_clipped_correlation_has_unit_diagonal(factor_panel):
    sigma, _ = marchenko_pastur_clipped(factor_panel)
    sd = np.sqrt(np.diag(sigma))
    correlation = sigma / np.outer(sd, sd)
    assert np.allclose(np.diag(correlation), 1.0, atol=1e-12)


def test_number_of_zero_eigenvalues_matches_the_mp_atom(wide_panel):
    """For q > 1 the theory predicts a point mass of size 1 - 1/q at zero."""
    n, p = wide_panel.shape
    q = p / n
    centred = wide_panel - wide_panel.mean(axis=0)
    standardised = centred / centred.std(axis=0, ddof=1)
    eigenvalues = np.linalg.eigvalsh(standardised.T @ standardised / n)
    observed = int((eigenvalues < 1e-10).sum())
    assert observed == pytest.approx(p * (1 - 1 / q), abs=2)


# ----------------------------------------------------------------- factor model


def test_pca_common_component_has_the_requested_rank(factor_panel):
    k = 3
    _, reported = pca_factor_model(factor_panel, n_factors=k)
    assert reported == k

    # Rebuild the common component explicitly and check its rank.
    centred = factor_panel - factor_panel.mean(axis=0)
    S = centred.T @ centred / len(centred)
    eigvals, eigvecs = np.linalg.eigh(S)
    top = np.argsort(eigvals)[::-1][:k]
    loadings = eigvecs[:, top] * np.sqrt(eigvals[top])
    assert np.linalg.matrix_rank(loadings @ loadings.T, tol=1e-8) == k


def test_pca_factor_model_preserves_variances(factor_panel):
    sigma, _ = pca_factor_model(factor_panel, n_factors=3)
    n = len(factor_panel)
    mle = sample_covariance(factor_panel) * (n - 1) / n
    assert np.allclose(np.diag(sigma), np.diag(mle), rtol=1e-10)


def test_pca_factor_model_validates_its_argument(factor_panel):
    with pytest.raises(ValueError):
        pca_factor_model(factor_panel, n_factors=0)
    with pytest.raises(ValueError):
        pca_factor_model(factor_panel, n_factors=factor_panel.shape[1])
