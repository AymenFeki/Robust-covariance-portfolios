"""Covariance matrix estimators for high-dimensional return panels.

Every estimator takes an ``(n, p)`` array of returns (rows = observations,
columns = assets) and returns a ``(p, p)`` covariance matrix.

The problem these solve
-----------------------
With ``p`` assets and ``n`` observations the sample covariance matrix has
``p(p+1)/2`` free parameters estimated from ``n*p`` numbers. Once ``p`` is
comparable to ``n`` the estimate is dominated by noise; once ``p > n`` it is
singular outright. Mean-variance optimisation inverts this matrix, so it
loads precisely onto the noisiest directions of the spectrum.

Three families of fix are implemented here:

* **Shrinkage** -- pull the sample estimate toward a low-variance, biased
  target. The optimal intensity is available in closed form.
* **Random matrix theory** -- treat eigenvalues below the Marchenko-Pastur
  edge as pure noise and flatten them.
* **Factor structure** -- impose a low-rank-plus-diagonal model.

References
----------
Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
    large-dimensional covariance matrices. *Journal of Multivariate
    Analysis*, 88(2), 365-411.
Ledoit, O. and Wolf, M. (2003). Improved estimation of the covariance
    matrix of stock returns with an application to portfolio selection.
    *Journal of Empirical Finance*, 10(5), 603-621.
Laloux, L., Cizeau, P., Potters, M. and Bouchaud, J.-P. (2000). Random
    matrix theory and financial correlations. *International Journal of
    Theoretical and Applied Finance*, 3(3), 391-397.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sample_covariance",
    "ledoit_wolf_identity",
    "ledoit_wolf_constant_correlation",
    "marchenko_pastur_clipped",
    "pca_factor_model",
    "ESTIMATORS",
    "condition_number",
    "is_psd",
]


def _demean(returns: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return centred observations plus the panel shape."""
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D (n, p) array, got shape {X.shape}")
    n, p = X.shape
    if n < 2:
        raise ValueError("need at least two observations")
    return X - X.mean(axis=0), n, p


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Plain sample covariance, the maximum-likelihood estimate.

    Unbiased, and unusable when ``p`` approaches ``n``: its smallest
    eigenvalues are biased toward zero, which is exactly what portfolio
    optimisation divides by. Included as the baseline that the others have
    to beat.
    """
    Xc, n, _ = _demean(returns)
    return Xc.T @ Xc / (n - 1)


def ledoit_wolf_identity(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity matrix.

    Returns ``(sigma, shrinkage_intensity)``.

    The convex combination ``lambda * F + (1 - lambda) * S`` minimises
    expected squared Frobenius distance to the true covariance. The optimal
    ``lambda`` is the ratio of the estimation error in ``S`` to the total
    distance between ``S`` and the target ``F = (tr(S)/p) * I``, so a noisy
    sample estimate is shrunk harder.
    """
    Xc, n, p = _demean(returns)
    S = Xc.T @ Xc / n

    mu = np.trace(S) / p
    F = mu * np.eye(p)

    # Denominator: squared Frobenius distance between sample and target.
    d2 = np.sum((S - F) ** 2)
    if d2 <= 0:
        return S, 0.0

    # Numerator: sum of asymptotic variances of the entries of S. Computed
    # from fourth moments without forming n outer products explicitly.
    X2 = Xc**2
    phi = np.sum(X2.T @ X2) / n - np.sum(S**2)
    b2 = float(min(max(phi / n, 0.0), d2))

    lam = b2 / d2
    return lam * F + (1.0 - lam) * S, lam


def ledoit_wolf_constant_correlation(
    returns: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2003) shrinkage toward the constant-correlation target.

    Returns ``(sigma, shrinkage_intensity)``.

    The target keeps each asset's sample variance but replaces every pairwise
    correlation with the panel average. For equities this is a far better
    structural prior than the identity: it encodes the fact that stocks are
    genuinely positively correlated through the market factor, so less has to
    be given up in bias to buy the variance reduction.
    """
    Xc, n, p = _demean(returns)
    S = Xc.T @ Xc / n

    sd = np.sqrt(np.diag(S))
    if np.any(sd <= 0):
        raise ValueError("asset with zero sample variance in the window")

    R = S / np.outer(sd, sd)
    rbar = (R.sum() - p) / (p * (p - 1))

    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(S))

    d2 = np.sum((S - F) ** 2)
    if d2 <= 0:
        return S, 0.0

    X2 = Xc**2
    pi_mat = X2.T @ X2 / n - S**2
    pi_hat = np.sum(pi_mat)

    lam = float(min(max((pi_hat / n) / d2, 0.0), 1.0))
    return lam * F + (1.0 - lam) * S, lam


def marchenko_pastur_clipped(returns: np.ndarray) -> tuple[np.ndarray, int]:
    """Random-matrix eigenvalue clipping (Laloux et al.).

    Returns ``(sigma, n_signal_eigenvalues)``.

    If the ``p`` assets were pure independent noise, the eigenvalues of their
    sample correlation matrix would fall inside the Marchenko-Pastur support
    ``[(1 - sqrt(q))^2, (1 + sqrt(q))^2]`` with ``q = p / n``. Eigenvalues
    above the upper edge carry genuine common structure; everything below it
    is indistinguishable from noise and is replaced by a single flat level
    chosen to preserve the trace.

    Clipping is done on the correlation matrix so the procedure is invariant
    to each asset's volatility, then rescaled back to covariance.
    """
    Xc, n, p = _demean(returns)
    sd = Xc.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("asset with zero sample variance in the window")

    Z = Xc / sd
    C = Z.T @ Z / n
    eigvals, eigvecs = np.linalg.eigh(C)

    q = p / n
    upper_edge = (1.0 + np.sqrt(q)) ** 2

    signal = eigvals > upper_edge
    if not signal.any():  # degenerate window: keep the top eigenvalue
        signal[-1] = True

    n_signal = int(signal.sum())
    if n_signal == p:  # nothing left to flatten
        clipped = eigvals
    else:
        bulk_level = (np.trace(C) - eigvals[signal].sum()) / (p - n_signal)
        clipped = np.where(signal, eigvals, bulk_level)

    C_clean = (eigvecs * clipped) @ eigvecs.T

    # Clipping perturbs the diagonal; renormalise to a true correlation
    # matrix before rescaling by the sample volatilities.
    d = np.sqrt(np.diag(C_clean))
    C_clean = C_clean / np.outer(d, d)

    return C_clean * np.outer(sd, sd), n_signal


def pca_factor_model(returns: np.ndarray, n_factors: int = 5) -> tuple[np.ndarray, int]:
    """Low-rank-plus-diagonal covariance from the leading principal components.

    Returns ``(sigma, n_factors)``.

    Keeps the ``k`` largest eigen-directions as common factors and treats the
    remainder as uncorrelated idiosyncratic risk. This is the statistical
    cousin of a fundamental factor model: the same structure, with factors
    extracted from the data rather than specified by an analyst.
    """
    Xc, n, p = _demean(returns)
    if not 1 <= n_factors < p:
        raise ValueError(f"n_factors must be in [1, {p}), got {n_factors}")

    S = Xc.T @ Xc / n
    eigvals, eigvecs = np.linalg.eigh(S)

    top = np.argsort(eigvals)[::-1][:n_factors]
    loadings = eigvecs[:, top] * np.sqrt(np.maximum(eigvals[top], 0.0))

    common = loadings @ loadings.T
    idiosyncratic = np.maximum(np.diag(S) - np.diag(common), 1e-12)

    return common + np.diag(idiosyncratic), n_factors


#: Registry used by the backtest scripts. Each entry returns
#: ``(covariance, diagnostic)`` where the diagnostic is estimator-specific
#: (shrinkage intensity, number of retained eigenvalues, ...).
ESTIMATORS = {
    "Sample": lambda X: (sample_covariance(X), np.nan),
    "LW-identity": ledoit_wolf_identity,
    "LW-constcorr": ledoit_wolf_constant_correlation,
    "MP-clipped": marchenko_pastur_clipped,
    "PCA-5factor": lambda X: pca_factor_model(X, n_factors=5),
}


def condition_number(sigma: np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue.

    A direct measure of how badly inversion will amplify estimation error.
    Values above roughly ``1e6`` mean the inverse is numerically meaningless.
    """
    eigvals = np.linalg.eigvalsh(sigma)
    largest, smallest = float(eigvals.max()), float(eigvals.min())
    if smallest <= 0 or largest / smallest > np.finfo(float).max:
        return float("inf")  # singular: the inverse does not exist
    return largest / smallest


def is_psd(sigma: np.ndarray, tol: float = 1e-10) -> bool:
    """Whether ``sigma`` is positive semi-definite to within ``tol``."""
    return bool(np.linalg.eigvalsh(sigma).min() > -tol)


def marchenko_pastur_density(x: np.ndarray, q: float) -> np.ndarray:
    """Marchenko-Pastur probability density on the noise support.

    The limiting eigenvalue density of a sample correlation matrix built from
    ``n`` observations of ``p`` independent unit-variance series, with
    ``q = p / n`` held fixed. Used to draw the theoretical noise band against
    the empirical spectrum.
    """
    lo = (1.0 - np.sqrt(q)) ** 2
    hi = (1.0 + np.sqrt(q)) ** 2
    x = np.asarray(x, dtype=float)
    inside = (x > lo) & (x < hi)
    out = np.zeros_like(x)
    out[inside] = np.sqrt((hi - x[inside]) * (x[inside] - lo)) / (
        2.0 * np.pi * q * x[inside]
    )
    return out
