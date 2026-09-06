"""Portfolio construction rules.

Each function maps a covariance matrix to a weight vector summing to one.
No expected-return input is used anywhere: every rule here is risk-based.

That is deliberate. Expected returns are estimated with far larger relative
error than covariances, and mean-variance optimisation is more sensitive to
errors in the mean than to errors in the covariance (Chopra and Ziemba,
1993). Holding expected returns fixed isolates the question this project
actually asks -- does a better covariance estimate produce a better
portfolio -- instead of confounding it with return forecasting.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "equal_weight",
    "inverse_variance",
    "global_minimum_variance",
    "min_variance_long_only",
    "equal_risk_contribution",
    "risk_contributions",
]


def equal_weight(sigma: np.ndarray) -> np.ndarray:
    """The 1/N portfolio. Ignores ``sigma`` entirely.

    A demanding benchmark in practice: DeMiguel, Garlappi and Uppal (2009)
    found it beat most optimised alternatives out of sample, because
    estimation error swamped the theoretical gains.
    """
    p = len(sigma)
    return np.ones(p) / p


def inverse_variance(sigma: np.ndarray) -> np.ndarray:
    """Weights proportional to ``1 / variance``; ignores correlations.

    Equivalent to the minimum-variance portfolio under the assumption that
    the correlation matrix is the identity.
    """
    inv_var = 1.0 / np.diag(sigma)
    return inv_var / inv_var.sum()


def global_minimum_variance(sigma: np.ndarray) -> np.ndarray:
    """Unconstrained minimum-variance portfolio, ``w = S^-1 1 / (1' S^-1 1)``.

    Long and short positions are both allowed, so this is the purest test of
    an estimator: nothing constrains the optimiser away from the noisy
    directions of the spectrum, and a bad covariance estimate shows up
    immediately as enormous offsetting positions.

    Solved with ``scipy``-free linear algebra: a Cholesky solve when the
    matrix is positive definite, falling back to the pseudo-inverse when it
    is singular (which is what the sample estimator is whenever ``p >= n``).
    """
    p = len(sigma)
    ones = np.ones(p)

    try:
        cho = np.linalg.cholesky(sigma)
        z = np.linalg.solve(cho, ones)
        raw = np.linalg.solve(cho.T, z)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(sigma) @ ones

    total = ones @ raw
    if not np.isfinite(total) or abs(total) < 1e-300:
        return equal_weight(sigma)
    return raw / total


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``{w : w >= 0, sum(w) = 1}``.

    Duchi et al. (2008), O(p log p) via a sort.
    """
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u) - 1.0
    idx = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - cumulative / idx > 0)[0][-1]
    theta = cumulative[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def min_variance_long_only(
    sigma: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Minimum-variance portfolio subject to ``w >= 0`` and ``sum(w) = 1``.

    Solved by accelerated projected gradient descent (FISTA) on
    ``w' S w`` over the simplex. The step size is ``1 / L`` with ``L`` the
    largest eigenvalue of ``sigma``, which is the Lipschitz constant of the
    gradient, so no line search is needed.

    The no-shorting constraint is itself a form of regularisation: Jagannathan
    and Ma (2003) showed it is equivalent to shrinking the covariance matrix,
    which is why a long-only portfolio built on a bad estimate degrades far
    more gracefully than an unconstrained one.
    """
    p = len(sigma)
    lipschitz = float(np.linalg.eigvalsh(sigma).max())
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        return equal_weight(sigma)
    step = 1.0 / lipschitz

    w = np.ones(p) / p
    y = w.copy()
    t = 1.0

    for _ in range(max_iter):
        w_new = _project_simplex(y - step * (sigma @ y))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w, t = w_new, t_new

    return w


def risk_contributions(weights: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Each asset's share of total portfolio volatility.

    Contributions are ``w * (S w) / sqrt(w' S w)`` and sum exactly to the
    portfolio volatility, by Euler's theorem on the homogeneous-of-degree-one
    risk measure.
    """
    portfolio_vol = np.sqrt(weights @ sigma @ weights)
    if portfolio_vol <= 0:
        raise ValueError("portfolio has zero variance")
    return weights * (sigma @ weights) / portfolio_vol


def equal_risk_contribution(
    sigma: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Risk-parity portfolio: every asset contributes equal volatility.

    Solved by cyclical coordinate descent (Griveau-Billion, Richard and
    Roncalli, 2013). The problem

        min_w  0.5 w' S w - (1/p) sum(log w),  w > 0

    is strictly convex, and its unique solution rescales to the equal-risk
    portfolio. Each coordinate update is the positive root of a quadratic, so
    the whole thing is closed-form per step and converges in a few dozen
    sweeps even for hundreds of assets.
    """
    p = len(sigma)
    target = 1.0 / p

    # Inverse-volatility start: exact when correlations are equal.
    w = 1.0 / np.sqrt(np.diag(sigma))
    w /= w.sum()

    Sw = sigma @ w
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(p):
            a = sigma[i, i]
            # Contribution of every other asset to asset i's marginal risk.
            b = Sw[i] - a * w[i]
            c = -target
            root = (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
            delta = root - w[i]
            if delta != 0.0:
                w[i] = root
                Sw += delta * sigma[:, i]
                max_change = max(max_change, abs(delta))
        if max_change < tol:
            break

    return w / w.sum()


#: Registry used by the backtest scripts.
PORTFOLIOS = {
    "GMV": global_minimum_variance,
    "GMV-longonly": min_variance_long_only,
    "ERC": equal_risk_contribution,
    "InverseVar": inverse_variance,
    "EqualWeight": equal_weight,
}
