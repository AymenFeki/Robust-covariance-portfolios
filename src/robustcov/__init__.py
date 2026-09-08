"""Robust covariance estimation for portfolio construction.

A small library for the high-dimensional covariance problem in finance:
when the number of assets is comparable to the number of observations, the
sample covariance matrix is too noisy to invert, and portfolio optimisation
inverts it. Shrinkage, random-matrix filtering and factor structure are
implemented here and compared in an honest walk-forward backtest.
"""

from .backtest import BacktestResult, summarise, walk_forward
from .data import CleaningReport, clean_price_panel, load_price_panel, to_log_returns
from .estimators import (
    ESTIMATORS,
    condition_number,
    is_psd,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_identity,
    marchenko_pastur_clipped,
    marchenko_pastur_density,
    pca_factor_model,
    sample_covariance,
)
from .inference import (
    annualized_return_ci,
    annualized_vol_ci,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    expected_max_sharpe,
    sharpe_ratio,
    sharpe_ratio_ci,
    skew_kurtosis,
)
from .portfolios import (
    PORTFOLIOS,
    equal_risk_contribution,
    equal_weight,
    global_minimum_variance,
    inverse_variance,
    min_variance_long_only,
    risk_contributions,
)

__version__ = "0.1.0"

__all__ = [
    "load_price_panel",
    "clean_price_panel",
    "to_log_returns",
    "CleaningReport",
    "sample_covariance",
    "ledoit_wolf_identity",
    "ledoit_wolf_constant_correlation",
    "marchenko_pastur_clipped",
    "marchenko_pastur_density",
    "pca_factor_model",
    "condition_number",
    "is_psd",
    "ESTIMATORS",
    "equal_weight",
    "inverse_variance",
    "global_minimum_variance",
    "min_variance_long_only",
    "equal_risk_contribution",
    "risk_contributions",
    "PORTFOLIOS",
    "walk_forward",
    "summarise",
    "BacktestResult",
    "block_bootstrap_ci",
    "sharpe_ratio",
    "sharpe_ratio_ci",
    "annualized_return_ci",
    "annualized_vol_ci",
    "skew_kurtosis",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "__version__",
]
