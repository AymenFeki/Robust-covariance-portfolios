# Methodology

Why each choice was made, and what would break if it were made differently.

---

## 1. The estimation problem

For `p` assets the covariance matrix has `p(p+1)/2` free parameters. At
`p = 428` that is 91,806, estimated from `n × p` return observations. The
governing quantity is the concentration ratio

```
q = p / n
```

Classical asymptotics assume `p` fixed and `n → ∞`, so `q → 0` and the sample
covariance is consistent. Finance never operates there: lengthening the window
to shrink `q` reaches back into a different volatility regime, so `n` is
bounded by economics, not by data availability.

What goes wrong is specific and it is not that the estimate is "a bit noisy".
The eigenvalues are biased **outward**: the largest are overestimated, the
smallest underestimated. As `q → 1` the smallest sample eigenvalue converges
to `(1 − √q)²`, which is 0 at `q = 1`. Since minimum-variance weights are
`w ∝ Σ⁻¹1`, the optimiser divides by those smallest eigenvalues, and it
therefore concentrates the portfolio on the directions where the estimate is
worst. This is why the sweep in the README peaks near `q = 1` rather than
degrading monotonically.

### The Marchenko–Pastur benchmark

If `p` independent unit-variance series are observed `n` times, the eigenvalue
density of the sample correlation matrix converges to

```
f(x) = √((λ₊ − x)(x − λ₋)) / (2π q x),   λ± = (1 ± √q)²
```

plus, when `q > 1`, an atom at zero of mass `1 − 1/q` (the matrix has rank
`n`, so `p − n` eigenvalues are exactly zero). This gives a *null hypothesis*
for "this direction is noise". On the first 250-day window the panel produces
179 zero eigenvalues against a predicted 178.0 — the theory is not
approximately right here, it is right.

## 2. Estimators

### Ledoit–Wolf shrinkage

Take a convex combination of the sample estimate `S` and a structured target
`F`:

```
Σ̂ = λF + (1 − λ)S
```

`S` is unbiased with high variance; `F` is biased with low variance. The `λ`
minimising expected squared Frobenius error has a closed form: the ratio of
the total estimation variance of `S` to the squared distance `‖S − F‖²`. No
cross-validation, no tuning parameter.

Two targets are implemented:

- **Scaled identity** (2004). Assumes zero correlation and equal variances.
  Badly wrong for equities, so `λ` stays small (0.056 on the first window) and
  the estimate stays close to the noisy `S`.
- **Constant correlation** (2003). Keeps each sample variance, replaces every
  pairwise correlation with the panel average. Much closer to the truth for
  stocks — they really are all positively correlated through the market — so
  the optimal intensity is far higher (0.314) and the variance reduction far
  larger. This is why it wins the volatility comparison.

The lesson generalises: shrinkage buys more when the target encodes real
structure. The intensity adapts automatically, so a better target is not a
risk.

### Eigenvalue clipping

Diagonalise the sample *correlation* matrix, keep every eigenvalue above
`λ₊`, and replace the rest with a single flat level chosen to preserve the
trace. Working on the correlation matrix makes the procedure invariant to
individual volatilities; the result is rescaled back afterwards.

Clipping perturbs the diagonal away from 1, so the implementation renormalises
to a proper correlation matrix before rescaling. Skipping that step silently
changes each asset's variance — `test_mp_clipping_preserves_asset_variances`
guards it.

### PCA factor model

Keep `k` leading eigen-directions as common factors and treat the residual as
uncorrelated: `Σ̂ = LLᵀ + D`. Structurally identical to a commercial risk
model, with factors extracted statistically instead of specified by an
analyst. `k = 5` is a convention, not an estimate; the number of eigenvalues
above the MP edge (6 on the first window) is the principled alternative, and
that is exactly what `marchenko_pastur_clipped` uses.

## 3. Portfolio rules

Only risk-based rules appear here. Chopra & Ziemba (1993) found errors in
expected returns matter roughly an order of magnitude more than errors in
covariances, so including return forecasts would make the covariance estimator
— the actual subject — the smaller term.

**Unconstrained minimum variance** is the diagnostic instrument. Nothing stops
the optimiser reaching into the noisy directions, so estimator quality shows
up undiluted. It is not a tradable portfolio at 411% short exposure, and is
not presented as one.

**Long-only minimum variance** is solved by accelerated projected gradient
(FISTA) with exact simplex projection, step size `1/λ_max(Σ)`. Roughly 40
lines, no solver dependency, and validated against SLSQP in the tests.

**Equal risk contribution** uses cyclical coordinate descent on

```
min  ½ wᵀΣw − (1/p) Σ log wᵢ,   w > 0
```

which is strictly convex, so each coordinate update is the positive root of a
quadratic. The tests confirm the risk contributions come out equal to 1e−6,
and that it reduces to inverse-volatility weighting when correlations vanish.

## 4. Backtest protocol

- **Estimation window**: rolling, default 250 days (about one year).
- **Rebalance**: every 21 trading days. Weights held fixed between.
- **Information set**: weights for the period starting at `t` use rows
  `[t − window, t)` only.
- **Costs**: linear, charged on one-way traded notional at each rebalance.
- **Metrics**: annualised volatility is the primary one — these are
  minimum-variance portfolios, so it is what they optimise. Sharpe ratios are
  reported but depend on realised returns, which nothing here forecasts.

Look-ahead is checked mechanically. `test_every_rebalance_uses_only_its_own_
trailing_window` perturbs one day before a rebalance and asserts the weights
change, then perturbs one day after and asserts they do not.

That test caught a real problem during development. Its first version used
independent Gaussian returns, on which Ledoit–Wolf shrinks all the way to a
scaled identity — whose minimum-variance portfolio is `1/N` regardless of the
data. The weights were constant, so the test passed while detecting nothing.
The fixture now generates returns with genuine factor structure. **A passing
test on data that cannot express the failure is not evidence of anything.**

## 5. Statistical inference: how much of the comparison is noise?

`src/robustcov/inference.py`, exercised by `scripts/07_significance.py`.

**Bootstrap confidence intervals.** Each estimator's out-of-sample return
series is resampled via a *circular block bootstrap* (Kunsch, 1989) rather
than an ordinary iid bootstrap: daily returns are approximately uncorrelated
in mean but not in variance (volatility clustering), so resampling single
days independently would understate the true sampling variance of any
path-dependent statistic. Blocks of consecutive days (default length
`n^(1/3)`, the standard consistency rate for block bootstraps of dependent
data) are drawn with wraparound at the series' end and concatenated back to
the original length; percentiles of the resulting distribution of, e.g.,
the Sharpe ratio give the confidence interval. This resamples the *realised
return path*, not the estimation procedure — it answers "how much would
this statistic vary across alternative histories the market could
plausibly have drawn", not "how much would the weights vary under a
different covariance estimate".

**The deflated Sharpe ratio** (Bailey & Lopez de Prado, 2014) addresses a
different problem: five estimators were compared and the best one's Sharpe
ratio reported, but the best of five noisy draws looks good even if every
single one has zero true skill. The test statistic

```
z = (SR_hat − SR_0) * sqrt(T − 1) / sqrt(1 − γ₃·SR_hat + (γ₄−1)/4 · SR_hat²)
```

compares the observed Sharpe ratio `SR_hat` against `SR_0`, the Sharpe ratio
the best of `N` *worthless* strategies would be expected to show by pure
chance (from the classical extreme-value approximation for the maximum of
`N` correlated Gaussians), while `γ₃` and `γ₄` (skewness and kurtosis of the
actual returns) correct for the fact that financial returns are not
Gaussian. With `γ₃ = 0` and `γ₄ = 3`, the denominator reduces to
Lo (2002)'s `1 + SR_hat²/2` — the Sharpe ratio estimator's own variance
depends on the true Sharpe ratio, since dividing by an estimated standard
deviation adds variance beyond what the mean alone would contribute.
`DSR = Φ(z)` is the resulting probability that the true Sharpe ratio
exceeds the chance benchmark.

Applied to this project's own results: the long-only comparison's best
performer (LW-identity, SR=0.76) has a deflated Sharpe ratio of 0.87 against
a chance benchmark of 0.18 for the best of five trials — a real edge is
plausible, but not established at the conventional 0.95 threshold. The
unconstrained comparison's best performer (PCA-5factor, SR=1.26 against a
chance benchmark of 0.28) clears it at 0.975. See the README's "Statistical
significance" section for the full tables.

## 6. Threats to validity

| Threat | Effect | Handling |
|---|---|---|
| Survivorship bias | Inflates returns | Documented; volatility comparison is far less affected since all estimators share the universe |
| Single regime, single dimensionality | Rankings may not hold under stress *at high `p`* | `scripts/05_crisis_robustness.py` confirms the phase-transition *mechanism* replicates independently in the 2008 crisis on a second dataset; it cannot confirm the specific four-estimator *ranking* under stress at high `p`, since the crisis panel only has 23 assets |
| No dividends | Understates returns ≈2%/yr | Immaterial for daily covariance |
| Multiple comparisons | Five estimators, one dataset | Quantified via `scripts/07_significance.py`: bootstrap 95% CIs on every Sharpe ratio cross zero; the long-only comparison's best performer has a deflated Sharpe ratio of 0.87 (Bailey & Lopez de Prado, 2014), not distinguishable from the best of five random strategies at the conventional 0.95 threshold. Volatility CIs are much tighter and remain the trustworthy comparison |
| No market impact | Understates the cost of high-turnover strategies | Linear costs only; the sample estimator's 8× monthly turnover would fare worse still |
| Data cleaning discretion | Excluding tickers is a choice | Every exclusion is listed by name and date in the cleaning report, for both datasets |

The single most important caveat: **this is one realisation of one market over
five years.** The mechanism — noisy eigenvalues, amplified by inversion — is
general and provable, and Section 5 above (the 2008 dataset) is direct
evidence that it is not an artefact of this specific five-year sample: the
same blow-up at `q=1` reappears on 23 different stocks, a different decade,
and a genuine crash. What is *not* independently confirmed is the specific
ordering of four regularised estimators separated by half a percentage point
of volatility — that comparison has only ever been run once, at high `p`, on
the 2012–17 panel.

## 7. What would strengthen it

In rough order of value added:

1. **A high-`p` dataset that also spans a crisis.** The current crisis check
   proves the *mechanism* generalises but, at `p=23`, can't test whether the
   *ranking* between estimators changes under stress at the dimensionality
   where it actually matters. This is the single most valuable next step.
2. **Nonlinear shrinkage** (Ledoit & Wolf, 2017), which shrinks each
   eigenvalue by a different optimal amount rather than applying one intensity
   to the whole matrix. Current state of the art.
3. **Time-varying covariance** (DCC-GARCH, or EWMA weighting inside the
   window), which addresses a limitation shared by every estimator here: they
   all weight a return from a year ago equally with yesterday's.
4. ~~Bootstrap confidence intervals on the comparison table~~ — done
   (`scripts/07_significance.py`, `src/robustcov/inference.py`); see Section 5.
