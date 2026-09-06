# The Mathematics Behind This Project

A ground-up derivation of everything in `src/robustcov/`, written for someone
who knows basic linear algebra and probability but hasn't seen these ideas
combined before. Each section builds on the last. Code references point to
the exact function implementing the math.

---

## 1. Why a covariance matrix is the object that matters

You hold a portfolio of `p` assets with weights `w = (w_1, ..., w_p)`,
`Σw_i = 1`. Asset `i` has daily return `r_i`, a random variable. The
portfolio's return is the weighted sum `R = w'r = Σ w_i r_i`.

Two things you might want to know about `R`: its expected value, and its
variance. The expectation is easy — it's linear:

```
E[R] = E[w'r] = w'E[r] = w'μ
```

where `μ = E[r]` is the vector of expected returns. The variance is where
correlations enter, and it's worth deriving fully rather than quoting:

```
Var(R) = E[(R - E[R])²] = E[(w'(r - μ))²] = E[w'(r-μ)(r-μ)'w]
       = w' E[(r-μ)(r-μ)'] w = w' Σ w
```

where `Σ = E[(r-μ)(r-μ)']` is the covariance matrix — a `p×p` matrix whose
entry `Σ_ij = Cov(r_i, r_j)` (and `Σ_ii = Var(r_i)`). The quantity `w'Σw` is
called a **quadratic form**, and it is the single most important expression
in this whole project: portfolio risk is a quadratic form in the covariance
matrix, and everything downstream — optimization, estimation error,
inversion — is about that one object.

**Why it must be positive semi-definite (PSD).** Variance can never be
negative, and `Var(R) = w'Σw` for *every possible* weight vector `w`
(including ones that don't sum to 1, if you allow arbitrary linear
combinations of the assets). So we need `w'Σw ≥ 0` for all `w` — that
condition, by definition, is what "positive semi-definite" means. Any
covariance matrix estimator you build had better output a PSD matrix, or it's
claiming some portfolio has negative risk. `is_psd()` in
`src/robustcov/estimators.py` checks exactly this.

---

## 2. Eigenvalues: what a covariance matrix actually *is*

Every real symmetric matrix `Σ` (and covariance matrices are always
symmetric, since `Cov(r_i,r_j) = Cov(r_j,r_i)`) can be written via the
**spectral decomposition**:

```
Σ = V Λ V'
```

where `Λ = diag(λ_1, ..., λ_p)` are the eigenvalues and the columns of `V`
are the corresponding eigenvectors, which form an orthonormal basis
(`V'V = I`). This is not a computational trick — it's a change of
coordinates. If you rotate your `p` assets into the eigenvector basis (i.e.
you hold portfolios `v_1, ..., v_p` instead of the original assets), those
rotated portfolios are **uncorrelated with each other**, and portfolio `v_k`
has variance exactly `λ_k`.

This gives eigenvalues a direct financial meaning: **each eigenvalue is the
variance of one particular, uncorrelated combination of your assets.** A
covariance matrix's eigenvalues are never negative (that's equivalent to PSD)
— a negative eigenvalue would mean some rotated portfolio has negative
variance, which is nonsense.

**Two facts you'll use constantly:**

- `trace(Σ) = Σ λ_k` — the sum of eigenvalues equals the sum of the
  individual variances (both equal `Σ w'Σw` traced out, or just: trace is
  basis-independent). This is "total variance in the system."
- The **condition number** `κ(Σ) = λ_max / λ_min` measures how badly
  `Σ` behaves under inversion. If `κ` is huge, then `Σ` has a direction
  (`λ_min`'s eigenvector) with almost no variance — a near-riskless
  combination of assets — and inverting `Σ` divides by that tiny number,
  blowing up. `condition_number()` in the code computes exactly this ratio.

---

## 3. The high-dimensional estimation problem

You never observe `Σ`. You observe `n` days of returns and compute the
**sample covariance matrix**:

```
S = (1/n) Σ_{t=1}^n (r_t - r̄)(r_t - r̄)'          [sample_covariance()]
```

This is the maximum-likelihood estimator under a Gaussian model, and it's
unbiased (up to the usual `n` vs `n-1` correction). So what's the problem?

**Count the parameters.** `Σ` has `p(p+1)/2` free entries (it's symmetric).
Your data is `n × p` numbers. For `p = 428` assets, that's 91,806 unknowns.
With `n = 250` days, you have 107,000 numbers to estimate them from — barely
more data points than parameters, and every one of those numbers is noisy
(a stock's daily return has an enormous amount of idiosyncratic randomness in
it, unlike, say, a physical measurement).

**Rank deficiency.** Here's the sharp version of the problem. `S` is built by
summing `n` rank-one matrices `(r_t - r̄)(r_t-r̄)'`. A sum of `n` rank-one
matrices has rank **at most** `n`. So if `p > n`, then `rank(S) ≤ n < p`,
meaning `S` is *exactly singular* — it has `p - n` eigenvalues that are
*exactly zero* (up to floating point). A singular matrix cannot be inverted.
Not "inverts badly" — cannot be inverted at all. `test_sample_covariance_is_
singular_when_p_exceeds_n` in the test suite checks this directly, and the
project's own data confirms it: at `p=428, n=250`, the sample correlation
matrix has 179 eigenvalues that are exactly zero.

**Even when `p < n`, the estimate is still bad**, and this is the more subtle
point that random matrix theory (next section) makes precise: the *largest*
sample eigenvalues are systematically biased upward and the *smallest* are
biased downward, relative to the truth, even though `S` is unbiased as a
whole matrix. Averaging over many noisy quantities and taking the max/min of
that average is a selection effect — like how the tallest person in a large
random sample is taller than the population's true tallest-percentile height
almost by definition. Portfolio optimization, as you'll see in Section 6,
divides by `Σ`, which means it divides by the smallest eigenvalues — exactly
the ones estimated worst.

---

## 4. Random matrix theory: what "pure noise" looks like

Here's a genuinely striking fact, and it's the theoretical anchor for the
whole eigenvalue-clipping idea.

**Setup.** Suppose your `p` assets are *actually* independent, unit-variance
noise — no correlation, no structure, nothing to estimate. You still only
have `n` observations, so your sample correlation matrix `C = (1/n) Z'Z`
(where `Z` is the standardized `n×p` data) won't be the identity matrix
exactly — it'll have spurious sample correlations purely from finite-sample
noise, and hence eigenvalues spread out above and below 1.

**The Marchenko–Pastur theorem** (1967) says that as `n, p → ∞` with the
ratio `q = p/n` held fixed, the histogram of eigenvalues of `C` converges to
a specific, known density:

```
f(x) = √((λ₊ - x)(x - λ₋)) / (2π q x),     for x ∈ [λ₋, λ₊]
λ± = (1 ± √q)²
```

(plus, if `q > 1`, a spike of probability mass `1 - 1/q` sitting exactly at
zero — this is the rank-deficiency fact from Section 3, restated
probabilistically.) `marchenko_pastur_density()` in the code implements this
formula exactly.

The astonishing part is that this law depends on **nothing except `q`**. Not
the number of assets, not the sample size individually, not any property of
the "noise" distribution beyond finite variance. It's a universal law, the
covariance-matrix analogue of the Central Limit Theorem.

**Why this matters practically.** It gives you a rigorous null hypothesis.
Take your *real* return data, compute its sample correlation eigenvalues, and
overlay the MP curve for your actual `q = p/n`. Any eigenvalue that falls
**inside** the MP band is statistically indistinguishable from what pure
noise would produce — you have no business trusting it as "real structure."
Any eigenvalue **above** the upper edge `λ₊ = (1+√q)²` is doing something no
amount of random noise could produce by chance; it is signal.

Run this on the project's data (`scripts/02_spectrum.py`) and you get a
striking number: of 428 eigenvalues, only **6 sit above the noise edge**, and
those 6 carry 44% of all the variance in the system. The other 422 directions
of the 428-dimensional return space are statistically indistinguishable from
noise. That's not a claim about this dataset being unusually bad — it's the
generic situation for large equity portfolios.

**`marchenko_pastur_clipped()`** (in `estimators.py`) operationalizes this:
diagonalize the sample correlation matrix, keep every eigenvalue above `λ₊`
untouched (it's real signal, trust it), and replace every eigenvalue below it
with a single flat value chosen so the trace (total variance) is preserved.
You're explicitly saying "these directions are noise, and the best estimate
of pure noise is that it's all the same size."

---

## 5. Shrinkage estimation: trading bias for variance

Random matrix theory tells you the sample estimate is bad. Shrinkage is a
general statistical technique — much older than its use in finance — for
fixing exactly this kind of problem, and it's worth understanding the
principle before the specific formulas.

**The bias–variance decomposition.** For any estimator `Σ̂` of the true `Σ`,
measuring quality by expected squared error,

```
E[‖Σ̂ - Σ‖²] = ‖E[Σ̂] - Σ‖²  +  E[‖Σ̂ - E[Σ̂]‖²]
             = (bias)²      +  variance
```

The sample covariance `S` is unbiased (first term is zero) but has large
variance when `p` is comparable to `n` — every one of its ~90,000 entries is
individually noisy. The idea of shrinkage is to accept *some* bias in
exchange for a large reduction in variance, if the total (the left-hand side)
comes out smaller. This is exactly the same idea as ridge regression, if
you've seen that: you shrink your OLS coefficients toward zero, accepting
bias, because it reduces variance enough to lower total prediction error when
your design matrix is ill-conditioned. A covariance matrix with `p` near `n`
is the ridge-regression problem's twin.

**Ledoit–Wolf shrinkage** (Ledoit & Wolf, 2003, 2004) makes this concrete.
Pick some structured "target" matrix `F` — something with very low variance
as an estimator (maybe even no randomness at all) but which is *wrong* in a
predictable way (has bias). Then estimate

```
Σ̂ = λF + (1-λ)S,        λ ∈ [0,1]
```

a straight convex combination. The question is: what's the best `λ`?

**Deriving the optimal shrinkage intensity.** We want to minimize
`E[‖λF + (1-λ)S - Σ‖²]` over `λ` (Frobenius norm squared, i.e. sum of squared
entries). Write `S = Σ + E` where `E` is the sample's estimation error
(mean zero, since `S` is unbiased). Then:

```
λF + (1-λ)S - Σ = λ(F - Σ) + (1-λ)(S - Σ) = λ(F-Σ) + (1-λ)E
```

Expand the squared norm and take expectation. The cross term vanishes because
`E[E] = 0` and it's (asymptotically) independent-ish of the deterministic
part:

```
E[‖·‖²] = λ² ‖F-Σ‖² + (1-λ)² E[‖E‖²]
```

This is now just a one-variable quadratic in `λ`. Let `d² = ‖F-Σ‖²` (how far
the target is from the truth) and `b² = E[‖E‖²]` (how noisy the sample
estimate is). Minimizing `λ²d² + (1-λ)²b²` by calculus (`d/dλ = 0`):

```
2λd² - 2(1-λ)b² = 0   ⟹   λ(d² + b²) = b²   ⟹   λ* = b² / (b² + d²)
```

This is the whole idea in one formula: **shrink harder (`λ*` closer to 1)
exactly when the sample estimator is noisier (`b²` large) relative to how
wrong the target is (`d²` small).** You can't observe `d² = ‖F - Σ‖²`
directly since you don't know `Σ`, but Ledoit and Wolf show `‖F-S‖²` is a
consistent estimator of it (it converges to the right thing as `n→∞`), and
`b²` can be estimated from the data's fourth moments (how variable each
individual entry of `S` is across observations) without ever needing the true
`Σ`. That's what makes the formula *usable*: `λ*` is computed entirely from
observable quantities, with no free parameter to tune and no cross-validation
needed. Compare this to how you might tune a ridge regression's penalty by
cross-validation — Ledoit-Wolf gets the analogous number in closed form.

Two choices of target are implemented, and the difference between them is a
direct lesson in *why the target matters*:

**Target 1 — scaled identity** (`ledoit_wolf_identity()`): `F = μI` where
`μ = trace(S)/p` is the average variance. This says "I have no information
about which assets are correlated with which; assume zero correlation and
equal variance." It's a weak target for stocks (they're obviously
correlated through the market), and the fitted `λ*` on this data comes out
tiny (0.056) — the algorithm correctly recognizes the target is a bad match
and refuses to lean on it much.

**Target 2 — constant correlation** (`ledoit_wolf_constant_correlation()`):
keep each asset's own sample variance, but replace every pairwise
correlation with the *average* correlation across the whole panel. This
encodes real structure — stocks genuinely are all positively correlated
through broad market movements — so it's a much better target, the fitted
`λ*` comes out far larger (0.314), and the resulting estimator performs
best in the backtest. **The lesson generalizes**: shrinkage's benefit scales
with how much genuine structure you can bake into the target; the intensity
formula automatically adapts to trust a good target more.

---

## 6. Factor models and PCA: the other way to add structure

Instead of shrinking every entry of `S` toward some target, you can impose an
explicit low-rank structure:

```
Σ̂ = LL' + D
```

where `L` is `p × k` (`k` "factor loadings," `k ≪ p`) and `D` is diagonal
("idiosyncratic" variance specific to each asset). This says: each asset's
return is driven by `k` common factors plus its own independent noise —
exactly the structure of a CAPM or Fama–French model, except here the
factors are *discovered from the data* rather than specified in advance
(market return, size, value, etc.).

**Where does `L` come from?** Take the spectral decomposition
`S = VΛV'` from Section 2, and keep only the `k` largest eigenvalues:

```
L = V_k √Λ_k          (columns = top-k eigenvectors, scaled by √eigenvalue)
```

**Why is this the *best possible* rank-`k` approximation?** This isn't a
heuristic — it's a theorem (Eckart–Young–Mirsky): among *all* rank-`k`
matrices `M`, the one minimizing `‖S - M‖²` (Frobenius norm) is exactly
`LL'` built from the top-`k` eigenvectors as above. So PCA isn't merely *a*
way to find factors, it's *the* optimal way to compress a covariance matrix
into rank `k`, in a precise mathematical sense. `pca_factor_model()`
implements this and adds back `D = diag(S) - diag(LL')` to preserve each
asset's total variance, distributing the leftover into the "idiosyncratic"
bucket.

**How many factors?** The code defaults to `k=5` as a convention, but Section
4 gives you a principled answer: use however many eigenvalues clear the
Marchenko–Pastur edge (6, on this data). Below that edge, an eigenvalue isn't
distinguishable from noise, so treating it as a "factor" is overfitting; that
choice is exactly what makes `marchenko_pastur_clipped()` and
`pca_factor_model()` closely related in spirit — both are rank-reduction
ideas, one hard (drop/flatten) and one soft (project onto a subspace).

---

## 7. Portfolio optimization: what you do with `Σ̂` once you have it

### 7.1 The unconstrained minimum-variance portfolio

You want the weights `w` minimizing risk `w'Σw` subject to `w'1 = 1` (fully
invested, no cash). This is a constrained optimization problem, solved with a
**Lagrange multiplier**:

```
L(w, γ) = w'Σw - γ(w'1 - 1)
```

Take the gradient with respect to `w` and set it to zero. (Useful identity:
`∇_w(w'Σw) = 2Σw` since `Σ` is symmetric.)

```
∂L/∂w = 2Σw - γ1 = 0   ⟹   w = (γ/2) Σ⁻¹1
```

Now use the constraint `w'1 = 1` to pin down the multiplier:

```
1'w = (γ/2) 1'Σ⁻¹1 = 1   ⟹   γ/2 = 1 / (1'Σ⁻¹1)
```

Substituting back:

```
w* = Σ⁻¹1 / (1'Σ⁻¹1)
```

That's the formula in `global_minimum_variance()` — not a guess, but the
exact closed-form solution of a quadratic program with one linear
constraint. It requires inverting `Σ`, which is precisely the operation
Section 3 showed is catastrophic when `Σ` is estimated from data with `p`
near or above `n`. This is the mathematical mechanism behind everything the
project measures: a good `Σ̂` here means a usable `w*`; a bad one means an
extreme, overleveraged `w*`.

### 7.2 Adding the no-shorting constraint

Real portfolios often can't go short: `w ≥ 0` in addition to `w'1=1`. Now
there's no clean closed form — the solution can have some weights pinned
exactly at zero, and *which* weights are zero isn't known in advance. This is
a proper **quadratic program**, and the project solves it with **projected
gradient descent**, which is worth understanding as a general tool (it shows
up everywhere in machine learning, not just here):

1. Take an ordinary gradient step: `y = w - η∇f(w) = w - η(2Σw)`.
2. Project `y` back onto the constraint set (here, the simplex
   `{w : w≥0, w'1=1}`) — find the closest point on the simplex to `y`.
3. Repeat.

Step 2's projection has a clean closed-form algorithm (sort the entries,
find a threshold, clip — implemented in `_project_simplex()`), which is why
this method is practical. The step size `η = 1/λ_max(Σ)` is not arbitrary:
`λ_max(Σ)` is the *Lipschitz constant* of the gradient `∇f(w) = 2Σw` (how
fast the gradient can change), and using its reciprocal as the step size is
the standard guarantee for gradient descent to converge without overshooting.
`min_variance_long_only()` additionally uses **Nesterov acceleration**
(the `y = w_new + ((t-1)/t_new)(w_new - w)` momentum term) — a
`O(1/k)` → `O(1/k²)` convergence speedup that costs almost nothing to
implement. The tests check this against `scipy.optimize.minimize` (SLSQP) to
confirm the hand-rolled solver reaches the same answer.

**Why does this constraint fix so much?** Jagannathan & Ma (2003) proved
something non-obvious: adding `w ≥ 0` to the minimum-variance problem is
*mathematically equivalent* to solving the unconstrained problem on a
*shrunk* covariance matrix. The no-shorting constraint does regularization
for you, automatically, which is exactly why the project's own results show
the five very different estimators converging to nearly the same answer
once shorting is banned (Section 7.4 of the README) — they'd already been
partially regularized by the constraint before the estimator ever got a
chance to matter.

### 7.3 Equal risk contribution (risk parity)

**Risk contribution.** Portfolio volatility is `σ_p = √(w'Σw)`. Because
`σ_p` is a homogeneous function of degree 1 in `w` (double every weight,
double the risk), **Euler's theorem** for homogeneous functions gives an
exact decomposition:

```
σ_p = Σ_i  w_i · (∂σ_p/∂w_i)  =  Σ_i  w_i (Σw)_i / σ_p
```

Each term `w_i(Σw)_i/σ_p` is asset `i`'s exact contribution to total risk,
and by construction they sum precisely to `σ_p` — no residual, no
approximation. That's `risk_contributions()` in the code.

**Risk parity** asks for weights making every contribution equal:
`w_i(Σw)_i = σ_p/p` for all `i`. There's no closed form for general `Σ`, but
it turns out to be the unique solution of a convex problem:

```
minimize   ½w'Σw - (1/p)Σ log(w_i)      subject to  w > 0
```

(the log-barrier term is what forces equal contributions at the optimum —
differentiate and set to zero, and you get exactly the risk-parity
condition). Because it's convex, **coordinate descent** — optimize one `w_i`
at a time, holding the others fixed, cycle until convergence — is guaranteed
to reach the global optimum. Fixing all `w_j` (`j≠i`) and solving for `w_i`
in `∂/∂w_i = 0` reduces to a single quadratic equation in `w_i`
(`equal_risk_contribution()` solves it with the quadratic formula on each
coordinate, each sweep). This lands strictly between minimum-variance and
1/N in the ordering `Var(GMV) ≤ Var(ERC) ≤ Var(1/N)` — checked in the tests —
which makes intuitive sense: it's less concentrated than the pure risk
minimizer, but still uses the covariance structure, unlike 1/N which ignores
it entirely.

### 7.4 Why no expected returns anywhere?

Everything above uses only `Σ`, never `μ = E[r]`. This is deliberate, and
there's a specific, quantified reason (Chopra & Ziemba, 1993, sometimes
called "Markowitz's error-maximizing property"): full mean-variance
optimization, `max w'μ - (κ/2)w'Σw`, is far more sensitive to errors in `μ`
than to errors in `Σ`. Intuitively, whatever small estimation error you have
in `μ` gets *scaled up* by the optimizer, because the optimal `w` for
mean-variance is `w* ∝ Σ⁻¹μ` — the same ill-conditioned inverse from Section
7.1, now multiplying an *already noisy* vector `μ` instead of the much more
stable vector `1`. Estimating `μ` accurately from historical returns is
notoriously close to hopeless (a stock's average daily return over 5 years
has a standard error on the same order as the mean itself). By dropping `μ`
entirely and using only risk-based rules, the project isolates the question
it's actually trying to answer — does a better `Σ̂` help — from a much larger
and separate source of error that would otherwise swamp it.

---

## 8. How the pieces fit together

| Section | Concept | Code |
|---|---|---|
| 1–2 | Covariance as a quadratic form; eigenvalues as risk of uncorrelated portfolios | `condition_number`, `is_psd` |
| 3 | Why `S` fails when `p` is comparable to `n` | `sample_covariance` |
| 4 | Marchenko–Pastur: what noise looks like, so you can subtract it | `marchenko_pastur_clipped`, `marchenko_pastur_density` |
| 5 | Shrinkage: bias/variance tradeoff, closed-form optimal intensity | `ledoit_wolf_identity`, `ledoit_wolf_constant_correlation` |
| 6 | Low-rank factor structure, optimal by Eckart–Young | `pca_factor_model` |
| 7.1 | Lagrangian derivation of `w = Σ⁻¹1/(1'Σ⁻¹1)` | `global_minimum_variance` |
| 7.2 | Projected gradient / FISTA for the constrained QP | `min_variance_long_only`, `_project_simplex` |
| 7.3 | Euler decomposition of risk; coordinate descent | `equal_risk_contribution`, `risk_contributions` |
| 7.4 | Why `μ` is excluded | (a design choice, not a function) |

If you want to go one level deeper on any single piece — actually working
the Ledoit–Wolf variance-of-entries calculation by hand, or deriving the MP
law's Stieltjes-transform proof, or proving Eckart–Young — say which one and
we can do just that, slowly, with the numbers from this dataset as the
running example.

## References (full derivations, if you want primary sources)

- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices.* J. Multivariate Analysis 88(2).
- Ledoit, O. & Wolf, M. (2003). *Improved estimation of the covariance
  matrix of stock returns.* J. Empirical Finance 10(5).
- Marchenko, V. & Pastur, L. (1967). *Distribution of eigenvalues for some
  sets of random matrices.* Mat. Sb. — the original theorem.
- Laloux, Cizeau, Potters, Bouchaud (2000). *Random matrix theory and
  financial correlations.* IJTAF 3(3) — the finance application.
- Jagannathan, R. & Ma, T. (2003). *Risk reduction in large portfolios: why
  imposing the wrong constraints helps.* J. Finance 58(4).
- Chopra, V. & Ziemba, W. (1993). *The effect of errors in means, variances,
  and covariances on optimal portfolio choice.* J. Portfolio Management.
- Griveau-Billion, Richard, Roncalli (2013). *A fast algorithm for computing
  high-dimensional risk parity portfolios* — the coordinate-descent method.
