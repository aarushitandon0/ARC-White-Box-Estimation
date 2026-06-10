# ARC Whitebox Estimation 

A study in estimating per-neuron expected activations in randomly initialized deep ReLU networks,
under a hardware-enforced FLOP budget. Three estimators are implemented and compared: pure
analytical covariance propagation, Monte Carlo with budget allocation, and antithetic variates.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Network Architecture](#network-architecture)
- [The flopscope Constraint](#the-flopscope-constraint)
- [Estimator 1: Covariance Propagation](#estimator-1-covariance-propagation)
- [Estimator 2: Monte Carlo with Budget Allocation](#estimator-2-monte-carlo-with-budget-allocation)
- [Estimator 3: Antithetic Variates](#estimator-3-antithetic-variates)
- [Benchmark Results](#benchmark-results)
- [FLOP Budget Analysis](#flop-budget-analysis)
- [Why the Gain Approximation Accumulates Bias](#why-the-gain-approximation-accumulates-bias)
- [Potential Improvements](#potential-improvements)
- [Repository Structure](#repository-structure)
- [Reproducing Results](#reproducing-results)

---

## Problem Statement

Given the weight matrices of a randomly initialized ReLU MLP, compute the matrix:

```
M[ell, i]  =  E[ h_i^(ell)(X) ]

    where   X ~ N(0, I_n)              (standard Gaussian input)
            h^(ell)  is the activation vector at layer ell
            i ranges over all neurons {1, ..., 256}
            ell ranges over all layers {1, ..., 8}
```

The output shape is `(8, 256)`. The true values are computed by the evaluator using a
high-sample Monte Carlo ground truth (`n = 1,000,000` samples). Estimator quality is
measured by MSE on the final layer, penalized by how much of the FLOP budget was used:

```
score  =  MSE_final  *  max(0.1,  C_used / B_total)

    B_total  =  6.8e10 FLOPs
    C_used   =  FLOPs charged by flopscope during predict()

Lower score is better.
```

The `max(0.1, ...)` term means: if you use less than 10% of the budget, you still pay 10% of
the MSE (a 10x discount, not a free pass). At large sample counts, MSE dominates and spending
the full budget always wins.

---

## Network Architecture

All networks in this challenge share one fixed architecture:

```
    width  = 256 neurons per layer
    depth  = 8   hidden layers
    init   = He-Gaussian: W_ij ~ N(0,  2 / fan_in)
```

The forward pass uses row-vector convention (batch along axis 0):

```
    Input:      X   shape (batch, 256)    X ~ N(0, I_256)
    Layer ell:  h^(ell) = ReLU( h^(ell-1) @ W_ell )
    Output:     h^(8)   shape (batch, 256)


    Layer 0 (input)          Layer 1                  Layer 8 (output)
    +-----------+            +-----------+            +-----------+
    | x_1       |            | h_1^(1)   |            | h_1^(8)   |
    | x_2       |  @ W_1 --> | h_2^(1)   |  ...  --> | h_2^(8)   |
    |   ...     |   ReLU     |   ...     |   ReLU     |   ...     |
    | x_256     |            | h_256^(1) |            | h_256^(8) |
    +-----------+            +-----------+            +-----------+
      N(0, I)                 truncated                 deeply
                              Gaussian                  non-Gaussian
```

He-Gaussian initialization is designed so that the variance of each pre-activation is
approximately 2 at layer 1 and stays near that value in expectation:

```
    Pre-activation at layer 1:
        z_i^(1) = sum_{j=1}^{256}  W_{ji}  *  x_j
        Var[z_i^(1)] = sum_j  Var[W_{ji}] * Var[x_j]  =  256 * (2/256) * 1  =  2

    The factor 2/n in He init is chosen so ReLU (which kills half the variance)
    leaves the signal magnitude roughly stable layer to layer.
```

Weight matrices are stored as `mlp.weights`, a Python list of 8 arrays each shaped `(256, 256)`.

---

## The flopscope Constraint

All computation inside `predict()` must go through `flopscope.numpy` (imported as `fnp`).
The library intercepts every array operation, tallies the FLOP cost, and raises
`BudgetExhaustedError` if the running total exceeds the budget. Uninstrumented code
(plain NumPy, Python loops over scalars, etc.) goes uncounted but is penalized through
a separate "uninstrumented ratio" metric.

### Operation Costs

```
    Operation                              FLOP cost
    ------------------------------------------------
    fnp.random.standard_normal(shape)      16 per element
    fnp.matmul((M,N), (N,K))               2 * M * N * K       <- both multiply AND add counted
    fnp.maximum(a, b)  pointwise           1 per element
    fnp.mean(a, axis=0)                    1 per element
    fnp.sqrt(a)  /  fnp.exp(a)            1 per element
    fnp.outer(u, v)  shape (n, n)          ~n*(n+1)/2          <- symmetric tag
    fnp.einsum("ij,ia,jb->ab", ...)        2 * n^3             (two contractions)
    flops.stats.norm.pdf / .cdf            measured per call
```

The 2x matmul cost is the most consequential budget constraint. A forward pass of a
`(K, 256)` batch through one `(256, 256)` weight matrix costs:

```
    2 * K * 256 * 256  =  131,072 * K  FLOPs
```

### BudgetContext Lifecycle

```
    setup(ctx)          <-- runs OUTSIDE BudgetContext. Free computation.
                            ctx.seed, ctx.width, ctx.depth, ctx.flop_budget available.

    predict(mlp, budget)  <-- runs INSIDE BudgetContext.
                              Every fnp.* operation is metered.
                              Exceeding budget raises BudgetExhaustedError
                              (predictions become zeros if that happens).

    teardown()          <-- runs OUTSIDE BudgetContext. Free.
```

Any expensive precomputation (loading lookup tables, fitting distributions, etc.) should
go in `setup()`. Only the inference path belongs in `predict()`.

---

## Estimator 1: Covariance Propagation

**File:** `estimator_covariance.py`
**FLOPs:** ~404 million
**C/B:** 0.006 (score multiplier capped at 0.1)

### Core Idea

For a Gaussian input, the first-layer pre-activations are exactly Gaussian. The ReLU
output is a truncated Gaussian. After the next linear layer, by the CLT-like aggregation
of 256 truncated Gaussians, the distribution is approximately Gaussian again.

This estimator tracks the distribution as `(mu, Sigma)` (mean vector and full covariance
matrix) through each layer, using closed-form formulas for the Gaussian-to-truncated-Gaussian
moment transformation at each ReLU.

### Layer Propagation

```
    State:  (mu, Sigma)   mu in R^256,  Sigma in R^(256x256)

    +--------------------- LINEAR STEP ---------------------+
    |                                                        |
    |   mu_pre    =  W^T @ mu                               |
    |                                                        |
    |   Sigma_pre =  W^T @ Sigma @ W                        |
    |              = einsum("ij,ia,jb->ab", Sigma, W, W)    |
    |                (preserves symmetry tag in flopscope)  |
    |                                                        |
    +--------------------------------------------------------+
                             |
                             v
    +------------------- RELU STEP -------------------------+
    |                                                        |
    |   For each neuron i:                                   |
    |     alpha_i  = mu_pre_i / sigma_pre_i                 |
    |     phi_i    = phi(alpha_i)    (standard normal PDF)  |
    |     Phi_i    = Phi(alpha_i)    (standard normal CDF)  |
    |                                                        |
    |   Exact marginals:                                     |
    |     mu_post_i  = mu_pre_i * Phi_i  +  sigma_i * phi_i |
    |     E[z_i^2]   = (mu_i^2 + var_i)*Phi_i + mu_i*sig*phi|
    |     var_post_i = E[z_i^2] - mu_post_i^2               |
    |                                                        |
    |   Off-diagonal (APPROXIMATE):                         |
    |     Sigma_post[i,j] = Phi_i * Phi_j * Sigma_pre[i,j]  |
    |     diagonal overwritten with exact var_post_i        |
    |                                                        |
    +--------------------------------------------------------+
```

The off-diagonal formula is known as the "gain approximation". It is exact when all
pre-activation means are zero (as at the first layer) and accumulates error as means
shift away from zero in later layers.

### Overflow Guard

When networks are deep and variances compound, the covariance diagonal can overflow float32.
A rescaling guard is applied before each layer:

```python
    max_var = float(fnp.max(fnp.diag(cov)))
    if max_var > 1e30:
        s = float(fnp.sqrt(max_var))
        mu  = mu  / s
        cov = cov / (s * s)
        log_scale += float(fnp.log(s))
```

The tracked `log_scale` is used to rescale the output means at the end of each layer.

### FLOP Cost Per Layer

```
    Operation                              FLOPs
    -----------------------------------------------
    w.T @ mu               (256,256)@(256,)     131,072
    einsum (Sigma,W,W)     2 * 256^3         33,554,432
    diag, sqrt, alpha      ~3 * 256                 768
    pdf + cdf              ~2 * 256 * C_pdf       ~8000
    outer(gain, gain)      ~256*(256+1)/2       32,896
    multiply(outer, cov)   256 * 256           65,536
    fill_diagonal          256                    256
    -----------------------------------------------
    Total per layer:                       ~33,800,000
    Total 8 layers:                       ~270,400,000

    (Measured by flopscope: ~50.5M per layer due to higher
     pdf/cdf costs; ~404M total for 8 layers.)
```

---

## Estimator 2: Monte Carlo with Budget Allocation

**File:** `estimator_hybrid.py`
**FLOPs:** ~61.1 billion
**C/B:** 0.899

### Core Idea

Run covariance propagation first (cheap, ~400M FLOPs), then spend the remaining 67.6B FLOPs
on a large batch of MC forward passes. The MC mean is unbiased. At K ~ 57,000 samples the
MC variance is far lower than the analytical bias, making MC the dominant estimator.

### Sample Count Derivation

```
    Budget:             B  =  6.8e10
    Cov prop cost:      C_analytical  =  depth * 51e6  =  408e6
    Remaining:          R  =  B - C_analytical  =  67,592e6

    FLOPs per one MC sample in a batch of K:
        sampling:     16 * 256                    =    4,096
        per layer:    2 * 256 * 256  (matmul)
                    + 256            (ReLU max)
                    + 256            (mean)
                    =  132,608
        all 8 layers: 8 * 132,608                = 1,060,864
        total:                                   = 1,064,960  (~1.06M FLOPs/sample)

    K  =  floor(0.90 * R / flops_per_sample)
       =  floor(0.90 * 67,592e6 / 1,064,960)
       ~  57,100 samples
```

The 0.90 factor leaves a 10% buffer so the actual flopscope count does not overshoot due
to rounding or constant-overhead operations.

### Forward Pass

```
    X  ~  N(0, I)    shape (K, 256)    K ~ 57,000

    h = X
    for ell in range(8):
        h = ReLU( h @ W_ell )          shape (K, 256)
        output[ell] = mean(h, axis=0)  shape (256,)

    return stack(output)               shape (8, 256)
```

---

## Estimator 3: Antithetic Variates

**File:** `estimator_control_variate.py`
**FLOPs:** ~61.1 billion
**C/B:** 0.899

### Core Idea

For any function f and X ~ N(0, I), the input -X is also distributed as N(0, I) by
symmetry of the standard Gaussian. Running both +x_k and -x_k through the network and
averaging gives an unbiased estimator with lower variance than using two independent draws.

```
                  +x_k  -->  [W_1,...,W_8]  -->  ReLU activations  -->  f(+x_k)
                  /                                                         \
    x_k ~ N(0,I)                                                              average
                  \                                                         /
                  -x_k  -->  [W_1,...,W_8]  -->  ReLU activations  -->  f(-x_k)

    Antithetic estimate:  mu_k  =  (f(+x_k) + f(-x_k)) / 2
```

### Variance Reduction Analysis

```
    Let  rho  =  Cor[ f_i(x),  f_i(-x) ]  for a fixed neuron i

    Var[ (f(x) + f(-x)) / 2 ]
        =  (1/4) * ( Var[f(x)] + Var[f(-x)] + 2*Cov[f(x), f(-x)] )
        =  (1/4) * ( 2*Var[f] + 2*rho*Var[f] )
        =  Var[f] * (1 + rho) / 2

    Averaging over K_pairs antithetic pairs:
        Var[mean]  =  Var[f] * (1 + rho) / (2 * K_pairs)

    Raw MC with the same 2*K_pairs samples (all independent):
        Var[mean]  =  Var[f] / (2 * K_pairs)

    Variance reduction ratio:  (1 + rho)
```

Measured on this architecture (width=256, depth=8, He init):

```
    rho  ~  -0.16
    Variance reduction:  1 + rho  =  0.84   (16% lower MSE than raw MC)
```

The negative correlation arises because the ReLU network partially "inverts" the sign of
the input. When a neuron is active for +x (pre-activation > 0), its mirrored counterpart
with -x tends to be inactive, creating a negative correlation between the two forward pass
outputs.

### Sample Count Derivation

```
    FLOPs per antithetic pair:
        sampling +x_k:               16 * 256     =    4,096
        forward pass for +x_k:       8 * (2*256^2 + 256 + 256)  =  1,064,960
        forward pass for -x_k:       same as above               =  1,064,960
        (-x_k costs 0 for negation; it reuses -X already in memory)
        -------------------------------------------------------
        total per pair:                                          ~  2,134,016

    K_pairs  =  floor(0.90 * 67,592e6 / 2,134,016)
             ~  28,500 pairs
             (equivalent information: 28,500 * 0.84 / 0.5  ~  47,880 independent samples)
```

### Per-Layer Output

```
    X  ~  N(0, I)    shape (K_pairs, 256)

    h_pos = X;   h_neg = -X

    for ell in range(8):
        h_pos = ReLU( h_pos @ W_ell )
        h_neg = ReLU( h_neg @ W_ell )
        output[ell] = ( mean(h_pos, axis=0) + mean(h_neg, axis=0) ) / 2

    return stack(output)    shape (8, 256)
```

---

## Benchmark Results

Reference ground truth: 1,000,000 independent MC samples, seed=0.
Architecture: width=256, depth=8.

```
    Estimator                  FLOPs Used     C / B    MSE (layer 8)    Score
    --------------------------------------------------------------------------
    Mean propagation           2.7M           0.000    1.04e-03         1.04e-04
    (diagonal only, no cov)

    Covariance propagation     404M           0.006    5.06e-05         5.06e-06
    (full 256x256 Sigma)

    Hybrid: cov + raw MC       61.1B          0.899    5.39e-06         4.85e-06
    (K ~ 57,100 samples)

    Antithetic variates        61.1B          0.899    4.82e-06         4.34e-06  <-- best
    (K_pairs ~ 28,500 pairs)
    --------------------------------------------------------------------------
```

Antithetic variates improve over raw MC by 11% in final score, and over baseline by 24x.

The covariance-only estimator has MSE ~10x higher than MC. Even with the 10x score discount
it cannot compete: `5.06e-5 * 0.1 = 5.06e-6` vs `4.82e-6 * 0.9 = 4.34e-6`. The
systematic bias from the gain approximation is the bottleneck, not sample variance.

---

## FLOP Budget Analysis

```
    Total budget:  6.8e10 FLOPs
    |
    +-- Covariance propagation:   ~404M    (0.6%)
    |     8 layers * ~50.5M/layer
    |     dominated by einsum over 256x256 covariance matrix
    |
    +-- Antithetic MC:            ~60.7B   (89.3%)
    |     |
    |     +-- Input sampling:     ~116M    (0.2%)
    |     |     16 * 256 * K_pairs
    |     |
    |     +-- Forward pass (+x):  ~30.3B   (44.5%)
    |     |     8 layers * 2*256^2*K_pairs
    |     |
    |     +-- Forward pass (-x):  ~30.3B   (44.5%)
    |     |     identical cost
    |     |
    |     +-- Layer means x2:     ~0.9B    (1.3%)
    |           2 * 8 * 256 * K_pairs
    |
    +-- Buffer (unused):          ~696M    (1.0%)
          10% safety margin for flopscope constant overhead
```

The budget is almost entirely consumed by the two forward passes. The 90% fill target
ensures the actual `flopscope.stats()` value lands at C/B ~ 0.89-0.90, staying in the
full-budget scoring regime.

---

## Why the Gain Approximation Accumulates Bias

The exact off-diagonal covariance after a ReLU is given by the bivariate normal formula:

```
    Cov[ ReLU(z_i),  ReLU(z_j) ]  =  Sigma_ij * Phi_2(0, 0; rho_ij)

    where  Phi_2  is the standard bivariate normal CDF
           rho_ij  =  Sigma_ij / (sigma_i * sigma_j)  is the pre-activation correlation
```

The gain approximation substitutes this with:

```
    Cov_approx[ ReLU(z_i),  ReLU(z_j) ]  =  Phi(alpha_i) * Phi(alpha_j) * Sigma_ij

    where  alpha_i  =  mu_i / sigma_i
```

This approximation is exact when all pre-activation means are zero (`alpha_i = 0`,
`Phi(0) = 0.5`). At layer 1, X ~ N(0, I) so `mu^(0) = 0` and the formula is exact.

After the first ReLU, however:
```
    mu^(1)  =  E[ ReLU(W^T X) ]  >  0  componentwise
```

The means are positive because ReLU zero-clips the negative half. From layer 2 onward,
every subsequent gain approximation uses a non-zero alpha, introducing systematic
underestimation of the off-diagonal covariance. Over 8 layers, this bias compounds
and becomes the dominant source of error in `estimator_covariance.py`.

---

## Potential Improvements

### 1. Exact Bivariate Covariance

Replace the gain approximation with the exact formula using `Phi_2`. The two-argument
bivariate CDF can be computed via Gauss-Legendre quadrature or series expansion. This
would eliminate the systematic bias, potentially making the analytical estimator competitive
after fewer MC samples.

### 2. Optimal Linear Blending

Given the unbiased antithetic estimate `mu_anti` and biased analytical estimate `mu_cov`,
the MSE-minimizing linear combination is:

```
    mu_final  =  alpha * mu_cov  +  (1 - alpha) * mu_anti

    alpha_opt  =  Var[anti] / ( Bias[cov]^2  +  Var[anti] )

    With current empirical values:
        Bias[cov]^2  ~  5.06e-5  (per neuron, final layer)
        Var[anti]    ~  4.82e-6  (per neuron, final layer)
        alpha_opt    ~  0.087

    Expected MSE improvement over pure antithetic:  ~8%
    (approximate, depends on bias/variance ratio per network)
```

### 3. Quasi-Monte Carlo

Replacing i.i.d. standard normal samples with a scrambled Sobol or Halton sequence
transformed to normal (via inverse CDF) typically reduces MC variance by an additional
factor proportional to `(log K)^d / K` vs `1/K`. Whether flopscope instruments the
inverse CDF at acceptable cost is an open question.

### 4. Exact arcsin Formula for Zero-Mean Networks

For zero-mean pre-activations, the bivariate normal CDF simplifies to an analytic form:

```
    Phi_2(0, 0; rho)  =  (1/4)  +  arcsin(rho) / (2*pi)
```

This can be evaluated exactly without numerical quadrature and costs only `arcsin` per
pair, which is `O(n^2)` over all pairs. The first layer qualifies (input is zero-mean),
and later layers are approximately zero-mean depending on the network state.

---

## Repository Structure

```
whest-starterkit/
    ARC-White-Box-Estimation/
        README.md                    <- this file
        estimator_covariance.py      <- analytical estimator (covariance propagation)
        estimator_hybrid.py          <- cov prop + raw Monte Carlo
        estimator_control_variate.py <- cov prop + antithetic MC (best)

    estimator.py                     <- submission entry point
    local_engine.py                  <- local benchmarking harness
    examples/
        01_random.py                 <- zero baseline
        02_mean_propagation.py       <- diagonal mean propagation
        03_covariance_propagation.py <- full covariance baseline
    docs/                            <- challenge documentation
    pyproject.toml
```

---

## Reproducing Results

Requires Python 3.10+ and `uv`.

```bash
    git clone <this-repo>
    cd whest-starterkit
    uv sync
```

Run the antithetic estimator locally (builds an MLP at seed=0 and prints an MSE table):

```bash
    uv run python ARC-White-Box-Estimation/estimator_control_variate.py
```

Run all three estimators for comparison:

```bash
    uv run python ARC-White-Box-Estimation/estimator_covariance.py
    uv run python ARC-White-Box-Estimation/estimator_hybrid.py
    uv run python ARC-White-Box-Estimation/estimator_control_variate.py
```

Each script outputs a convergence table comparing the estimator's predictions against
increasing MC sample counts, with per-layer MSE and total FLOPs used.

To run against the public mini split (100 MLPs, requires network):

```bash
    cp ARC-White-Box-Estimation/estimator_control_variate.py estimator.py
    uv run whest run \
        --estimator estimator.py \
        --dataset hf://aicrowd/arc-whestbench-public-2026 \
        --split mini \
        --runner local
```
