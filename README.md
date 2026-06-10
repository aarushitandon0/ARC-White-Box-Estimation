# ARC Whitebox Estimation Challenge 2026 (WhestBench)

This repository contains my solution to the [ARC Whitebox Estimation Challenge 2026](https://www.aicrowd.com/) hosted on AIcrowd. The task is to estimate the expected post-ReLU activation of every neuron in every layer of a randomly initialized MLP, subject to a strict FLOP budget enforced at runtime by the `flopscope` library.

---

## Challenge Description

Given a randomly initialized ReLU MLP with 8 hidden layers and 256 neurons per layer (He-Gaussian initialization, variance 2/n), estimate the per-neuron expected activation:

```
E[h_i^(ell)(X)]  for all i in {1,...,256}, ell in {1,...,8}
```

where inputs X are drawn from a standard Gaussian distribution N(0, I_n).

The output is a matrix of shape (8, 256) containing the estimated means.

### Scoring

```
score = MSE_final_layer * max(0.1, C_used / B_total)
```

where `B_total = 6.8e10` FLOPs. Using less than 10% of the budget applies a 10x discount to the MSE penalty, but the penalty itself dominates at large sample counts. Lower is better.

All array operations inside `predict()` must use `flopscope.numpy` (aliased as `fnp`) so that every operation is tracked and counted. The budget is enforced at runtime: exceeding it raises `BudgetExhaustedError` and zeroes the predictions.

---

## Files

```
whest-starterkit/
    estimator.py                  -- AIcrowd submission entry point (copy best estimator here)
    estimator_covariance.py       -- Pure covariance propagation (analytical, ~400M FLOPs)
    estimator_hybrid.py           -- Covariance propagation + raw Monte Carlo (~61B FLOPs)
    estimator_control_variate.py  -- Covariance propagation + antithetic MC (best score)
    local_engine.py               -- Local benchmarking harness (unchanged from starter kit)
    examples/
        02_mean_propagation.py    -- Diagonal mean propagation baseline
        03_covariance_propagation.py  -- Full covariance baseline
    pyproject.toml
```

---

## Approach

### 1. Covariance Propagation (`estimator_covariance.py`)

Tracks the full 256x256 covariance matrix through each layer analytically.

**Linear layer:** Given input with mean mu and covariance Sigma, after multiplication by W:

```
mu_pre  = W^T @ mu
cov_pre = W^T @ Sigma @ W
```

Implemented via `fnp.einsum("ij,ia,jb->ab", cov, w, w)` to preserve the symmetry tag in flopscope and avoid a `SymmetryLossWarning`.

**ReLU layer:** For a Gaussian pre-activation z_i ~ N(mu_i, sigma_i^2), the exact marginal statistics are:

```
alpha_i     = mu_i / sigma_i
phi(alpha)  = standard normal PDF
Phi(alpha)  = standard normal CDF

E[ReLU(z_i)]     = mu_i * Phi(alpha_i) + sigma_i * phi(alpha_i)
E[ReLU(z_i)^2]   = (mu_i^2 + sigma_i^2) * Phi(alpha_i) + mu_i * sigma_i * phi(alpha_i)
Var[ReLU(z_i)]   = E[z_i^2] * Phi(alpha) - E[ReLU]^2
```

For the off-diagonal covariance after ReLU, the gain approximation is used:

```
Cov_post[i,j] = Phi(alpha_i) * Phi(alpha_j) * Cov_pre[i,j]
```

This is exact only when pre-activation means are zero. It accumulates systematic bias over 8 layers.

**Budget:** ~404M FLOPs (0.6% of budget). Gets the 0.1x score multiplier.

### 2. Hybrid Estimator (`estimator_hybrid.py`)

Runs covariance propagation first, then uses 90% of the remaining FLOP budget for raw Monte Carlo forward passes.

Budget calculation:
- Covariance propagation: 8 layers x ~51M FLOPs/layer = ~408M FLOPs
- Remaining: ~67.6B FLOPs
- FLOPs per MC sample: `16*n + depth * (2*n^2 + 2*n)` where the 2x on matmul accounts for flopscope counting both multiply and add
- With n=256, depth=8: 1,056,768 FLOPs/sample
- K = floor(0.90 * 67.6B / 1,056,768) = ~57,570 samples

The MC forward pass:
```
X ~ N(0, I_{256})  shape (K, 256)
for each layer: h = ReLU(h @ W)
output[ell] = mean(h_ell, axis=0)
```

MC gives unbiased estimates, eliminating the gain approximation's systematic error.

### 3. Antithetic Control Variate (`estimator_control_variate.py`)

Replaces raw MC with antithetic pairs (x_k, -x_k). Since X ~ N(0, I), both +x and -x are valid samples from the same distribution.

**Variance reduction:** For two correlated estimators f(x) and f(-x):

```
Var[(f(x) + f(-x)) / 2] = Var[f] * (1 + rho) / 2
```

where rho = Cor[f(x), f(-x)].

Measured empirically on this architecture: rho = -0.16.

This gives a variance reduction factor of (1 + rho) = 0.84, meaning 16% lower MSE than raw MC at identical compute cost. Each antithetic pair uses the same budget as two independent samples but has variance equal to 0.84 times that of two independent samples.

Budget calculation:
- FLOPs per pair: `16*n + 2 * depth * (2*n^2 + n) + 2 * depth * n`
- With n=256, depth=8: 2,109,440 FLOPs/pair
- K_pairs = floor(0.90 * 67.6B / 2,109,440) = ~28,873 pairs (equivalent to 57,746 samples)

The output is the antithetic mean for each layer:
```
output[ell] = (mean(f_pos[ell]) + mean(f_neg[ell])) / 2
```

---

## Benchmark Results

Reference: 1,000,000 Monte Carlo samples (near-exact ground truth). Architecture: width=256, depth=8, seed=0.

| Estimator | FLOPs Used | C / B | MSE (final layer) | Score |
|---|---|---|---|---|
| Mean propagation (baseline) | 2.7M | 0.000 | 1.04e-03 | 1.04e-04 |
| estimator_covariance.py | 404M | 0.006 | 5.06e-05 | 5.06e-06 |
| estimator_hybrid.py | 61.1B | 0.899 | 5.39e-06 | 4.85e-06 |
| estimator_control_variate.py | 61.1B | 0.899 | 4.82e-06 | **4.34e-06** |

The antithetic control variate achieves a 24x improvement in score over the baseline mean propagation and an 11% improvement over the raw MC hybrid. The covariance-only estimator has systematic bias from the gain approximation that dominates its 10x score discount.

---

## Key Implementation Details

### flopscope Cost Table

Operations inside `predict()` are instrumented by flopscope:

| Operation | Cost |
|---|---|
| standard_normal(shape) | 16 FLOPs/element |
| matmul (M, N) @ (N, K) | 2 * M * N * K FLOPs (counts multiply + add separately) |
| pointwise ops (add, mul, max) | 1 FLOP/element |
| mean, sum | 1 FLOP/element |
| norm.pdf, norm.cdf | counted per call |

The 2x matmul cost is the most important: flopscope counts both the multiply and the accumulate, so `(K, 256) @ (256, 256)` costs 2 * K * 256 * 256 FLOPs, not K * 256 * 256.

### Forward Pass Convention

The weight matrix convention in this challenge uses row-vector batch form:

```python
h_new = ReLU(h_old @ W)   # h shape: (batch, width), W shape: (width, width)
```

Equivalently, for single samples: `h_new = ReLU(W.T @ h_old)`.

The weights are stored as `mlp.weights`, a list of 8 arrays each with shape `(width, width)` = `(256, 256)`.

### Why the Gain Approximation Has Bias

The exact formula for off-diagonal covariance after a ReLU requires a bivariate normal CDF:

```
Cov[ReLU(z_i), ReLU(z_j)] = Sigma_ij * Phi_2(0, 0; rho_ij)
```

The gain approximation uses `Phi(alpha_i) * Phi(alpha_j) * Sigma_ij` instead, which is exact only when all pre-activation means are zero. After the first layer, the means become non-zero and bias accumulates. After 8 layers, MSE is ~5e-5 vs ~5e-6 for MC at equivalent compute.

---

## Setup and Reproduction

Requires Python 3.10+ and `uv` (the package manager used by the starter kit).

```bash
git clone <this-repo>
cd whest-starterkit
uv sync
```

Run any estimator locally:

```bash
uv run python estimator_control_variate.py
uv run python estimator_hybrid.py
uv run python estimator_covariance.py
```

Each file has a `__main__` block that builds an MLP (width=256, depth=8, seed=0) and calls `compare_against_monte_carlo()` from `local_engine.py`, printing a convergence table.

Validate the estimator contract (shapes, types, budget compliance):

```bash
uv run whest validate --estimator estimator.py
```

Run against the public mini split (100 MLPs, requires network access):

```bash
uv run whest run --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026 \
    --split mini \
    --runner local
```

---

## Submission

The AIcrowd grader runs `estimator.py` as the entry point. Copy the best estimator to that file before submitting:

```bash
copy whest-starterkit\estimator_control_variate.py whest-starterkit\estimator.py
```

Then package and submit:

```bash
cd whest-starterkit
uv run whest login        # one-time: enter your AIcrowd API key
uv run whest submit --estimator estimator.py --watch
```

Alternatively, build the tarball and upload manually via the AIcrowd portal:

```bash
uv run whest package --estimator estimator.py --output submission.tar.gz
```

The grader runs in a sandboxed container with no network access, no GPU, 16 vCPUs, 64 GB RAM, and a 60-second wall clock. The `setup()` method runs once before any predictions; `predict()` is called once per MLP in the evaluation suite.

---

## Potential Improvements

The current implementation uses pure antithetic MC. Several directions could improve the score further:

**Exact bivariate covariance formula.** Replacing the gain approximation with the true formula `Cov_post[i,j] = Sigma_ij * Phi_2(0,0; rho_ij)` would eliminate the systematic bias in `estimator_covariance.py`. If the bias is then lower than the MC variance, an optimal linear combination of analytical and MC estimates would win.

**Optimal linear blending.** Given unbiased antithetic estimate `mu_anti` and biased analytical estimate `mu_cov`, the MSE-minimizing combination is:

```
mu_final = alpha * mu_cov + (1 - alpha) * mu_anti
alpha_opt = Var[anti] / (Bias[cov]^2 + Var[anti])
```

With current numbers (Bias^2 = 5e-5, Var = 5e-6), alpha_opt = 0.09. This gives approximately 8% additional MSE reduction over pure antithetic MC.

**Quasi-random sampling.** Replacing standard normal samples with scrambled Sobol sequences (if supported by flopscope) could further reduce MC variance beyond antithetic pairs.

**Layer-adaptive budget.** Deep layers contribute more variance per FLOP since the activation distribution is further from zero-mean Gaussian. Allocating more samples to later layers (by computing intermediates at lower resolution) could improve the score weighting.
