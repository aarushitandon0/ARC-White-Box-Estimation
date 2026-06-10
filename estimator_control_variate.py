"""Control variate estimator: covariance propagation + antithetic MC.

Strategy:
  1. Run full covariance propagation to get mu_cov[ell] for all 8 layers.
  2. Use remaining budget for antithetic MC pairs (x_k, -x_k).
     Since X ~ N(0,I), both x_k and -x_k are valid samples.

Why antithetic works:
  For each final-layer neuron: Cor[f(x), f(-x)] = rho ~ -0.16 (measured).
  Antithetic pair estimator: mu_k = (f(x_k) + f(-x_k)) / 2
  Var[mu_k] = Var[f] * (1 + rho) / 2 = 0.84/2 * Var[f]
  Average over K_pairs: Var[mean] = 0.84 * Var[f] / (2 * K_pairs)

vs raw MC with same compute (2 * K_pairs samples):
  Var[raw_mean] = Var[f] / (2 * K_pairs)
  Reduction: (1 + rho) = 0.84  -> 16% variance reduction.

Budget: K_pairs = 0.90 * (budget - cov_prop) / flops_per_pair
  where flops_per_pair = sampling + 2 * forward_passes.

The covariance propagation result is used as a baseline to blend:
  mu_final = alpha * mu_cov + (1-alpha) * mu_anti
where alpha is chosen to minimize MSE given estimated bias^2 vs MC variance.
For simplicity alpha = 0 (pure anti) since MC is unbiased and typically wins
at this K with the 0.1x discount helping cov only at very small K.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e30
_COV_PROP_FLOPS_PER_LAYER = 51_000_000  # conservative estimate, actual ~50.5M


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        rng = fnp.random.default_rng(mlp.seed)
        width = mlp.width
        depth = mlp.depth

        # --- Step 1: covariance propagation ---
        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        log_scale = 0.0
        cov_rows = []

        for w in mlp.weights:
            cov_diag = fnp.diag(cov)
            max_var = float(fnp.max(cov_diag))
            if max_var > _COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))

            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi_a = flops.stats.norm.pdf(alpha)
            Phi_a = flops.stats.norm.cdf(alpha)

            mu = mu_pre * Phi_a + sigma_pre * phi_a
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            gain = fnp.array(
                fnp.where(
                    fnp.asarray(sigma_pre, dtype=fnp.float64) > 1e-12,
                    fnp.asarray(Phi_a, dtype=fnp.float64),
                    0.0,
                ).astype(fnp.float32)
            )
            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            fnp.fill_diagonal(cov, var_post)

            scale_factor = float(fnp.exp(log_scale))
            cov_rows.append(mu * scale_factor)

        # --- Step 2: antithetic MC budget calculation ---
        # Each pair needs: sampling (16*n) + 2 forward passes (2 * depth * 2*n^2 + n)
        cov_prop_cost = depth * _COV_PROP_FLOPS_PER_LAYER
        remaining = budget - cov_prop_cost

        flops_per_pair = (
            16 * width                           # sampling x_k (–x_k is free)
            + 2 * depth * (2 * width * width + width)  # two forward passes (2x matmul)
            + 2 * depth * width                  # two means (one per sign)
        )
        K_pairs = max(1, int(0.90 * remaining / flops_per_pair))

        # --- Step 3: antithetic MC forward passes ---
        X = fnp.array(rng.standard_normal((K_pairs, width)).astype(fnp.float32))

        # Positive pass: f(x)
        h_pos = X
        mean_rows_pos = []
        for w in mlp.weights:
            h_pos = fnp.maximum(fnp.matmul(h_pos, w), 0.0)
            mean_rows_pos.append(fnp.mean(h_pos, axis=0))

        # Negative pass: f(-x)  — same input distribution by symmetry of N(0,I)
        h_neg = -X
        mean_rows_neg = []
        for w in mlp.weights:
            h_neg = fnp.maximum(fnp.matmul(h_neg, w), 0.0)
            mean_rows_neg.append(fnp.mean(h_neg, axis=0))

        # --- Step 4: antithetic estimate per layer ---
        result_rows = []
        for ell in range(depth):
            anti_mean = (mean_rows_pos[ell] + mean_rows_neg[ell]) * 0.5
            result_rows.append(anti_mean)

        return fnp.stack(result_rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=8, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(6.8e10))
