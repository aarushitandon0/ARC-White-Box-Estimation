"""Covariance propagation estimator — full (width x width) covariance tracking.

Propagates (mu, Sigma) through each linear + ReLU layer using:
  - Linear: mu_pre = W^T mu,  cov_pre = W^T cov W  (via einsum to preserve symmetry)
  - ReLU:   mu_post[i] = mu_pre[i]*Phi(alpha[i]) + sigma_pre[i]*phi(alpha[i])
            cov_post[i,j] ~ gain[i]*gain[j]*cov_pre[i,j]  (gain[i] = Phi(alpha[i]))
            diag(cov_post)[i] = exact marginal variance

Budget usage: ~400M FLOPs (< 1% of the 6.8e10 budget), so the 0.1x score
multiplier cap applies. This estimator is purely analytical — no MC sampling.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e30


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        width = mlp.width

        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        log_scale = 0.0

        rows = []
        for w in mlp.weights:
            # Overflow guard
            cov_diag = fnp.diag(cov)
            max_var = float(fnp.max(cov_diag))
            if max_var > _COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))

            # Linear propagation
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_a = flops.stats.norm.pdf(alpha)
            Phi_a = flops.stats.norm.cdf(alpha)

            # Post-ReLU mean (exact per neuron)
            mu = mu_pre * Phi_a + sigma_pre * phi_a

            # Post-ReLU diagonal variance (exact per neuron)
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            # Off-diagonal covariance: gain[i]*gain[j]*cov_pre[i,j]
            sigma_f64 = fnp.asarray(sigma_pre, dtype=fnp.float64)
            Phi_f64 = fnp.asarray(Phi_a, dtype=fnp.float64)
            gain_f64 = fnp.where(sigma_f64 > 1e-12, Phi_f64, 0.0)
            gain = fnp.array(gain_f64.astype(fnp.float32))

            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            fnp.fill_diagonal(cov, var_post)

            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor)

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=8, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(6.8e10))
