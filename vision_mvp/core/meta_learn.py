"""First-order Reptile meta-learning (OQ2: scale inference).

OQ2 asks whether per-agent scale assignments can be *inferred* from (task,
role) descriptions instead of hand-assigned. Reptile (Nichol, Achiam, Schulman,
2018) is a first-order meta-learning algorithm that trains an initialization
θ such that a few SGD steps on any task from the meta-distribution produces a
task-specific model θ_task with low loss:

    for i in 1..K:
        sample task T
        θ_T = SGD(θ, T, k_steps)
        θ ← θ + ε (θ_T − θ)

That's it. No second-order gradients, no inner-loop differentiation — just
first-order SGD with a slow outer aggregation rate ε.

In this repo we apply Reptile to *scale inference*: given a feature vector
x = [task_embedding ⊕ role_embedding] we want a regressor y = f_θ(x) that
predicts the appropriate scale level. A meta-distribution of (task, role,
best_scale) triples is available from Phase 1-2 logs; Reptile learns an
initialization that adapts in 1-5 inner steps to any new role/task pair.

This module ships a pure-NumPy linear regressor (sufficient for the
tabular-features regime) with a Reptile outer loop. A torch-backed nonlinear
version is a Tier-3 upgrade deferred until we have richer features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# --------------------------------------------------- linear task: OLS step

def ols_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of ½‖Xθ − y‖² / n wrt θ."""
    n = X.shape[0]
    return (X.T @ (X @ theta - y)) / max(n, 1)


def inner_sgd(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    k_steps: int,
    lr: float,
) -> np.ndarray:
    """k_steps of plain gradient descent on OLS loss starting from θ."""
    theta = theta.copy()
    for _ in range(k_steps):
        theta = theta - lr * ols_gradient(theta, X, y)
    return theta


# --------------------------------------------------- Reptile outer loop

@dataclass
class ReptileConfig:
    inner_lr: float = 0.05
    outer_lr: float = 0.1
    inner_steps: int = 5


def reptile_train(
    tasks: list[tuple[np.ndarray, np.ndarray]],
    feature_dim: int,
    n_outer: int = 200,
    cfg: ReptileConfig | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Train a linear-regression initialisation via Reptile.

    `tasks` is a list of (X, y) pairs, each representing one meta-training task.
    Returns θ (shape `(feature_dim,)`) — the learned initialisation.
    """
    if cfg is None:
        cfg = ReptileConfig()
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(feature_dim) * 0.01

    for _ in range(n_outer):
        X, y = tasks[rng.integers(0, len(tasks))]
        theta_task = inner_sgd(theta, X, y, cfg.inner_steps, cfg.inner_lr)
        # Outer Reptile update
        theta = theta + cfg.outer_lr * (theta_task - theta)

    return theta


def adapt(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
          cfg: ReptileConfig | None = None) -> np.ndarray:
    """Fine-tune θ on a new task (X, y) for cfg.inner_steps gradient steps."""
    if cfg is None:
        cfg = ReptileConfig()
    return inner_sgd(theta, X, y, cfg.inner_steps, cfg.inner_lr)


# --------------------------------------------------- scale-inference helpers

def build_scale_task(
    task_embed: np.ndarray,
    role_embed: np.ndarray,
    scale: int,
    n_samples: int = 32,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a few-shot supervised task around a (task, role, scale) triple.

    Inputs are task_embed⊕role_embed features plus small Gaussian jitter;
    targets are the scalar scale with i.i.d. label noise. Useful when
    logged Phase 1-2 data is sparse — enables meta-training on a broader
    meta-distribution.
    """
    rng = np.random.default_rng(seed)
    base = np.concatenate([task_embed, role_embed])
    X = base[None, :] + rng.standard_normal((n_samples, base.size)) * noise
    y = float(scale) + rng.standard_normal(n_samples) * noise
    return X, y


def evaluate(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Mean-squared error of θ on (X, y)."""
    pred = X @ theta
    return float(np.mean((pred - y) ** 2))
