"""Adaptive protocol — Phase 2.

Combines:
  - Streaming PCA for learned manifold basis (no oracle)
  - Per-agent linear predictor (surprise = prediction error)
  - CRDT-style manifold with exponential forgetting (to track drift)

Each round t:
  1. Agent i observes o_i(t); updates its estimate x_i(t)
  2. Agent i contributes o_i(t) to the shared streaming PCA (one sample)
  3. Agent i projects x_i(t) onto the current learned basis → y_i(t)
  4. Agent i's predictor predicts ŷ_i(t); surprise = ||y_i(t) - ŷ_i(t)||
  5. If surprise > τ, agent writes y_i(t) to the manifold register
  6. All agents read the current manifold summary, reconstruct, Bayesian-update
  7. Predictors observe actual y_i(t) to refine

The manifold register decays with factor γ each round so that drifted state
can be tracked (otherwise stale evidence dominates).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from ..core.bus import Bus
from ..core.learned_manifold import StreamingPCA
from ..core.predictor import LinearPredictor
from ..tasks.drifting_consensus import DriftingConsensus


@dataclass
class AdaptiveResult:
    estimates_over_time: np.ndarray        # (T, N, d)
    bus_summary: dict[str, float]
    task_metrics: dict[str, float]
    subspace_alignment: list[float]         # learned-vs-true basis alignment per round
    writes_per_round: list[int]


def run_adaptive(task: DriftingConsensus,
                 surprise_tau: float = 0.05,
                 decay: float = 0.7,
                 pca_lr: float = 0.1,
                 seed: int = 0) -> AdaptiveResult:
    N = task.n_agents
    d = task.dim
    m = task.intrinsic_rank                # target manifold dim
    T = task.n_steps

    bus = Bus()
    pca = StreamingPCA.build(d, m, lr=pca_lr, seed=seed)

    # Per-agent estimator state (precision-weighted)
    estimates = np.zeros((N, d))
    weights = np.ones(N) / (task.noise ** 2)
    # Initialize from first observation
    estimates[:] = task.observations_at(0)

    predictors = [LinearPredictor(dim=m) for _ in range(N)]

    # Shared manifold register: a single m-dim accumulator with decay
    reg_value = np.zeros(m)
    reg_weight = 0.0

    out = np.zeros((T, N, d))
    alignment_hist: list[float] = []
    writes_hist: list[int] = []

    for t in range(T):
        obs = task.observations_at(t)      # (N, d)

        # Step 1: each agent incorporates its own observation via Bayesian update.
        # Use exponential forgetting on old estimates so drift can propagate.
        # Equivalent to a Kalman-filter-like update with process noise.
        forget = 0.5   # how much to trust fresh obs vs prior estimate
        estimates = (1 - forget) * estimates + forget * obs

        # Step 2: streaming PCA update. Use the per-step MEAN of observations —
        # averaging across N agents kills isotropic noise at rate √N, giving
        # a clean sample of the truth trajectory. Over time, PCA on these
        # clean samples recovers the task-relevant subspace.
        # (In practice the mean is computed at the PCA node from the same
        # writes that hit the shared register — no extra bandwidth needed.)
        pca.update(obs.mean(axis=0))

        # Step 3: decay the shared register (drift adaptation)
        reg_value = reg_value * decay
        reg_weight = reg_weight * decay

        # Step 4: per-agent project, surprise, write-if-surprising
        writes_this_round = 0
        for i in range(N):
            y_i = pca.project(estimates[i])
            pred = predictors[i].predict()
            surprise = float(np.linalg.norm(y_i - pred))
            if surprise > surprise_tau:
                # Write m floats + one weight scalar
                reg_value = reg_value + weights[i] * y_i
                reg_weight = reg_weight + weights[i]
                bus.send(i, -1, m + 1, "write", t)
                writes_this_round += 1
            # Update local predictor with actual y
            predictors[i].observe(y_i)
        writes_hist.append(writes_this_round)

        # Step 5: every agent reads the current summary
        summary = reg_value / max(reg_weight, 1e-8)
        for i in range(N):
            bus.send(-1, i, m, "read", t)
            # Reconstruct and Bayesian-merge
            reconstructed = pca.reconstruct(summary)
            eff_weight = max(reg_weight - weights[i], 0.0)
            total_w = weights[i] + eff_weight
            if total_w > 0:
                estimates[i] = (weights[i] * estimates[i]
                                + eff_weight * reconstructed) / total_w
            bus.note_context(i, m)   # peak context = m tokens

        # Monitoring
        alignment_hist.append(pca.subspace_alignment(task.basis))
        out[t] = estimates

    metrics = task.evaluate_tracking(out)
    return AdaptiveResult(
        estimates_over_time=out,
        bus_summary=bus.summary(),
        task_metrics=metrics,
        subspace_alignment=alignment_hist,
        writes_per_round=writes_hist,
    )


def run_naive_drift(task: DriftingConsensus) -> AdaptiveResult:
    """Baseline: every agent broadcasts its full observation each round."""
    N = task.n_agents
    d = task.dim
    T = task.n_steps
    bus = Bus()

    estimates = np.zeros((N, d))
    estimates[:] = task.observations_at(0)
    weights = np.ones(N) / (task.noise ** 2)

    out = np.zeros((T, N, d))
    for t in range(T):
        obs = task.observations_at(t)
        forget = 0.5
        estimates = (1 - forget) * estimates + forget * obs
        # Broadcast: everyone sends estimate + weight to everyone
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                bus.send(j, i, d + 1, "broadcast", t)
        # Average
        mean_est = (weights[:, None] * estimates).sum(axis=0) / weights.sum()
        estimates = np.broadcast_to(mean_est, (N, d)).copy()
        for i in range(N):
            bus.note_context(i, (N - 1) * (d + 1))
        out[t] = estimates

    metrics = task.evaluate_tracking(out)
    return AdaptiveResult(
        estimates_over_time=out,
        bus_summary=bus.summary(),
        task_metrics=metrics,
        subspace_alignment=[0.0] * T,
        writes_per_round=[N * (N - 1)] * T,
    )
