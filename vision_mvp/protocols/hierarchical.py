"""Hierarchical protocol — Phase 3.

Combines all Phase-3 pieces:
  - Learned manifold basis (streaming PCA, from Phase 2)
  - Basis-invariant neural-net predictor per agent
  - Global Workspace: only top-k most-surprised agents write each round
  - Role hierarchy: orchestrator reads manifold, broadcasts summary goal;
    workers use goal to modulate their forgetting rate

Per-round protocol:
  1. Each agent gets a new observation and updates its state (with forget)
  2. Streaming PCA updates (mean-of-observations signal)
  3. Each agent's predictor computes surprise on its (prev → now) transition
  4. Workspace selects top-k most-surprised agents
  5. Selected agents write projection y_i to shared register; others skip
  6. Orchestrator reads register, broadcasts summary goal (m floats) to all
  7. All agents reconstruct from summary, Bayesian-merge into their estimate
  8. Predictors update (SGD step) on the actual (prev → now) transition

Expected improvements over Phase 2:
  - Writes per round drop from N to k = O(log N) — orders-of-magnitude less
    bus traffic at scale.
  - Predictor stays meaningful as PCA rotates, so surprise filter actually
    fires (which is what lets workspace be sparse).
  - Orchestrator goal broadcast adds only m floats per agent per round.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..core.bus import Bus
from ..core.learned_manifold import StreamingPCA
from ..core.neural_predictor import NeuralPredictor
from ..core.vectorized_predictor import PredictorBank
from ..core.workspace import Workspace
from ..core.hierarchy import split_roles
from ..tasks.drifting_consensus import DriftingConsensus


@dataclass
class HierResult:
    estimates_over_time: np.ndarray
    bus_summary: dict[str, float]
    task_metrics: dict[str, float]
    subspace_alignment: list[float]
    writes_per_round: list[int]
    workspace_size: int


def run_hierarchical(task: DriftingConsensus,
                     surprise_tau: float = 0.5,
                     decay: float = 0.75,
                     pca_lr: float = 0.15,
                     pred_lr: float = 0.01,
                     pred_hidden: int = 16,
                     seed: int = 0,
                     workspace_epsilon: float = 0.05) -> HierResult:
    N = task.n_agents
    d = task.dim
    m = task.intrinsic_rank
    T = task.n_steps

    bus = Bus()
    pca = StreamingPCA.build(d, m, lr=pca_lr, seed=seed)
    bank = PredictorBank.build(N, d, hidden=pred_hidden, lr=pred_lr, seed=seed)
    workspace = Workspace(n_agents=N, epsilon=workspace_epsilon)

    n_orch, n_work = split_roles(N)

    # Per-agent estimator state
    estimates = task.observations_at(0).copy()
    weights = np.ones(N) / (task.noise ** 2)

    # Shared register
    reg_value = np.zeros(m)
    reg_weight = 0.0

    # Previous-step agent state (for predictor input)
    prev_estimates = estimates.copy()

    out = np.zeros((T, N, d))
    alignment_hist: list[float] = []
    writes_hist: list[int] = []

    for t in range(T):
        obs = task.observations_at(t)

        # Step 1: incorporate own observation with forgetting
        forget = 0.4
        new_estimates = (1 - forget) * estimates + forget * obs

        # Step 2: update streaming PCA with clean (mean) signal
        pca.update(obs.mean(axis=0))

        # Step 3: decay shared register
        reg_value = reg_value * decay
        reg_weight = reg_weight * decay

        # Step 4: compute surprise per agent (prediction error in d-space)
        pred = bank.predict(prev_estimates)
        saliences = np.linalg.norm(new_estimates - pred, axis=1)

        # Step 5: workspace selects top-k most-surprised
        admitted = workspace.select(saliences, seed=seed + t)
        # Additionally require salience > τ to write (both conditions)
        writes_this_round = 0
        for i in admitted:
            if saliences[i] > surprise_tau:
                y_i = pca.project(new_estimates[i])
                reg_value = reg_value + weights[i] * y_i
                reg_weight = reg_weight + weights[i]
                bus.send(int(i), -1, m + 1, "workspace_write", t)
                writes_this_round += 1
        writes_hist.append(writes_this_round)

        # Step 6: orchestrator reads the register, broadcasts goal to all
        summary = reg_value / max(reg_weight, 1e-8)
        # Cost: one orchestrator read (m) + N broadcast reads (m each)
        bus.send(-1, -1, m, "orch_read", t)
        for i in range(N):
            bus.send(-1, i, m, "goal_broadcast", t)
            bus.note_context(i, m)

        # Step 7: each agent reconstructs and Bayesian-merges
        reconstructed = pca.reconstruct(summary)
        eff_weight = max(reg_weight, 0.0)
        total_w = weights + eff_weight
        safe_w = np.where(total_w > 0, total_w, 1.0)
        merged = (weights[:, None] * new_estimates
                  + eff_weight * reconstructed[None, :]) / safe_w[:, None]
        new_estimates = merged

        # Step 8: predictors update on the (prev → new) transition (vectorized)
        bank.observe(prev_estimates, new_estimates)

        prev_estimates = new_estimates.copy()
        estimates = new_estimates
        out[t] = estimates

        alignment_hist.append(pca.subspace_alignment(task.basis))

    metrics = task.evaluate_tracking(out)
    return HierResult(
        estimates_over_time=out,
        bus_summary=bus.summary(),
        task_metrics=metrics,
        subspace_alignment=alignment_hist,
        writes_per_round=writes_hist,
        workspace_size=workspace.capacity(),
    )
