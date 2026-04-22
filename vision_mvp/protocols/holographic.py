"""Holographic-boundary protocol — VISION_MILLIONS Idea 5.

Inspired by the holographic principle in physics: the information in a
region is encoded on its *boundary*, not distributed through its bulk.
Ryu-Takayanagi formula: S(A) = Area(γ_A) / 4G_N.

For multi-agent teams: designate ~N^{2/3} "boundary" agents that maintain
a compressed summary of the full system state. The other (N − N^{2/3})
"interior" agents never write to the shared register; they only read the
boundary summary on demand. Interior agents are cheap — they carry a
small local state, plus the ability to fetch from the boundary.

Compare to Phase 3 hierarchical (workspace = log N): holographic makes a
different trade-off. Boundary storage is O(N^{2/3}) rather than O(log N),
but per-agent cost is still O(log N) and the boundary can handle richer
queries (agent-specific sub-slices of the summary, not just a scalar goal).

This MVP uses the simplest realization:
  - Boundary agents participate in the streaming PCA and the shared
    register (like Phase 3 workspace members).
  - Interior agents ONLY update their own estimate from observations and
    read the boundary summary.
  - Boundary size = ceil(N^{2/3}).

Resulting bandwidth:
  - Writes per round: up to workspace size (⌈log N⌉), but only boundary
    agents are eligible. Because boundary is much larger than workspace,
    the workspace always fills.
  - Total system state: O(N^{2/3}) (on the boundary) vs O(N) (Phase 3).

This gets interesting at N ≥ 10 000 where N^{2/3} ~ 500 is much less than N
but much more than log N ~ 14. The extra boundary capacity means the
reconstruction can be richer.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from ..core.bus import Bus
from ..core.learned_manifold import StreamingPCA
from ..core.vectorized_predictor import PredictorBank
from ..core.workspace import Workspace
from ..tasks.drifting_consensus import DriftingConsensus


@dataclass
class HoloResult:
    estimates_over_time: np.ndarray
    bus_summary: dict[str, float]
    task_metrics: dict[str, float]
    subspace_alignment: list[float]
    writes_per_round: list[int]
    boundary_size: int
    workspace_size: int


def run_holographic(task: DriftingConsensus,
                    surprise_tau: float = 0.5,
                    decay: float = 0.85,
                    pca_lr: float = 0.1,
                    pred_lr: float = 0.01,
                    seed: int = 0) -> HoloResult:
    N = task.n_agents
    d = task.dim
    m = task.intrinsic_rank
    T = task.n_steps

    bus = Bus()
    pca = StreamingPCA.build(d, m, lr=pca_lr, seed=seed)
    bank = PredictorBank.build(N, d, hidden=16, lr=pred_lr, seed=seed)

    # Boundary selection — first N^{2/3} agents are boundary
    boundary_size = max(int(math.ceil(N ** (2.0 / 3.0))), 2 * m)
    boundary_size = min(boundary_size, N)
    boundary = set(range(boundary_size))

    workspace = Workspace(n_agents=boundary_size, epsilon=0.05)

    estimates = task.observations_at(0).copy()
    weights = np.ones(N) / (task.noise ** 2)

    reg_value = np.zeros(m)
    reg_weight = 0.0

    prev_estimates = estimates.copy()
    out = np.zeros((T, N, d))
    alignment_hist: list[float] = []
    writes_hist: list[int] = []

    for t in range(T):
        obs = task.observations_at(t)
        forget = 0.4
        new_estimates = (1 - forget) * estimates + forget * obs

        # PCA updates from boundary mean only (boundary is the sensor surface)
        pca.update(obs[:boundary_size].mean(axis=0))

        reg_value = reg_value * decay
        reg_weight = reg_weight * decay

        # Vectorized predictor forward (all agents)
        pred = bank.predict(prev_estimates)
        saliences = np.linalg.norm(new_estimates - pred, axis=1)

        # Workspace: top-k from BOUNDARY only
        bound_sal = saliences[:boundary_size]
        admitted_local = workspace.select(bound_sal, seed=seed + t)
        # Map back to global agent ids (same here since boundary is 0..B-1)
        admitted = admitted_local
        writes_this_round = 0
        for i in admitted:
            if bound_sal[i] > surprise_tau:
                y_i = pca.project(new_estimates[int(i)])
                reg_value = reg_value + weights[int(i)] * y_i
                reg_weight = reg_weight + weights[int(i)]
                bus.send(int(i), -1, m + 1, "boundary_write", t)
                writes_this_round += 1
        writes_hist.append(writes_this_round)

        # Every agent reads the boundary summary
        summary = reg_value / max(reg_weight, 1e-8)
        reconstructed = pca.reconstruct(summary)
        for i in range(N):
            bus.send(-1, i, m, "holo_read", t)
            bus.note_context(i, m)

        total_w = weights + reg_weight
        safe_w = np.where(total_w > 0, total_w, 1.0)
        merged = (weights[:, None] * new_estimates
                  + reg_weight * reconstructed[None, :]) / safe_w[:, None]
        new_estimates = merged

        bank.observe(prev_estimates, new_estimates)
        prev_estimates = new_estimates.copy()
        estimates = new_estimates
        out[t] = estimates

        alignment_hist.append(pca.subspace_alignment(task.basis))

    metrics = task.evaluate_tracking(out)
    return HoloResult(
        estimates_over_time=out,
        bus_summary=bus.summary(),
        task_metrics=metrics,
        subspace_alignment=alignment_hist,
        writes_per_round=writes_hist,
        boundary_size=boundary_size,
        workspace_size=workspace.capacity(),
    )
