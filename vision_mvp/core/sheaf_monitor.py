"""Sheaf cohomology monitor — localize team disagreement.

A cellular sheaf on the agent graph assigns a stalk (vector space) to each
agent and a stalk to each edge, with restriction maps saying how adjacent
agents must agree on their shared interface. The coboundary operator δ
computes the "disagreement" on each edge: δ(x)_e = F_{v◁e}(x_v) − F_{u◁e}(x_u).

- **H⁰ = ker(δ)**: the set of globally-consistent belief assignments
  (what the team can agree on).
- **H¹ = coker(δ)**: obstructions — inconsistencies that can't be
  resolved by any choice of local beliefs.

In practice we don't need full cohomology; we need a per-edge discord
score: ‖δ(x)_e‖². High score = "agents u, v disagree on edge e." Drives
auto-spawning of reconciliation tasks only where needed.

Reference: Robinson (2017); Hansen-Ghrist (2019) "Toward a spectral
theory of cellular sheaves"; Bodnar-Di Giovanni et al. (NeurIPS 2022)
"Neural Sheaf Diffusion."
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SheafMonitor:
    """Computes per-edge disagreement on an agent graph.

    Each agent carries a belief vector in ℝ^d. Each edge carries the
    restriction: "only the first d_e coordinates need to match" where
    d_e ≤ d. Concretely we use identity restrictions on the interface
    dimensions.
    """
    stalk_dim: int                                  # d, per agent
    interface_dim: int                              # d_e, per edge
    # Edges as list of (u, v, interface_indices). `interface_indices` is
    # a list of length interface_dim picking which stalk coordinates form
    # the shared interface for that edge.
    edges: list[tuple[int, int, list[int]]] = field(default_factory=list)

    def add_edge(self, u: int, v: int,
                 interface_indices: list[int] | None = None) -> None:
        if interface_indices is None:
            interface_indices = list(range(self.interface_dim))
        if len(interface_indices) != self.interface_dim:
            raise ValueError("interface_indices length != interface_dim")
        self.edges.append((u, v, interface_indices))

    def edge_discord(self, beliefs: dict[int, np.ndarray]) -> list[dict]:
        """For each edge, compute ‖(restriction of u) - (restriction of v)‖²."""
        out = []
        for (u, v, idx) in self.edges:
            bu, bv = beliefs.get(u), beliefs.get(v)
            if bu is None or bv is None:
                out.append({"u": u, "v": v, "discord": None, "reason": "missing"})
                continue
            ru = bu[idx]
            rv = bv[idx]
            diff = ru - rv
            score = float(np.dot(diff, diff))
            out.append({"u": u, "v": v, "discord": score, "indices": idx,
                        "residual": diff.tolist()})
        return out

    def global_residual(self, beliefs: dict[int, np.ndarray]) -> float:
        """Sum of per-edge discord — a single scalar health metric."""
        edges = self.edge_discord(beliefs)
        return float(sum(e["discord"] for e in edges if e["discord"] is not None))

    def top_disagreements(self, beliefs: dict[int, np.ndarray],
                          k: int = 5) -> list[dict]:
        """Return k edges with highest discord — "where the team disagrees"."""
        edges = [e for e in self.edge_discord(beliefs) if e["discord"] is not None]
        edges.sort(key=lambda e: -e["discord"])
        return edges[:k]

    # ---- H^0 / H^1 via sheaf Laplacian (exact) ----

    def coboundary_matrix(self, agent_ids: list[int]) -> np.ndarray:
        """Build the sparse coboundary B such that (B·x)_e = F_{v◁e}(x_v) − F_{u◁e}(x_u).

        Rows: edges × interface_dim. Cols: agents × stalk_dim.
        """
        n = len(agent_ids)
        id_to_row = {aid: i for i, aid in enumerate(agent_ids)}
        E = len(self.edges)
        d_v = self.stalk_dim
        d_e = self.interface_dim
        B = np.zeros((E * d_e, n * d_v))
        for ei, (u, v, idx) in enumerate(self.edges):
            if u not in id_to_row or v not in id_to_row:
                continue
            ur = id_to_row[u]
            vr = id_to_row[v]
            for k, coord in enumerate(idx):
                B[ei * d_e + k, ur * d_v + coord] = -1.0
                B[ei * d_e + k, vr * d_v + coord] = +1.0
        return B

    def cohomology_dims(self, agent_ids: list[int]) -> dict:
        """Return dim H⁰ (agreed beliefs) and dim H¹ (irreducible obstructions)."""
        B = self.coboundary_matrix(agent_ids)
        if B.size == 0:
            return {"dim_H0": 0, "dim_H1": 0, "rank": 0}
        r = int(np.linalg.matrix_rank(B, tol=1e-8))
        return {
            "dim_H0": B.shape[1] - r,
            "dim_H1": B.shape[0] - r,
            "rank": r,
            "shape": B.shape,
        }
