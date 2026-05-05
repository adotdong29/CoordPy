"""AgentKeyIndex — learned keys per agent + clustered index for O(√N) routing.

Each agent owns a key vector `k_i ∈ R^d` that describes "what I do." Messages
carry a query vector `q_m = φ(message)`. Routing delivers each message to the
top-k agents by `⟨q_m, k_i⟩ / √d`.

Key learning (self-supervised):
  - Positive signal: when agent j's reply is re-used or @-mentioned → nudge
    j's key toward the triggering query.
  - Negative signal: when j is selected but its reply is ignored → nudge j's
    key slightly away.

Cluster index for scale:
  - k-means over keys, C = ⌈√N⌉ clusters.
  - Route: dot-product against C centroids, descend into top-2 clusters,
    score within, pick top-k recipients.
  - Re-cluster every RECLUSTER_EVERY messages.

This is a classical Mixture-of-Experts router (Switch / Mixtral / Routing
Transformer) applied to agent messaging rather than to feed-forward experts.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


@dataclass
class AgentKeyIndex:
    n_agents: int
    dim: int
    learning_rate: float = 0.1
    negative_rate: float = 0.01
    recluster_every: int = 200
    seed: int = 0

    _keys: np.ndarray = field(default=None)          # type: ignore  (N, d)
    _usage_count: np.ndarray = field(default=None)   # type: ignore  (N,)
    _centroids: np.ndarray = field(default=None)     # type: ignore  (C, d)
    _cluster_of: np.ndarray = field(default=None)    # type: ignore  (N,) int
    _cluster_members: dict = field(default_factory=dict)
    _msg_count_since_recluster: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        if self._keys is None:
            self._keys = l2_normalize(rng.standard_normal((self.n_agents, self.dim)))
        if self._usage_count is None:
            self._usage_count = np.zeros(self.n_agents, dtype=np.int64)
        self._recluster()

    # ---- cluster maintenance ----

    def _recluster(self, n_iters: int = 8) -> None:
        """Simple k-means from scratch (numpy only, no sklearn)."""
        n_clusters = max(2, min(self.n_agents, int(math.ceil(math.sqrt(self.n_agents)))))
        rng = np.random.default_rng(self.seed + self._msg_count_since_recluster)
        # Init: pick n_clusters random distinct agents' keys
        idx = rng.choice(self.n_agents, size=n_clusters, replace=False)
        centroids = self._keys[idx].copy()
        for _ in range(n_iters):
            # Assign
            sims = self._keys @ centroids.T                        # (N, C)
            assign = np.argmax(sims, axis=1)                        # (N,)
            # Update
            new_cent = np.zeros_like(centroids)
            for c in range(n_clusters):
                mask = assign == c
                if mask.any():
                    new_cent[c] = self._keys[mask].mean(axis=0)
                else:
                    new_cent[c] = self._keys[rng.integers(self.n_agents)]
            new_cent = l2_normalize(new_cent)
            if np.allclose(new_cent, centroids, atol=1e-4):
                break
            centroids = new_cent
        self._centroids = centroids
        self._cluster_of = np.argmax(self._keys @ centroids.T, axis=1)
        self._cluster_members = {c: np.where(self._cluster_of == c)[0].tolist()
                                 for c in range(n_clusters)}
        self._msg_count_since_recluster = 0

    # ---- routing ----

    def route(self, query: np.ndarray, top_k: int = 5,
              exclude: set[int] | None = None,
              load_penalty: float = 0.01,
              top_clusters: int = 2) -> list[int]:
        """Return top-k agent ids for the given query vector.

        - `exclude`: agent ids that should not be considered (e.g. sender).
        - `load_penalty`: subtract λ · log(1 + usage_i) from scores to
          discourage hot-hoarding (Switch-style auxiliary loss, inline).
        """
        q = l2_normalize(query)
        # 1) Compare against centroids
        cent_scores = q @ self._centroids.T                         # (C,)
        top_c = np.argsort(-cent_scores)[:top_clusters]
        # 2) Gather candidates
        cand = []
        for c in top_c:
            cand.extend(self._cluster_members.get(int(c), []))
        # Also include a small ε random sample for exploration
        rng = np.random.default_rng(int(np.sum(q * 1000)) % (2**31))
        eps = max(1, self.n_agents // 50)
        cand.extend(rng.integers(0, self.n_agents, size=eps).tolist())
        cand = list(set(cand))
        if exclude:
            cand = [i for i in cand if i not in exclude]
        if not cand:
            return []
        cand_arr = np.array(cand, dtype=np.int64)
        scores = self._keys[cand_arr] @ q - load_penalty * np.log1p(self._usage_count[cand_arr])
        order = np.argsort(-scores)[:top_k]
        chosen = cand_arr[order].tolist()
        # Track usage
        self._usage_count[chosen] += 1
        self._msg_count_since_recluster += 1
        if self._msg_count_since_recluster >= self.recluster_every:
            self._recluster()
        return [int(i) for i in chosen]

    # ---- key learning ----

    def update_positive(self, agent_id: int, query: np.ndarray) -> None:
        """Signal that agent_id's response was REUSED by downstream agents.
        Move its key toward the triggering query."""
        q = l2_normalize(query)
        self._keys[agent_id] = l2_normalize(
            self._keys[agent_id] + self.learning_rate * (q - self._keys[agent_id])
        )

    def update_negative(self, agent_id: int, query: np.ndarray) -> None:
        """Agent was routed to but its response was ignored. Nudge away."""
        q = l2_normalize(query)
        self._keys[agent_id] = l2_normalize(
            self._keys[agent_id] - self.negative_rate * (q - self._keys[agent_id])
        )

    def set_key(self, agent_id: int, key: np.ndarray) -> None:
        self._keys[agent_id] = l2_normalize(key)

    def get_key(self, agent_id: int) -> np.ndarray:
        return self._keys[agent_id].copy()

    @property
    def n_clusters(self) -> int:
        return self._centroids.shape[0]
