"""Consensus + Innovations distributed Kalman filter.

Kar & Moura (2012). Each agent i runs a local Kalman update and then averages
its state estimate with neighbors via a consensus gossip step. Convergence to
the centralized filter is guaranteed under weak connectivity conditions:

  x̂_i(k+1) = A x̂_i(k) + K_i(k) · [y_i(k) − C x̂_i(k)]   (innovations)
  x̂_i(k+1) ← x̂_i(k+1) + β · Σ_{j ∈ N(i)} [x̂_j(k+1) − x̂_i(k+1)]   (consensus)

With β ∈ (0, 1/d_max), the consensus step preserves stability of the overall
filter. Average per-agent error matches the centralized Kalman optimum at
rate O(1/k).

Used in CASR as a principled upgrade over "everyone sends observations to an
orchestrator": each agent maintains its own estimate, shares only estimates
with neighbors, and provable convergence follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CIKalman:
    """Consensus + Innovations Kalman filter for a team of N agents.

    Linear Gaussian system: x_{k+1} = A x_k + w_k,   w_k ~ N(0, Q)
    agent-i measurement: y_i = C_i x + v_i,          v_i ~ N(0, R_i)
    """
    A: np.ndarray             # (d, d)
    Q: np.ndarray             # (d, d)
    C: list[np.ndarray]       # per-agent measurement matrices (m_i, d)
    R: list[np.ndarray]       # per-agent measurement noises (m_i, m_i)
    adjacency: np.ndarray     # (N, N) binary, symmetric; 1 = neighbor
    beta: float = 0.1
    _state: list[np.ndarray] = field(init=False)
    _cov: list[np.ndarray] = field(init=False)

    def __post_init__(self):
        N = len(self.C)
        if len(self.R) != N:
            raise ValueError("C and R must have the same length")
        d = self.A.shape[0]
        self._state = [np.zeros(d) for _ in range(N)]
        self._cov = [np.eye(d) for _ in range(N)]

    def step(self, measurements: list[np.ndarray]) -> list[np.ndarray]:
        """One innovations + one consensus step per agent. Returns new states."""
        N = len(self._state)
        d = self.A.shape[0]
        if len(measurements) != N:
            raise ValueError("need one measurement per agent")

        # --- predict ---
        pred = []
        pred_cov = []
        for i in range(N):
            x_hat_m = self.A @ self._state[i]
            P_m = self.A @ self._cov[i] @ self.A.T + self.Q
            pred.append(x_hat_m)
            pred_cov.append(P_m)

        # --- innovations update ---
        new_state = []
        new_cov = []
        for i in range(N):
            C_i = self.C[i]
            R_i = self.R[i]
            y_i = measurements[i]
            S = C_i @ pred_cov[i] @ C_i.T + R_i
            K = pred_cov[i] @ C_i.T @ np.linalg.inv(S)
            x_new = pred[i] + K @ (y_i - C_i @ pred[i])
            P_new = (np.eye(d) - K @ C_i) @ pred_cov[i]
            new_state.append(x_new)
            new_cov.append(P_new)

        # --- consensus step ---
        consensus_state = []
        for i in range(N):
            nbrs = np.where(self.adjacency[i] > 0)[0]
            delta = sum(new_state[j] - new_state[i] for j in nbrs) if nbrs.size else 0
            consensus_state.append(new_state[i] + self.beta * delta)

        self._state = consensus_state
        self._cov = new_cov
        return [x.copy() for x in self._state]

    def estimate(self, agent_id: int) -> np.ndarray:
        return self._state[agent_id].copy()
