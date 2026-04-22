"""Vectorized neural predictors — one forward/backward pass over all agents.

Instead of a Python loop over N predictors, we stack their weights into
bank tensors of shape (N, d, hidden) etc., and use einsum to run the whole
team's forward/backward in one call. This is what makes N=10^5 feasible —
at that scale the per-agent Python loop takes minutes, the vectorized
version takes seconds.

Each agent has the same architecture as in neural_predictor.py:
    x --[W1,b1]--> h --ReLU--> [W2,b2]--> residual
    x̂ = x + residual

All agents' weights are bundled. Forward/backward is across the leading
N axis. Gradient clipping + weight clipping carry over.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PredictorBank:
    n_agents: int
    dim: int
    hidden: int = 16
    lr: float = 0.01
    _W1: np.ndarray = None   # type: ignore  (N, d, h)
    _b1: np.ndarray = None   # type: ignore  (N, h)
    _W2: np.ndarray = None   # type: ignore  (N, h, d)
    _b2: np.ndarray = None   # type: ignore  (N, d)

    @classmethod
    def build(cls, n_agents: int, dim: int, hidden: int = 16,
              lr: float = 0.01, seed: int = 0) -> "PredictorBank":
        rng = np.random.default_rng(seed)
        scale = 0.01
        obj = cls(
            n_agents=n_agents, dim=dim, hidden=hidden, lr=lr,
            _W1=scale * rng.standard_normal((n_agents, dim, hidden)),
            _b1=np.zeros((n_agents, hidden)),
            _W2=scale * rng.standard_normal((n_agents, hidden, dim)),
            _b2=np.zeros((n_agents, dim)),
        )
        return obj

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, d). Returns (N, d) predicted next state."""
        # h = X @ W1  →  einsum("nd,ndh->nh", X, W1)
        h = np.einsum("nd,ndh->nh", X, self._W1) + self._b1
        h_act = np.maximum(h, 0)
        # residual = h_act @ W2 → einsum("nh,nhd->nd", h_act, W2)
        residual = np.einsum("nh,nhd->nd", h_act, self._W2) + self._b2
        return X + residual

    def observe(self, X_prev: np.ndarray, X_now: np.ndarray) -> np.ndarray:
        """Update all predictors. Returns (N,) surprise per agent."""
        # Forward
        h = np.einsum("nd,ndh->nh", X_prev, self._W1) + self._b1
        h_act = np.maximum(h, 0)
        residual = np.einsum("nh,nhd->nd", h_act, self._W2) + self._b2
        pred = X_prev + residual
        err = X_now - pred                                   # (N, d)
        surprises = np.linalg.norm(err, axis=1)

        # Backprop (loss = 0.5‖err‖²)
        d_residual = -err                                    # (N, d)
        # gW2[n] = outer(h_act[n], d_residual[n])
        gW2 = np.einsum("nh,nd->nhd", h_act, d_residual)
        gb2 = d_residual
        d_h_act = np.einsum("nd,nhd->nh", d_residual, self._W2)
        d_h = d_h_act * (h > 0)
        gW1 = np.einsum("nd,nh->ndh", X_prev, d_h)
        gb1 = d_h

        # Per-agent gradient clipping
        def clip(G: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
            # Norm over all dims except the leading N axis
            axes = tuple(range(1, G.ndim))
            n = np.sqrt((G * G).sum(axis=axes, keepdims=True)) + 1e-8
            scale = np.minimum(1.0, max_norm / n)
            return G * scale

        gW1 = clip(gW1); gb1 = clip(gb1); gW2 = clip(gW2); gb2 = clip(gb2)

        self._W1 -= self.lr * gW1
        self._b1 -= self.lr * gb1
        self._W2 -= self.lr * gW2
        self._b2 -= self.lr * gb2

        # Weight clipping
        np.clip(self._W1, -1.0, 1.0, out=self._W1)
        np.clip(self._W2, -1.0, 1.0, out=self._W2)

        return surprises
