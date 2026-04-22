"""Modern Hopfield networks — attention-compatible associative memory.

Ramsauer et al. (NeurIPS 2020) showed that softmax attention is equivalent to
a single update of a modern continuous Hopfield network. Given stored patterns
X ∈ ℝ^{M×d} and a query ξ ∈ ℝ^d, the retrieval step is

    ξ_new = Xᵀ · softmax(β X ξ)

This has exponential storage capacity (unlike classical Hopfield's linear one).
Slots naturally into `core/stigmergy.py` as an associative-retrieval primitive:
agents write (key, value) into X; queries retrieve the convex combination
most-aligned with the query key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModernHopfield:
    """In-memory associative store with softmax-attention retrieval."""

    d: int
    beta: float = 1.0
    _patterns: list[np.ndarray] = field(default_factory=list)

    def store(self, pattern: np.ndarray) -> int:
        """Add a pattern; return its slot index."""
        p = np.asarray(pattern, dtype=float).ravel()
        if p.size != self.d:
            raise ValueError(f"pattern must be length {self.d}")
        self._patterns.append(p.copy())
        return len(self._patterns) - 1

    def capacity(self) -> int:
        return len(self._patterns)

    def matrix(self) -> np.ndarray:
        if not self._patterns:
            return np.zeros((0, self.d))
        return np.stack(self._patterns, axis=0)

    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """One modern-Hopfield update: X^T softmax(β X q)."""
        q = np.asarray(query, dtype=float).ravel()
        if q.size != self.d:
            raise ValueError(f"query must be length {self.d}")
        if not self._patterns:
            return q.copy()
        X = self.matrix()
        scores = X @ q
        m = float(scores.max())
        w = np.exp(self.beta * (scores - m))
        w = w / w.sum()
        return X.T @ w

    def attention_weights(self, query: np.ndarray) -> np.ndarray:
        """Same softmax as `retrieve` but returns the weight vector (M,)."""
        q = np.asarray(query, dtype=float).ravel()
        if not self._patterns:
            return np.zeros(0)
        X = self.matrix()
        scores = X @ q
        m = float(scores.max())
        w = np.exp(self.beta * (scores - m))
        return w / w.sum()

    def iterate(self, query: np.ndarray, n_steps: int = 3) -> np.ndarray:
        """Repeated retrieval — provably converges to a fixed point for
        moderate β; reaches one of the stored patterns with high probability."""
        x = np.asarray(query, dtype=float).ravel().copy()
        for _ in range(n_steps):
            x = self.retrieve(x)
        return x
