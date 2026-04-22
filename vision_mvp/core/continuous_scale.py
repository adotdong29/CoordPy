"""Continuous scale — VISION_MILLIONS Idea 10.

Instead of a discrete 5-level scale hierarchy (Token → Statement → Function
→ Module → System), each agent has a continuous scale parameter s ∈ [0, log N]
that adapts to task demand. Scale projections are smoothly parametrized:

    P_s: ℝ^d → ℝ^{m(s)},  m(s) = max(1, ⌈d · 2^{−s}⌉).

Lower s → finer detail (larger m), higher s → coarser summary (smaller m).
The projection operator at scale s is orthogonal projection onto the top-m(s)
principal directions of a learned basis B (streaming PCA of the agent stream).

Adaptation: each agent monitors its own surprise signal. Large surprise
→ drop to finer scale (more detail needed). Small surprise → raise to
coarser scale (can compress more).

This gives a smooth, task-adaptive alternative to the discrete hierarchy
used in Phase 1-4. It is conceptually closer to Berkovich spaces
(Volume 4 section 10) than to the discrete Parisi hierarchy of Volume 4
section 4.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


def dim_at_scale(d_input: int, scale: float) -> int:
    """Return the projection dimension m(s) at continuous scale s."""
    if scale <= 0:
        return d_input
    factor = 2.0 ** (-scale)
    return max(1, min(d_input, math.ceil(d_input * factor)))


@dataclass
class ContinuousScaleProjector:
    """Orthogonal projector at a continuous scale s on a learned basis.

    The basis B ∈ ℝ^{d × d} is an orthonormal matrix whose columns are
    sorted by importance (typically eigenvectors of a covariance matrix in
    descending eigenvalue order). Projection at scale s keeps the top-m(s)
    columns.
    """
    d_input: int
    basis: np.ndarray                     # (d, d), orthonormal, desc-sorted
    scale: float = 0.0

    def dim(self) -> int:
        return dim_at_scale(self.d_input, self.scale)

    def set_scale(self, scale: float) -> None:
        """Smoothly adjust scale. Caller handles any smoothing across rounds."""
        self.scale = max(0.0, scale)

    def project(self, x: np.ndarray) -> np.ndarray:
        m = self.dim()
        B_m = self.basis[:, :m]
        return B_m.T @ x

    def reconstruct(self, y: np.ndarray) -> np.ndarray:
        m = len(y)
        B_m = self.basis[:, :m]
        return B_m @ y

    def write_cost(self) -> int:
        return self.dim()

    def read_cost(self) -> int:
        return self.dim()


@dataclass
class AdaptiveScale:
    """Track and adjust an agent's continuous scale based on surprise.

    Policy: EMA of surprise; if EMA > high threshold, lower scale (more
    detail); if EMA < low threshold, raise scale (more compression).
    Bounded in [0, max_scale].
    """
    max_scale: float = 4.0       # ~ log₂(16) — default 5-level equivalent
    current_scale: float = 2.0   # start at mid scale
    lr: float = 0.1              # scale adaptation rate
    high_threshold: float = 0.5
    low_threshold: float = 0.1
    _ema_surprise: float = 0.5

    def update(self, surprise: float) -> float:
        self._ema_surprise = 0.9 * self._ema_surprise + 0.1 * float(surprise)
        if self._ema_surprise > self.high_threshold:
            # Too much surprise — drop scale (increase m)
            self.current_scale = max(0.0, self.current_scale - self.lr)
        elif self._ema_surprise < self.low_threshold:
            # Stable — can compress more
            self.current_scale = min(self.max_scale, self.current_scale + self.lr)
        return self.current_scale
