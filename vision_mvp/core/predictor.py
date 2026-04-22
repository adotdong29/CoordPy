"""Per-agent predictor — a tiny world model.

Each agent maintains a local predictor of its own projected state in the
manifold. The predictor is a linear AR(1) model:

    ŷ(t+1) = α · y(t) + β

fitted with exponential-moving-average updates. When the actual projection
arrives, prediction error = ||y(t) - ŷ(t)|| is the surprise signal.

This is the degenerate (linear) form of Idea 2 from VISION_MILLIONS: each
agent has a generative world model and only writes when reality deviates
from prediction. With an LLM, the predictor would be the LLM itself.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LinearPredictor:
    dim: int
    lr_alpha: float = 0.1       # learning rate for α
    lr_beta: float = 0.1        # learning rate for β
    _alpha: float = 1.0         # persistence coefficient (starts at identity)
    _beta: np.ndarray = None    # type: ignore  (dim,) intercept
    _last_y: np.ndarray = None  # type: ignore  (dim,) last observed

    def __post_init__(self):
        if self._beta is None:
            self._beta = np.zeros(self.dim)

    def predict(self) -> np.ndarray:
        if self._last_y is None:
            return self._beta.copy()
        return self._alpha * self._last_y + self._beta

    def observe(self, y: np.ndarray) -> float:
        """Observe the new value. Returns prediction error (surprise)."""
        pred = self.predict()
        err = float(np.linalg.norm(y - pred))
        # Gradient-style updates to (α, β):
        # minimize ||y - (α y_prev + β)||²
        if self._last_y is not None:
            resid = y - (self._alpha * self._last_y + self._beta)
            # ∂/∂α  = -2 y_prev · resid
            grad_alpha = -float(np.dot(self._last_y, resid))
            self._alpha -= self.lr_alpha * grad_alpha / (np.dot(self._last_y, self._last_y) + 1e-6)
            # ∂/∂β  = -2 resid
            self._beta = self._beta + self.lr_beta * resid
        self._last_y = y.copy()
        return err
