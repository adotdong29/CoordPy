"""Basis-invariant neural-net predictor — Phase 3.

Each agent keeps a tiny 2-layer MLP that predicts its own next-step state
given the current state:

    x̂(t+1) = f_θ( x(t) )

Because the predictor lives in the original d-dim agent state space (not
in the PCA projection), rotations of the learned basis do NOT disturb it.
That was the Phase-2 bug: the predictor tracked y(t) = B(t)^T x(t), and
every time B rotated, all predictors looked maximally surprised even when
nothing had changed.

Surprise = ‖x(t+1) − x̂(t+1)‖. Written to bus only if surprise > τ.

Implementation: pure numpy, no torch. Layers:
    x   --[W1,b1]-->  h   --ReLU-->  h   --[W2,b2]-->  residual
    x̂(t+1) = x(t) + residual

Residual formulation is critical: a zero-residual predictor says "stay put",
which is the right bias for a slowly-drifting process. The net only learns
to predict deviations from persistence.

SGD with a very small learning rate for online stability.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class NeuralPredictor:
    dim: int
    hidden: int = 16
    lr: float = 0.005
    # Weights (initialized small so initial prediction ≈ "stay put")
    _W1: np.ndarray = None  # type: ignore  (dim, hidden)
    _b1: np.ndarray = None  # type: ignore  (hidden,)
    _W2: np.ndarray = None  # type: ignore  (hidden, dim)
    _b2: np.ndarray = None  # type: ignore  (dim,)
    _last_x: np.ndarray = None  # type: ignore
    _seeded: bool = False

    def _init_weights(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # He-like init, but shrunk so initial residual ~ 0.
        scale = 0.01
        self._W1 = scale * rng.standard_normal((self.dim, self.hidden))
        self._b1 = np.zeros(self.hidden)
        self._W2 = scale * rng.standard_normal((self.hidden, self.dim))
        self._b2 = np.zeros(self.dim)
        self._seeded = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._seeded:
            self._init_weights()
        h = x @ self._W1 + self._b1
        h_act = np.maximum(h, 0)         # ReLU
        residual = h_act @ self._W2 + self._b2
        return x + residual

    def observe(self, x_prev: np.ndarray, x_now: np.ndarray) -> float:
        """Observe the (prev → now) transition. Returns surprise.

        Surprise is ‖x_now − predict(x_prev)‖. Also takes one SGD step on
        squared prediction error.
        """
        if not self._seeded:
            self._init_weights()
        # Forward pass
        h = x_prev @ self._W1 + self._b1
        h_act = np.maximum(h, 0)
        residual = h_act @ self._W2 + self._b2
        pred = x_prev + residual
        err = x_now - pred                    # gradient of loss wrt pred
        surprise = float(np.linalg.norm(err))

        # Backprop: d_loss/d_residual = -err  (minimizing 0.5‖err‖²)
        d_residual = -err
        # W2 gradient: h_act[:,None] * d_residual[None,:]
        gW2 = np.outer(h_act, d_residual)
        gb2 = d_residual
        # Backprop to h: d_residual @ W2.T
        d_h_act = d_residual @ self._W2.T
        # Through ReLU
        d_h = d_h_act * (h > 0)
        gW1 = np.outer(x_prev, d_h)
        gb1 = d_h

        # Gradient clipping to prevent blow-up under shocks
        max_norm = 1.0
        for g in (gW1, gb1, gW2, gb2):
            n = np.linalg.norm(g)
            if n > max_norm:
                g *= max_norm / n

        # SGD step
        self._W1 -= self.lr * gW1
        self._b1 -= self.lr * gb1
        self._W2 -= self.lr * gW2
        self._b2 -= self.lr * gb2

        # Weight clipping — keep the net's effective range bounded so
        # predictions can't explode far beyond observed state magnitudes.
        w_max = 1.0
        np.clip(self._W1, -w_max, w_max, out=self._W1)
        np.clip(self._W2, -w_max, w_max, out=self._W2)
        return surprise

    @property
    def parameters(self) -> int:
        """Total learnable parameter count (for accounting)."""
        if not self._seeded:
            return 0
        return (self._W1.size + self._b1.size
                + self._W2.size + self._b2.size)
