"""Public API — the one class most users will ever touch.

Usage
-----

    from vision_mvp import CASRRouter
    import numpy as np

    router = CASRRouter(n_agents=1000, state_dim=64, task_rank=10)

    for t in range(100):
        obs = np.random.randn(1000, 64)    # (N, d) noisy observations
        estimates = router.step(obs)        # (N, d) consensus estimates

    print(router.stats)

Behind the scenes this runs the Phase-3 hierarchical protocol with streaming
PCA, a vectorized neural predictor, and a Global-Workspace admission
mechanism — giving O(log N) peak per-agent context and bounded writes per
round.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field

from .core.bus import Bus
from .core.learned_manifold import StreamingPCA
from .core.vectorized_predictor import PredictorBank
from .core.workspace import Workspace


@dataclass
class CASRRouter:
    """Black-box routing layer for an N-agent coordination task.

    Parameters
    ----------
    n_agents : int
        Team size N.
    state_dim : int
        Dimensionality d of each agent's state vector.
    task_rank : int | None
        Expected rank of the task-relevant subspace. If None, defaults to
        ⌈log₂ N⌉ (the CASR-theoretic optimum for consensus tasks).
    observation_noise : float
        Standard deviation σ of per-observation noise — used for Bayesian
        weighting.
    surprise_threshold : float
        Only agents with prediction error above this threshold are admitted
        to the shared workspace (on top of top-k selection).
    decay : float
        Exponential-decay factor on the shared register (0, 1). Smaller =
        faster forgetting of stale evidence; larger = longer memory.
        Default 0.8 is a reasonable tracking-speed/stability balance.
    pca_lr : float
        Streaming-PCA learning rate (EMA coefficient on covariance).
    predictor_hidden : int
        Hidden dimension of each agent's neural predictor.
    predictor_lr : float
        SGD learning rate for the per-agent neural predictor.
    seed : int
        PRNG seed; all randomness is deterministic given this.
    """
    n_agents: int
    state_dim: int
    task_rank: int | None = None
    observation_noise: float = 1.0
    surprise_threshold: float = 0.5
    decay: float = 0.8
    pca_lr: float = 0.1
    predictor_hidden: int = 8
    predictor_lr: float = 0.005
    seed: int = 0

    # Internal state (hidden)
    _manifold_dim: int = 0
    _round_idx: int = 0
    _pca: StreamingPCA = field(default=None)          # type: ignore
    _bank: PredictorBank = field(default=None)        # type: ignore
    _workspace: Workspace = field(default=None)       # type: ignore
    _bus: Bus = field(default=None)                   # type: ignore
    _estimates: np.ndarray = field(default=None)      # type: ignore
    _weights: np.ndarray = field(default=None)        # type: ignore
    _prev_estimates: np.ndarray = field(default=None) # type: ignore
    _reg_value: np.ndarray = field(default=None)      # type: ignore
    _reg_weight: float = 0.0
    _initialized: bool = False

    def __post_init__(self):
        if self.n_agents < 2:
            raise ValueError("n_agents must be ≥ 2")
        if self.state_dim < 1:
            raise ValueError("state_dim must be ≥ 1")
        self._manifold_dim = (self.task_rank
                              if self.task_rank is not None
                              else max(2, math.ceil(math.log2(self.n_agents))))
        self._manifold_dim = min(self._manifold_dim, self.state_dim)

        self._pca = StreamingPCA.build(
            self.state_dim, self._manifold_dim,
            lr=self.pca_lr, seed=self.seed,
        )
        self._bank = PredictorBank.build(
            self.n_agents, self.state_dim,
            hidden=self.predictor_hidden, lr=self.predictor_lr,
            seed=self.seed,
        )
        self._workspace = Workspace(
            n_agents=self.n_agents, epsilon=0.05,
        )
        self._bus = Bus()
        self._weights = np.ones(self.n_agents) / (self.observation_noise ** 2)
        self._reg_value = np.zeros(self._manifold_dim)

    # ---------- Primary API ----------

    def step(self, observations: np.ndarray) -> np.ndarray:
        """Execute one synchronous round with all N observations.

        Parameters
        ----------
        observations : (N, d) float array.
            One row per agent.

        Returns
        -------
        estimates : (N, d) float array.
            Each agent's post-routing consensus estimate.
        """
        if observations.ndim != 2:
            raise ValueError(f"observations must be 2D, got {observations.shape}")
        if observations.shape != (self.n_agents, self.state_dim):
            raise ValueError(
                f"observations shape {observations.shape} != "
                f"(n_agents, state_dim)={self.n_agents, self.state_dim}"
            )
        if not self._initialized:
            # Cold-start: initialize each agent's estimate to its first
            # observation.
            self._estimates = observations.copy()
            self._prev_estimates = observations.copy()
            self._initialized = True

        # --- adaptive round (mirrors protocols/hierarchical.py) ---
        forget = 0.4
        new_est = (1 - forget) * self._estimates + forget * observations
        self._pca.update(observations.mean(axis=0))

        self._reg_value = self._reg_value * self.decay
        self._reg_weight = self._reg_weight * self.decay

        pred = self._bank.predict(self._prev_estimates)
        saliences = np.linalg.norm(new_est - pred, axis=1)

        admitted = self._workspace.select(saliences, seed=self.seed + self._round_idx)
        for i in admitted:
            if saliences[i] > self.surprise_threshold:
                y_i = self._pca.project(new_est[int(i)])
                self._reg_value = self._reg_value + self._weights[int(i)] * y_i
                self._reg_weight = self._reg_weight + self._weights[int(i)]
                self._bus.send(int(i), -1, self._manifold_dim + 1, "write",
                               self._round_idx)

        summary = self._reg_value / max(self._reg_weight, 1e-8)
        reconstructed = self._pca.reconstruct(summary)
        for i in range(self.n_agents):
            self._bus.send(-1, i, self._manifold_dim, "read", self._round_idx)
            self._bus.note_context(i, self._manifold_dim)

        total_w = self._weights + self._reg_weight
        safe_w = np.where(total_w > 0, total_w, 1.0)
        merged = (self._weights[:, None] * new_est
                  + self._reg_weight * reconstructed[None, :]) / safe_w[:, None]

        self._bank.observe(self._prev_estimates, merged)
        self._prev_estimates = merged.copy()
        self._estimates = merged
        self._round_idx += 1
        return merged.copy()

    # ---------- Queries ----------

    @property
    def estimates(self) -> np.ndarray:
        """Current (N, d) consensus estimates."""
        return self._estimates.copy() if self._estimates is not None else None

    def get_estimate(self, agent_id: int) -> np.ndarray:
        """Return the consensus estimate held by one particular agent."""
        if self._estimates is None:
            raise RuntimeError("Call step() at least once before querying.")
        if not 0 <= agent_id < self.n_agents:
            raise IndexError(agent_id)
        return self._estimates[agent_id].copy()

    @property
    def stats(self) -> dict:
        """Bus & scaling statistics.

        Returns
        -------
        dict with keys:
            peak_context_per_agent : int — ≤ manifold_dim by construction
            total_tokens           : int
            total_messages         : int
            mean_context_per_agent : float
            manifold_dim           : int (= ⌈log₂ N⌉ by default)
            workspace_size         : int (= ⌈log₂ N⌉ by default)
            rounds_executed        : int
        """
        s = self._bus.summary()
        return {
            "peak_context_per_agent": int(s["peak_agent_context"]),
            "total_tokens": int(s["total_tokens"]),
            "total_messages": int(s["n_messages"]),
            "mean_context_per_agent": float(s["mean_agent_context"]),
            "manifold_dim": int(self._manifold_dim),
            "workspace_size": int(self._workspace.capacity()),
            "rounds_executed": int(self._round_idx),
        }

    def reset(self) -> None:
        """Clear all state and reinitialize. Useful for repeated experiments."""
        self.__post_init__()
        self._round_idx = 0
        self._estimates = None
        self._prev_estimates = None
        self._initialized = False
