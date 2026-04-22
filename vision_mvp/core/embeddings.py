"""Metric embeddings: Johnson–Lindenstrauss and Bourgain.

These are *baselines* for the learned streaming-PCA manifold. Their tight
log-N distortion bounds are a key part of the theoretical argument that
O(log N) manifold dimension is not arbitrary. Shipping them as sibling
classes in the embedding family gives a head-to-head empirical comparison
on CASR tasks: does learned PCA actually beat JL/Bourgain, or are we just
paying extra for no gain?

JL (Johnson & Lindenstrauss 1984):
    For any N points in ℝ^d and ε ∈ (0, 1), there exists a linear map into
    ℝ^k with k = O(log N / ε²) preserving pairwise distances up to (1±ε).
    Construction: random Gaussian matrix scaled by 1/√k.

Bourgain (1985):
    Every N-point metric space embeds into ℓ² with distortion O(log N),
    and this is tight. Construction: embed via distances to O(log N) random
    subset-reference points, stacked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JLEmbedding:
    """Gaussian JL projection ℝ^d → ℝ^k."""

    in_dim: int
    out_dim: int
    seed: int = 0
    _W: np.ndarray = None  # type: ignore

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # Entries N(0, 1/k)
        self._W = rng.standard_normal((self.in_dim, self.out_dim)) / np.sqrt(self.out_dim)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x @ self._W

    @classmethod
    def for_n_points(cls, n_points: int, in_dim: int, eps: float = 0.1, seed: int = 0):
        """Output dimension per JL lemma: k = ⌈8 ln N / ε²⌉."""
        if not 0 < eps < 1:
            raise ValueError("eps must be in (0, 1)")
        if n_points < 2:
            raise ValueError("need at least 2 points")
        k = int(np.ceil(8 * np.log(n_points) / (eps * eps)))
        k = min(k, in_dim)
        return cls(in_dim=in_dim, out_dim=k, seed=seed)


@dataclass
class BourgainEmbedding:
    """Bourgain embedding via distances to O(log²N) random subsets.

    For a finite metric (X, d), Bourgain's construction picks log N levels
    i = 1..log₂ N; at each level pick 2^i random subsets, each of size log N
    picked uniformly from X. The coordinate function is the distance to the
    subset; the full embedding is the vector of all such coordinates divided
    by √(log N). Gives O(log N) distortion.

    For our typical use (N ≤ 1e5, vector data in ℝ^d), we approximate: pick
    `n_landmarks = ⌈log₂ N · log₂ N⌉` random subsets of the training data, and
    map any query to its vector of distances to these subsets.
    """

    landmarks: np.ndarray    # (L, d) subset centroids
    _scale: float = 1.0

    @classmethod
    def fit(cls, points: np.ndarray, seed: int = 0) -> "BourgainEmbedding":
        P = np.asarray(points, dtype=float)
        N, d = P.shape
        levels = max(2, int(np.ceil(np.log2(N))))
        n_landmarks = levels * levels
        rng = np.random.default_rng(seed)
        landmarks = np.zeros((n_landmarks, d))
        idx = 0
        for i in range(1, levels + 1):
            # 2^i subsets (capped by levels)
            n_sub = min(2 ** i, levels)
            for _ in range(n_sub):
                if idx >= n_landmarks:
                    break
                size = max(1, int(np.round(N / (2 ** i))))
                subset = rng.choice(N, size=size, replace=False)
                landmarks[idx] = P[subset].mean(axis=0)
                idx += 1
        # Trim to what we actually filled
        landmarks = landmarks[:idx]
        return cls(landmarks=landmarks, _scale=1.0 / max(np.sqrt(np.log(max(N, 2))), 1.0))

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            dists = np.linalg.norm(self.landmarks - x[None, :], axis=1)
            return dists * self._scale
        return (
            np.linalg.norm(self.landmarks[None, :, :] - x[:, None, :], axis=2)
            * self._scale
        )

    @property
    def out_dim(self) -> int:
        return self.landmarks.shape[0]


def distortion(
    embedding_fn,
    points: np.ndarray,
    n_pairs: int = 200,
    seed: int = 0,
) -> float:
    """Empirical worst-case distortion over `n_pairs` random pairs."""
    P = np.asarray(points, dtype=float)
    N = P.shape[0]
    rng = np.random.default_rng(seed)
    max_ratio = 1.0
    min_ratio = 1.0
    for _ in range(n_pairs):
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        orig = float(np.linalg.norm(P[i] - P[j]))
        e_i = embedding_fn(P[i])
        e_j = embedding_fn(P[j])
        emb = float(np.linalg.norm(e_i - e_j))
        if orig < 1e-12:
            continue
        ratio = emb / orig
        max_ratio = max(max_ratio, ratio)
        min_ratio = min(min_ratio, ratio) if min_ratio > 0 else min_ratio
    # distortion = sup(ratio) / inf(ratio)
    return max_ratio / max(min_ratio, 1e-12)
