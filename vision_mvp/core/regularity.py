"""Frieze-Kannan weak regularity via SVD.

Szemerédi's regularity lemma (1975) partitions any large graph into O(1)
blocks that behave pseudo-randomly. The Frieze-Kannan (1999) *weak* variant
gives a similar result based on the matrix cut-norm, which is approximable
within a constant factor by the spectral norm:

    ‖A − Σ_k σ_k u_k v_k^T‖_cut ≤ O(‖residual‖_2)

Concretely: take the top-k singular components of the (N×N) adjacency, keep
them, reconstruct, and the per-block adjacency densities will be within
ε = O(σ_{k+1}) of the smoothed reconstruction. The block partition is read off
from the *sign patterns* of u_k, v_k: worlds that co-cluster in the top
singular vectors form regular blocks.

We use this as a block-routing primitive in CASR: instead of O(N²) routing
computations, group agents into O(1) regular blocks and route at the block
level. Pure-numpy SVD is sufficient for N ≤ 10⁴; scipy.sparse.linalg.svds
is a drop-in swap for larger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegularityPartition:
    block_labels: np.ndarray   # (N,) int in [0, n_blocks)
    n_blocks: int
    residual_norm: float       # spectral norm of the residual after truncation
    singular_values: np.ndarray

    def summary(self) -> str:
        return (
            f"{self.n_blocks} blocks, "
            f"residual ‖·‖₂ = {self.residual_norm:.4f}"
        )


def weak_regularity_partition(
    adjacency: np.ndarray,
    n_blocks: int = 4,
    top_k: int | None = None,
) -> RegularityPartition:
    """Frieze-Kannan-style partition via sign patterns of top singular vectors.

    top_k defaults to ⌈log2(n_blocks)⌉ singular vectors — enough bits to
    distinguish n_blocks cluster labels.
    """
    A = np.asarray(adjacency, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    N = A.shape[0]
    if top_k is None:
        top_k = max(1, int(np.ceil(np.log2(max(n_blocks, 2)))))

    # Full SVD (fine at small scale; replace with svds for larger)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    top_k = min(top_k, s.size)
    # sign bits of the top_k left singular vectors give an integer label
    bits = (U[:, :top_k] > 0).astype(int)
    labels = bits @ (1 << np.arange(top_k))
    # Remap to a compact [0..n_blocks) range
    uniq, compact = np.unique(labels, return_inverse=True)
    labels = compact
    # If we wanted fewer blocks, greedily merge the smallest
    while uniq.size > n_blocks:
        counts = np.bincount(labels)
        smallest = int(np.argmin(counts))
        # Merge `smallest` into the nearest block (in terms of mean-row distance)
        means = np.stack([A[labels == k].mean(axis=0) for k in range(len(uniq))])
        distances = np.linalg.norm(means - means[smallest], axis=1)
        distances[smallest] = np.inf
        target = int(np.argmin(distances))
        labels[labels == smallest] = target
        uniq, compact = np.unique(labels, return_inverse=True)
        labels = compact

    # Residual: reconstruct with top_k components, measure what's left
    A_approx = (U[:, :top_k] * s[:top_k]) @ Vt[:top_k]
    residual = A - A_approx
    res_norm = float(np.linalg.norm(residual, 2))

    return RegularityPartition(
        block_labels=labels,
        n_blocks=int(uniq.size),
        residual_norm=res_norm,
        singular_values=s,
    )
