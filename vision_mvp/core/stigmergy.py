"""Stigmergic environment — CRDT-style shared state.

From VISION_MILLIONS Idea 3: agents coordinate through a shared environment
that they read and write locally. No direct agent-to-agent communication.

We implement a simple structure:
  - M bins (cells) on the manifold
  - Each bin is a CRDT register with (value, weight, version)
  - Agents with projected coords near bin b read bin b, write to bin b
  - Reads return only the cells in a local neighborhood (locality)

Crucial: agents only write when their local delta exceeds a threshold τ.
This is the CASR Stage 3 surprise filter — only transmit what's novel.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class StigmergyBin:
    value: np.ndarray       # sum in this bin
    weight: float = 0.0     # count/weight
    version: int = 0

    def merge(self, other_value: np.ndarray, other_weight: float) -> None:
        self.value = self.value + other_value
        self.weight += other_weight
        self.version += 1

    def mean(self) -> np.ndarray:
        if self.weight <= 0:
            return np.zeros_like(self.value)
        return self.value / self.weight


@dataclass
class Stigmergy:
    """Environment = grid of bins indexed by projected manifold coords.

    Bins partition the manifold into regions. An agent writes only to its
    "home" bin (nearest bin in projected coords), and reads a local
    neighborhood of bins. This is the physical locality assumption that
    makes coordination sublinear.
    """
    n_bins: int
    dim: int
    _bins: list[StigmergyBin] = field(default_factory=list)
    # A set of "anchor" points in manifold space, one per bin
    _anchors: np.ndarray = None  # type: ignore

    @classmethod
    def build(cls, n_bins: int, dim: int, seed: int = 0) -> "Stigmergy":
        rng = np.random.default_rng(seed)
        # Anchors placed at random in a unit cube; in practice could be
        # adaptive (k-means on observed projections)
        anchors = rng.standard_normal((n_bins, dim))
        bins = [StigmergyBin(value=np.zeros(dim), weight=0.0) for _ in range(n_bins)]
        return cls(n_bins=n_bins, dim=dim, _bins=bins, _anchors=anchors)

    def bin_of(self, coords: np.ndarray) -> int:
        """Find nearest bin to given manifold coords."""
        d = np.linalg.norm(self._anchors - coords[None, :], axis=1)
        return int(np.argmin(d))

    def write(self, coords: np.ndarray, value: np.ndarray, weight: float = 1.0) -> int:
        """Write a value to the bin nearest to `coords`. Returns the bin id."""
        b = self.bin_of(coords)
        self._bins[b].merge(value, weight)
        return b

    def read_local(self, coords: np.ndarray, k: int) -> np.ndarray:
        """Read the k nearest bins to `coords`, return aggregated mean."""
        d = np.linalg.norm(self._anchors - coords[None, :], axis=1)
        idx = np.argsort(d)[:k]
        total_val = np.zeros(self.dim)
        total_wt = 0.0
        for i in idx:
            b = self._bins[i]
            total_val = total_val + b.value
            total_wt += b.weight
        if total_wt <= 0:
            return np.zeros(self.dim)
        return total_val / total_wt

    def read_cost(self, k: int) -> int:
        """Tokens to read k nearest bins (each bin = dim floats)."""
        return k * self.dim

    def write_cost(self) -> int:
        """Tokens to write one bin (dim floats + metadata)."""
        return self.dim + 1
