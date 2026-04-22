"""Hyperdimensional / Vector Symbolic Architecture (VSA) computing.

Kanerva (2009); Plate (1995). A VSA uses very-high-dimensional (≥ 10⁴) random
vectors with three compositional operations:

  bind(a, b)        — element-wise multiply (bipolar) or XOR (binary);
                      resulting vector is near-orthogonal to both operands.
  bundle({a, b, …}) — element-wise majority (bipolar) or sum-and-threshold;
                      resulting vector is similar to each operand.
  permute(a, k)     — fixed permutation (cyclic roll) by k;
                      generates a new vector uncorrelated with a.

Key property: with d = 10⁴ bipolar entries, random vectors have cosine ≈ 0
with each other (concentration of measure on the hypercube). Compositional
structures are recoverable via exact-or-approximate unbinding.

Useful as an alternative to the low-rank manifold in SLM (Idea 1): every
agent state is a compositional HDC vector of fixed dimension, independent of
how much content has been packed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def random_bipolar(d: int, seed: int = 0) -> np.ndarray:
    """A random ±1 bipolar hypervector of dimension d."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=d) * 2 - 1).astype(np.int8)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiply (bipolar XOR)."""
    a = np.asarray(a, dtype=np.int8)
    b = np.asarray(b, dtype=np.int8)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    return (a * b).astype(np.int8)


def bundle(vectors: list[np.ndarray]) -> np.ndarray:
    """Element-wise majority-sign of bipolar vectors."""
    if not vectors:
        raise ValueError("need at least one vector")
    stack = np.stack([np.asarray(v, dtype=np.int32) for v in vectors], axis=0)
    s = stack.sum(axis=0)
    # ties broken toward +1
    result = np.where(s >= 0, 1, -1).astype(np.int8)
    return result


def permute(a: np.ndarray, k: int = 1) -> np.ndarray:
    """Cyclic shift by k positions — generates a new uncorrelated vector."""
    return np.roll(np.asarray(a, dtype=np.int8), k)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = float(a @ b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return num / denom


@dataclass
class CodeBook:
    """Named symbol table mapping symbol names to hypervectors."""

    d: int
    seed: int = 0
    _symbols: dict[str, np.ndarray] = None  # type: ignore

    def __post_init__(self):
        self._symbols = {}

    def ensure(self, name: str) -> np.ndarray:
        if name not in self._symbols:
            # deterministic: hash name and base seed
            h = (hash(name) ^ self.seed) & 0xFFFFFFFF
            self._symbols[name] = random_bipolar(self.d, seed=h)
        return self._symbols[name]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.ensure(name)

    def cleanup(self, noisy: np.ndarray) -> tuple[str, float]:
        """Find the stored symbol most similar to `noisy`. Returns (name, cos)."""
        if not self._symbols:
            return "", 0.0
        best_name, best_sim = "", -np.inf
        for name, vec in self._symbols.items():
            s = cosine(noisy, vec)
            if s > best_sim:
                best_name, best_sim = name, s
        return best_name, best_sim
