"""Probabilistic sketches: Count-Min, HyperLogLog, reservoir sampling.

Streaming summaries with sublinear memory that give provable bounds on the
error of their estimates. Used to augment `bus.py` telemetry: count per-agent
traffic, estimate number of unique event types, reservoir-sample payloads
for offline analysis — all without growing memory with stream length.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np


def _hash_pair(data: bytes, seed: int) -> int:
    h = hashlib.blake2b(data, digest_size=8, key=seed.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


def _to_bytes(x: object) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8")
    if isinstance(x, int):
        return x.to_bytes(8, "little", signed=True)
    return repr(x).encode("utf-8")


# ================================================================== Count-Min

class CountMinSketch:
    """Count-Min sketch (Cormode & Muthukrishnan 2004) for frequency estimates.

    Parameters:
      width  w = ⌈e / ε⌉       — controls additive error
      depth  d = ⌈ln(1 / δ)⌉   — controls failure prob

    Estimate ĉ(x) satisfies ĉ ≥ c and ĉ ≤ c + ε N with prob 1 − δ,
    where N is the total number of updates.
    """

    def __init__(self, width: int = 1024, depth: int = 5, seed: int = 0):
        if width < 1 or depth < 1:
            raise ValueError("width, depth must be ≥ 1")
        self.width, self.depth = width, depth
        self.seed = seed
        self._table = np.zeros((depth, width), dtype=np.int64)
        self._n_updates = 0

    def _cols(self, x: object) -> list[int]:
        b = _to_bytes(x)
        return [(_hash_pair(b, self.seed + r) % self.width) for r in range(self.depth)]

    def update(self, x: object, count: int = 1) -> None:
        self._n_updates += count
        for r, c in enumerate(self._cols(x)):
            self._table[r, c] += count

    def estimate(self, x: object) -> int:
        return int(min(self._table[r, c] for r, c in enumerate(self._cols(x))))

    @classmethod
    def for_accuracy(cls, eps: float = 0.01, delta: float = 0.01, seed: int = 0):
        w = int(math.ceil(math.e / eps))
        d = int(math.ceil(math.log(1 / delta)))
        return cls(width=w, depth=d, seed=seed)


# ================================================================= HyperLogLog

class HyperLogLog:
    """Flajolet et al. (2007) cardinality estimator.

    Memory = 2^p bytes; relative error ≈ 1.04 / √(2^p). p=10 → 1024 bytes,
    ~3% error; p=14 → 16 KB, ~1% error.
    """

    def __init__(self, p: int = 12, seed: int = 0):
        if not 4 <= p <= 18:
            raise ValueError("p must be in [4, 18]")
        self.p = p
        self.m = 1 << p
        self.seed = seed
        self._registers = np.zeros(self.m, dtype=np.int8)

    @property
    def _alpha(self) -> float:
        if self.m >= 128:
            return 0.7213 / (1 + 1.079 / self.m)
        return {16: 0.673, 32: 0.697, 64: 0.709}.get(self.m, 0.7213)

    def add(self, x: object) -> None:
        h = _hash_pair(_to_bytes(x), self.seed)
        # top p bits = bucket, remaining bits = stream for leading-zero count
        idx = h >> (64 - self.p)
        w = (h << self.p) & ((1 << 64) - 1)
        if w == 0:
            rank = 64 - self.p + 1
        else:
            rank = (w.bit_length() ^ 64) + 1
            # leading-zero count + 1
            # bit_length returns position of highest set bit; we want zeros.
            rank = 64 - w.bit_length() + 1
        if rank > self._registers[idx]:
            self._registers[idx] = rank

    def estimate(self) -> int:
        # Harmonic mean of 2^registers
        z = float(np.sum(2.0 ** (-self._registers.astype(float))))
        raw = self._alpha * self.m * self.m / z
        # Small-range correction
        v = int((self._registers == 0).sum())
        if raw <= 2.5 * self.m and v > 0:
            raw = self.m * math.log(self.m / v)
        return int(round(raw))


# ============================================================== Reservoir

class ReservoirSampler:
    """Vitter's algorithm R — uniform-random sample of size k over a stream."""

    def __init__(self, k: int, seed: int = 0):
        if k < 1:
            raise ValueError("k must be ≥ 1")
        self.k = k
        self._sample: list = []
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def add(self, item) -> None:
        self._seen += 1
        if len(self._sample) < self.k:
            self._sample.append(item)
            return
        # Replace with probability k / seen
        j = int(self._rng.integers(0, self._seen))
        if j < self.k:
            self._sample[j] = item

    def sample(self) -> list:
        return list(self._sample)
