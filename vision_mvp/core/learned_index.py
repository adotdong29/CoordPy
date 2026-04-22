"""Learned index — piecewise-linear model predicts position-in-sorted-array.

Kraska et al. (SIGMOD 2018) "The Case for Learned Index Structures": replace
B-tree lookups with a small model that predicts a key's approximate position
in a sorted array. Correction is a bounded linear scan around the prediction.

For our case (CASR causal footprints): monotonic keys (timestamps, sorted
message ids) fit a piecewise-linear regressor in < 1 KB, giving O(1)-expected
lookup with a small known error envelope. Concretely:

  predict(key) = floor(a_i · key + b_i)   where i is the segment
  actual_pos ∈ [predict(key) - max_err, predict(key) + max_err]

All pure NumPy; trained once at build, constant-time query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class LearnedIndex:
    """Two-stage linear learned index with per-segment correction envelope."""

    keys: np.ndarray                      # sorted ascending
    n_segments: int = 16
    _segment_starts: np.ndarray = field(init=False)
    _segment_slopes: np.ndarray = field(init=False)
    _segment_intercepts: np.ndarray = field(init=False)
    _max_error: int = field(init=False, default=0)

    def __post_init__(self):
        k = np.asarray(self.keys, dtype=float)
        if not np.all(np.diff(k) >= 0):
            raise ValueError("keys must be sorted ascending")
        n = k.size
        # Segment endpoints equi-spaced by quantile
        q = np.linspace(0, n, self.n_segments + 1).astype(int)
        self._segment_starts = np.array([float(k[max(0, q[i] - 1)]) for i in range(self.n_segments)])
        slopes = np.zeros(self.n_segments)
        ints = np.zeros(self.n_segments)
        # Fit a simple linear regression per segment over (key -> position)
        for i in range(self.n_segments):
            a, b = q[i], q[i + 1]
            if b - a < 2:
                slopes[i], ints[i] = 0.0, float(a)
                continue
            x = k[a:b]
            y = np.arange(a, b, dtype=float)
            xm, ym = x.mean(), y.mean()
            denom = float(((x - xm) ** 2).sum())
            if denom < 1e-12:
                slopes[i] = 0.0
                ints[i] = ym
            else:
                slopes[i] = float(((x - xm) * (y - ym)).sum()) / denom
                ints[i] = ym - slopes[i] * xm
        self._segment_slopes = slopes
        self._segment_intercepts = ints
        # Max per-segment prediction error
        predicted = np.zeros(n, dtype=float)
        for i in range(self.n_segments):
            a, b = q[i], q[i + 1]
            predicted[a:b] = slopes[i] * k[a:b] + ints[i]
        errors = np.abs(predicted - np.arange(n))
        self._max_error = int(np.ceil(errors.max())) if errors.size else 0

    def predict(self, key: float) -> int:
        """Return predicted position in the sorted array."""
        seg = int(np.searchsorted(self._segment_starts, key, side="right") - 1)
        seg = max(0, min(seg, self.n_segments - 1))
        p = int(round(self._segment_slopes[seg] * key + self._segment_intercepts[seg]))
        return max(0, min(p, self.keys.size - 1))

    def find(self, key: float) -> int:
        """Binary-search within the predicted error envelope. Returns −1 if
        `key` is not present.
        """
        p = self.predict(key)
        lo = max(0, p - self._max_error - 1)
        hi = min(self.keys.size - 1, p + self._max_error + 1)
        # numpy searchsorted over the narrow range
        window = self.keys[lo:hi + 1]
        rel = np.searchsorted(window, key)
        if rel < window.size and window[rel] == key:
            return int(lo + rel)
        return -1

    @property
    def max_error(self) -> int:
        return self._max_error
