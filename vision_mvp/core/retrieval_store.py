"""Retrieval-augmented coordination — in-memory vector store with brute-force
k-NN as the pure-NumPy fallback (hnswlib-less).

Each agent writes (embedding, payload) tuples into a shared store; other
agents query by k-NN on embedding. This inverts the push-broadcast pattern
of Idea 8 (market routing) into a *pull* pattern: recipients fetch only what
they need.

For production-scale, swap `BruteForceVectorStore` for a HNSW backend via
the `hnswlib` library; the interface is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Record:
    embedding: np.ndarray
    payload: Any
    metadata: dict = field(default_factory=dict)


class BruteForceVectorStore:
    """Simple cosine-similarity k-NN over stored embeddings."""

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError("dim must be ≥ 1")
        self.dim = dim
        self._records: list[Record] = []
        self._matrix: np.ndarray | None = None       # (M, d) cached
        self._dirty = True

    def add(self, embedding: np.ndarray, payload: Any,
            metadata: dict | None = None) -> int:
        e = np.asarray(embedding, dtype=float).ravel()
        if e.size != self.dim:
            raise ValueError(f"embedding must be dim {self.dim}")
        self._records.append(Record(
            embedding=e, payload=payload,
            metadata=dict(metadata) if metadata else {},
        ))
        self._dirty = True
        return len(self._records) - 1

    def __len__(self) -> int:
        return len(self._records)

    def _refresh_matrix(self) -> None:
        if not self._dirty:
            return
        if not self._records:
            self._matrix = np.zeros((0, self.dim))
        else:
            self._matrix = np.stack([r.embedding for r in self._records], axis=0)
        self._dirty = False

    def knn(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float, Record]]:
        """Return top-k records by cosine similarity, most similar first."""
        q = np.asarray(query, dtype=float).ravel()
        if q.size != self.dim:
            raise ValueError(f"query must be dim {self.dim}")
        self._refresh_matrix()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        norms = np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(q) + 1e-12
        sims = (self._matrix @ q) / norms
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i]), self._records[int(i)]) for i in order]

    def clear(self) -> None:
        self._records.clear()
        self._matrix = None
        self._dirty = True
