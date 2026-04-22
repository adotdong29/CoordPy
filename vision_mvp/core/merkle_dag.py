"""Content-addressed Merkle DAG — SHA-256-backed object store with
proof-of-inclusion.

Benet (2014) IPFS / IPLD; same idea powers Git. Every block is keyed by the
SHA-256 of its content; links between blocks are just hash references. The
*same content* always produces the *same hash* — so dedup is automatic, and
inclusion proofs are a sibling-hash path (logarithmic in the store's depth).

Used in CASR as the substrate for "git-for-context": versioned, branchable,
content-addressed shared stores where agents cheaply share ancestry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _canonical_encode(obj: Any) -> bytes:
    """Deterministic JSON encoding so equal content → equal hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_hash(obj: Any) -> str:
    return hashlib.sha256(_canonical_encode(obj)).hexdigest()


@dataclass
class MerkleDAG:
    """In-memory content-addressed object store."""

    _store: dict[str, Any] = field(default_factory=dict)

    def put(self, obj: Any) -> str:
        """Canonical-encode `obj`, store, return its hash."""
        h = content_hash(obj)
        if h not in self._store:
            self._store[h] = obj
        return h

    def get(self, h: str) -> Any:
        if h not in self._store:
            raise KeyError(h)
        return self._store[h]

    def __contains__(self, h: str) -> bool:
        return h in self._store

    def __len__(self) -> int:
        return len(self._store)

    # ---------------- Merkle tree with inclusion proofs ------------------

    def build_merkle_tree(self, leaves: list[Any]) -> tuple[str, list[list[str]]]:
        """Hash-tree of `leaves`. Returns (root_hash, level_hashes).

        level_hashes[0] = leaf hashes, level_hashes[k+1] = parent hashes.
        """
        level = [content_hash(leaf) for leaf in leaves]
        levels = [level]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level = level + [level[-1]]       # duplicate last
            nxt = []
            for i in range(0, len(level), 2):
                combined = level[i] + level[i + 1]
                nxt.append(hashlib.sha256(combined.encode("utf-8")).hexdigest())
            levels.append(nxt)
            level = nxt
        return levels[-1][0], levels

    def inclusion_proof(
        self, levels: list[list[str]], leaf_index: int,
    ) -> list[tuple[str, str]]:
        """Sibling-hash path from `leaf_index` to root.

        Returns a list of (sibling_hash, side) where side ∈ {'L', 'R'}.
        """
        proof = []
        idx = leaf_index
        for lvl in levels[:-1]:
            sib_idx = idx ^ 1
            if sib_idx >= len(lvl):
                sib_idx = idx                   # duplicated last
            side = "L" if sib_idx < idx else "R"
            proof.append((lvl[sib_idx], side))
            idx //= 2
        return proof

    @staticmethod
    def verify_inclusion(
        leaf: Any, proof: list[tuple[str, str]], root: str,
    ) -> bool:
        """Verify that `leaf` is at the position encoded by `proof`."""
        h = content_hash(leaf)
        for sib, side in proof:
            if side == "L":
                combined = sib + h
            else:
                combined = h + sib
            h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return h == root
