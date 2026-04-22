"""Consistent and rendezvous hashing for dynamic-membership routing.

Classical hash-mod-N routing is fine until N changes: then ~all keys re-map.
Karger et al. (1997) consistent hashing: place N nodes on a hash ring and
route each key to the next node clockwise. Adding or removing a node moves
only K/N keys. Deployed throughout distributed-database land.

Rendezvous ("HRW") hashing (Thaler & Ravishankar 1998): each (key, node) pair
hashes to a value; route the key to the node with the largest hash. Requires
no ring data structure and distributes load perfectly evenly.

Both are simple to implement against `hashlib.blake2b` — no external deps —
and give CASR the dynamic-membership support the plan flags as missing.
"""

from __future__ import annotations

import bisect
import hashlib
from typing import Iterable


def _hash64(data: bytes, seed: int = 0) -> int:
    h = hashlib.blake2b(data, digest_size=8, key=seed.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


def _key_bytes(key: object) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, int):
        return key.to_bytes(8, "little", signed=True)
    return repr(key).encode("utf-8")


class ConsistentHashRing:
    """Dynamic-membership hash ring with virtual nodes for load balance."""

    def __init__(self, replicas: int = 64, seed: int = 0):
        if replicas < 1:
            raise ValueError("replicas must be ≥ 1")
        self.replicas = replicas
        self.seed = seed
        # sorted list of (hash, node_id) pairs
        self._ring: list[tuple[int, str]] = []
        self._nodes: set[str] = set()

    def add_node(self, node: str) -> None:
        if node in self._nodes:
            return
        for r in range(self.replicas):
            h = _hash64(f"{node}#{r}".encode(), self.seed)
            bisect.insort(self._ring, (h, node))
        self._nodes.add(node)

    def remove_node(self, node: str) -> None:
        if node not in self._nodes:
            return
        self._ring = [(h, n) for h, n in self._ring if n != node]
        self._nodes.discard(node)

    def nodes(self) -> set[str]:
        return set(self._nodes)

    def route(self, key: object) -> str:
        """Return the node responsible for `key`."""
        if not self._ring:
            raise RuntimeError("ring is empty")
        h = _hash64(_key_bytes(key), self.seed)
        idx = bisect.bisect_right(self._ring, (h, ""))
        if idx == len(self._ring):
            idx = 0
        return self._ring[idx][1]


def rendezvous_route(key: object, nodes: Iterable[str], seed: int = 0) -> str:
    """HRW / rendezvous hashing — pick node with max hash(key‖node).

    Routing with O(|nodes|) per lookup and perfect load balance in expectation.
    Missing data structure, so trivial to scale to dynamic membership.
    """
    nodes_list = list(nodes)
    if not nodes_list:
        raise ValueError("nodes must be non-empty")
    kb = _key_bytes(key)
    best_node = nodes_list[0]
    best_hash = -1
    for n in nodes_list:
        h = _hash64(kb + b"\x00" + n.encode("utf-8"), seed)
        if h > best_hash:
            best_hash = h
            best_node = n
    return best_node
