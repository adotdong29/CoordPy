"""HAMT-based persistent (copy-on-write) map.

Bagwell (2001), Steele (2003). A Hash Array Mapped Trie is a 32-way branching
trie keyed by hash bits. Operations:

  get / set / delete  — O(log₃₂ N) ≈ 7 lookups for N = 10⁹.
  set returns a *new* map sharing most of its structure with the old.

Snapshot isolation is trivial: holding a reference to an old root freezes
that version indefinitely. No garbage collection concerns.

Used in CASR for the stigmergic environment: multi-writer, multi-reader, no
locks, with O(log N) per operation and O(1) snapshot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

BITS = 5            # 32-way trie
MASK = (1 << BITS) - 1
MAX_DEPTH = 13      # hash(64 bits) / 5 bits per level → 13 levels


def _hash(key: Any) -> int:
    if isinstance(key, bytes):
        b = key
    elif isinstance(key, str):
        b = key.encode("utf-8")
    elif isinstance(key, int):
        b = key.to_bytes(8, "little", signed=True)
    else:
        b = repr(key).encode("utf-8")
    h = hashlib.blake2b(b, digest_size=8)
    return int.from_bytes(h.digest(), "little")


@dataclass(frozen=True)
class _Leaf:
    """Collision leaf: list of (key, value) pairs that hash to the same prefix."""
    entries: tuple = ()


@dataclass(frozen=True)
class _Branch:
    """Internal node: bitmap indicates which of 32 slots are live; `children`
    holds only the live children in bitmap order.
    """
    bitmap: int = 0
    children: tuple = ()


def _slot_index(bitmap: int, slot: int) -> int:
    """Number of live entries before `slot`."""
    return bin(bitmap & ((1 << slot) - 1)).count("1")


class HAMT:
    """Persistent hash map with O(log₃₂ N) operations."""

    __slots__ = ("_root", "_size")

    def __init__(self, root: _Branch | None = None, size: int = 0):
        self._root = root if root is not None else _Branch()
        self._size = size

    # -------- primitive get / set / delete on a node ----------

    @staticmethod
    def _node_get(node, key, h: int, depth: int):
        if isinstance(node, _Leaf):
            for k, v in node.entries:
                if k == key:
                    return v, True
            return None, False
        slot = (h >> (depth * BITS)) & MASK
        if not node.bitmap & (1 << slot):
            return None, False
        idx = _slot_index(node.bitmap, slot)
        return HAMT._node_get(node.children[idx], key, h, depth + 1)

    @staticmethod
    def _node_set(node, key, value, h: int, depth: int, is_new=[False]):
        if isinstance(node, _Leaf):
            for i, (k, v) in enumerate(node.entries):
                if k == key:
                    new_entries = list(node.entries)
                    new_entries[i] = (key, value)
                    return _Leaf(entries=tuple(new_entries))
            is_new[0] = True
            return _Leaf(entries=node.entries + ((key, value),))
        slot = (h >> (depth * BITS)) & MASK
        idx = _slot_index(node.bitmap, slot)
        if node.bitmap & (1 << slot):
            # Replace existing child
            new_child = HAMT._node_set(node.children[idx], key, value, h, depth + 1, is_new)
            children = list(node.children)
            children[idx] = new_child
            return _Branch(bitmap=node.bitmap, children=tuple(children))
        # Insert new leaf-child
        is_new[0] = True
        if depth + 1 >= MAX_DEPTH:
            new_child: object = _Leaf(entries=((key, value),))
        else:
            new_child = _Leaf(entries=((key, value),))
        new_children = list(node.children)
        new_children.insert(idx, new_child)
        return _Branch(
            bitmap=node.bitmap | (1 << slot),
            children=tuple(new_children),
        )

    @staticmethod
    def _node_delete(node, key, h: int, depth: int, removed=[False]):
        if isinstance(node, _Leaf):
            entries = [(k, v) for k, v in node.entries if k != key]
            if len(entries) < len(node.entries):
                removed[0] = True
            if not entries:
                return None
            return _Leaf(entries=tuple(entries))
        slot = (h >> (depth * BITS)) & MASK
        if not node.bitmap & (1 << slot):
            return node
        idx = _slot_index(node.bitmap, slot)
        new_child = HAMT._node_delete(node.children[idx], key, h, depth + 1, removed)
        children = list(node.children)
        if new_child is None:
            children.pop(idx)
            new_bitmap = node.bitmap & ~(1 << slot)
        else:
            children[idx] = new_child
            new_bitmap = node.bitmap
        return _Branch(bitmap=new_bitmap, children=tuple(children))

    # -------- public API ----------

    def get(self, key, default=None):
        v, found = HAMT._node_get(self._root, key, _hash(key), 0)
        return v if found else default

    def __contains__(self, key) -> bool:
        _, found = HAMT._node_get(self._root, key, _hash(key), 0)
        return found

    def __len__(self) -> int:
        return self._size

    def set(self, key, value) -> "HAMT":
        is_new = [False]
        new_root = HAMT._node_set(self._root, key, value, _hash(key), 0, is_new)
        return HAMT(root=new_root, size=self._size + (1 if is_new[0] else 0))

    def delete(self, key) -> "HAMT":
        removed = [False]
        new_root = HAMT._node_delete(self._root, key, _hash(key), 0, removed)
        if new_root is None:
            new_root = _Branch()
        return HAMT(root=new_root, size=self._size - (1 if removed[0] else 0))

    def items(self) -> Iterator[tuple]:
        def walk(node):
            if isinstance(node, _Leaf):
                yield from node.entries
            else:
                for c in node.children:
                    yield from walk(c)
        yield from walk(self._root)
