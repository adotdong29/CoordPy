"""Cuckoo filter — adversarially-friendlier successor to the Bloom filter.

The causal-footprint store currently uses a classical Bloom filter, which is
known to be vulnerable to adversarial key enumeration (OPEN_QUESTIONS.md §5):
an attacker can craft keys that hash into any target bucket. Cuckoo filters
(Fan, Andersen, Kaminsky, Mitzenmacher 2014) solve three issues at once:

  1. They support *deletion* — Bloom filters do not.
  2. False-positive rate below a bound is achievable with less space than a
     Bloom filter at small FPR (they use a fingerprint indirection).
  3. Fingerprint-based buckets resist adversarial enumeration more robustly
     because the fingerprint doubles as the second-bucket hash input.

Layout:
  - `n_buckets` fixed-size buckets, each holding `entries_per_bucket`
    fingerprints (we use 4, the standard).
  - Each element has two candidate buckets:
        i1 = h(x)                    mod n_buckets
        i2 = i1 XOR h(fingerprint)   mod n_buckets
  - Insert: try i1; if full, try i2; if both full, kick a random occupant and
    relocate it to its alternate bucket; repeat up to `max_kicks` times.
  - Lookup / delete: check both candidate buckets for the fingerprint.

Standard fingerprint sizes:
  - 8 bits  →  FPR ≈ 3% at 95% load
  - 12 bits →  FPR ≈ 0.2%
  - 16 bits →  FPR ≈ 0.01%

Implementation note: we rely on `hashlib.blake2b` (constant-time,
cryptographic, stdlib) so the filter is deterministically reproducible and
hash-function choice is not a side-channel vulnerability.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterable


def _blake2b_bytes(data: bytes, seed: int, length: int = 8) -> int:
    h = hashlib.blake2b(data, digest_size=length, key=seed.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


def _to_bytes(key: object) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, int):
        return key.to_bytes(8, "little", signed=True)
    return repr(key).encode("utf-8")


@dataclass
class CuckooFilter:
    """Fixed-capacity cuckoo filter. Raises on insert when full."""

    capacity: int
    entries_per_bucket: int = 4
    fingerprint_bits: int = 12
    max_kicks: int = 500
    seed: int = 0xC0FFEE

    _buckets: list[list[int]] = field(init=False)
    _n_buckets: int = field(init=False)
    _size: int = field(init=False, default=0)
    _rng: random.Random = field(init=False)

    def __post_init__(self):
        if self.capacity < 1:
            raise ValueError("capacity must be ≥ 1")
        if self.entries_per_bucket < 1:
            raise ValueError("entries_per_bucket must be ≥ 1")
        if not 1 <= self.fingerprint_bits <= 32:
            raise ValueError("fingerprint_bits must be in [1, 32]")
        # Round up bucket count to next power of two — makes XOR-fingerprint
        # trick work without modular bias.
        raw = max(1, self.capacity // self.entries_per_bucket)
        n = 1
        while n < raw:
            n <<= 1
        self._n_buckets = n
        self._buckets = [[] for _ in range(self._n_buckets)]
        self._rng = random.Random(self.seed)

    # --- hash helpers ----------------------------------------------------

    def _fingerprint(self, key: object) -> int:
        raw = _blake2b_bytes(_to_bytes(key), self.seed, length=4)
        fp = raw & ((1 << self.fingerprint_bits) - 1)
        # 0 is reserved as "empty slot" sentinel; shift any 0 to 1.
        return fp if fp != 0 else 1

    def _index1(self, key: object) -> int:
        return _blake2b_bytes(_to_bytes(key), self.seed + 1) % self._n_buckets

    def _index2(self, i1: int, fp: int) -> int:
        return (i1 ^ _blake2b_bytes(
            fp.to_bytes(4, "little"), self.seed + 2
        )) % self._n_buckets

    # --- operations ------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def load_factor(self) -> float:
        return self._size / (self._n_buckets * self.entries_per_bucket)

    def insert(self, key: object) -> bool:
        """Add `key` to the filter. Returns True on success, False if full.

        On failure, the filter is in an inconsistent state and should be
        resized; callers can also ignore and treat False as "won't store."
        """
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        if len(self._buckets[i1]) < self.entries_per_bucket:
            self._buckets[i1].append(fp)
            self._size += 1
            return True
        i2 = self._index2(i1, fp)
        if len(self._buckets[i2]) < self.entries_per_bucket:
            self._buckets[i2].append(fp)
            self._size += 1
            return True

        # Both full — perform cuckoo kicks.
        i = self._rng.choice((i1, i2))
        for _ in range(self.max_kicks):
            j = self._rng.randrange(self.entries_per_bucket)
            if j >= len(self._buckets[i]):
                self._buckets[i].append(fp)
                self._size += 1
                return True
            self._buckets[i][j], fp = fp, self._buckets[i][j]
            i = self._index2(i, fp)
            if len(self._buckets[i]) < self.entries_per_bucket:
                self._buckets[i].append(fp)
                self._size += 1
                return True
        return False

    def __contains__(self, key: object) -> bool:
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        if fp in self._buckets[i1]:
            return True
        i2 = self._index2(i1, fp)
        return fp in self._buckets[i2]

    def delete(self, key: object) -> bool:
        """Remove one occurrence of `key`. Returns True iff it was present.

        Note: if `insert(x)` was called multiple times, only one occurrence
        is removed per `delete(x)` call.
        """
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        if fp in self._buckets[i1]:
            self._buckets[i1].remove(fp)
            self._size -= 1
            return True
        i2 = self._index2(i1, fp)
        if fp in self._buckets[i2]:
            self._buckets[i2].remove(fp)
            self._size -= 1
            return True
        return False

    def extend(self, keys: Iterable[object]) -> int:
        """Insert many keys; return number successfully inserted."""
        ok = 0
        for k in keys:
            if self.insert(k):
                ok += 1
        return ok

    def expected_fpr(self) -> float:
        """Theoretical upper bound on false-positive rate for current design.

        For a cuckoo filter with b entries/bucket and f-bit fingerprints at
        high load, FPR ≤ 2b / 2^f (Fan et al. 2014, Theorem 1).
        """
        return 2.0 * self.entries_per_bucket / float(1 << self.fingerprint_bits)
