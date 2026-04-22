"""Minimal prime-field GF(p) arithmetic — stdlib only.

Used by Shamir secret sharing (H1), coded distributed computation (C4), and
polar codes (C5). Python's built-in `pow(a, -1, p)` gives modular inverse in
Python 3.8+, so we don't need a dedicated finite-field library.

This is a *teaching* implementation — no side-channel hardening, no constant
time. Suitable for simulation/experiments; swap for the `galois` library or
a hardened backend for production.
"""

from __future__ import annotations

from dataclasses import dataclass


# A convenient prime: 2^31 - 1 (Mersenne) — fits in one 32-bit word.
DEFAULT_PRIME = (1 << 31) - 1


def inv_mod(a: int, p: int) -> int:
    return pow(a % p, -1, p)


def add(a: int, b: int, p: int) -> int:
    return (a + b) % p


def sub(a: int, b: int, p: int) -> int:
    return (a - b) % p


def mul(a: int, b: int, p: int) -> int:
    return (a * b) % p


def div(a: int, b: int, p: int) -> int:
    return (a * inv_mod(b, p)) % p


@dataclass
class GFVec:
    """Vector over GF(p). Stored as a list of ints, for readability."""

    values: list[int]
    p: int = DEFAULT_PRIME

    def __post_init__(self):
        self.values = [v % self.p for v in self.values]

    def __add__(self, other: "GFVec") -> "GFVec":
        if self.p != other.p or len(self.values) != len(other.values):
            raise ValueError("incompatible vectors")
        return GFVec([add(a, b, self.p) for a, b in zip(self.values, other.values)], self.p)

    def __matmul__(self, other: "GFVec") -> int:
        """Dot product over GF(p)."""
        if self.p != other.p or len(self.values) != len(other.values):
            raise ValueError("incompatible vectors")
        total = 0
        for a, b in zip(self.values, other.values):
            total = add(total, mul(a, b, self.p), self.p)
        return total

    def scaled(self, c: int) -> "GFVec":
        return GFVec([mul(v, c, self.p) for v in self.values], self.p)


def eval_poly(coeffs: list[int], x: int, p: int) -> int:
    """Horner evaluation of Σ coeffs[i] · x^i at x in GF(p)."""
    acc = 0
    for c in reversed(coeffs):
        acc = add(mul(acc, x, p), c, p)
    return acc


def lagrange_interpolate(xs: list[int], ys: list[int], at: int, p: int) -> int:
    """Evaluate the unique polynomial through (xs, ys) at x = `at`, over GF(p)."""
    if len(xs) != len(ys):
        raise ValueError("xs, ys must have same length")
    k = len(xs)
    total = 0
    for i in range(k):
        xi, yi = xs[i], ys[i]
        num = 1
        den = 1
        for j in range(k):
            if i == j:
                continue
            num = mul(num, sub(at, xs[j], p), p)
            den = mul(den, sub(xi, xs[j], p), p)
        total = add(total, mul(yi, div(num, den, p), p), p)
    return total
