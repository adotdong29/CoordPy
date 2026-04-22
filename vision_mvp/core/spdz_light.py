"""SPDZ-light — additive MPC with Beaver triples.

Damgård, Pastro, Smart, Zakarias (CRYPTO 2012). SPDZ runs secure
multi-party computation over GF(p) via two primitives:

  - *Additive shares*: x = x_1 + x_2 + … + x_n (mod p), each party holds one.
    Addition is trivial: parties add their shares locally.
  - *Beaver triples*: precomputed (a, b, c) with c = a·b. Given triples,
    multiplication reduces to a reveal-open-multiply-and-adjust protocol.

To multiply shared x by shared y given a triple (a, b, c):
  d = x − a, e = y − b     (open — everyone learns d and e)
  xy = c + d·b + e·a + d·e  (compute locally, distribute additively)

Information-theoretic security against honest-but-curious adversaries in the
preprocessing model. Does not cover malicious adversaries (that requires
extra MACs; see full SPDZ).

Used as the concrete realisation of Idea 9 in `VISION_MILLIONS.md`: pre-
shared randomness (Beaver triples) → near-zero online communication for
common operations.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from .gf import DEFAULT_PRIME, add, mul, sub


def additive_shares(value: int, n: int, p: int = DEFAULT_PRIME,
                     seed: int | None = None) -> list[int]:
    """Split value into n uniformly-random additive shares over GF(p)."""
    if seed is not None:
        import random
        rng = random.Random(seed)
        shares = [rng.randrange(p) for _ in range(n - 1)]
    else:
        shares = [secrets.randbelow(p) for _ in range(n - 1)]
    last = value % p
    for s in shares:
        last = sub(last, s, p)
    return shares + [last]


def reconstruct_additive(shares: list[int], p: int = DEFAULT_PRIME) -> int:
    total = 0
    for s in shares:
        total = add(total, s, p)
    return total


@dataclass
class BeaverTriple:
    """Shares of (a, b, c) with c = a·b mod p held across parties."""
    a_shares: list[int]
    b_shares: list[int]
    c_shares: list[int]
    p: int

    def n_parties(self) -> int:
        return len(self.a_shares)


def generate_triple(n: int, p: int = DEFAULT_PRIME,
                    seed: int | None = None) -> BeaverTriple:
    """Honest-dealer triple generation (preprocessing phase)."""
    if seed is not None:
        import random
        rng = random.Random(seed)
        a = rng.randrange(p)
        b = rng.randrange(p)
    else:
        a = secrets.randbelow(p)
        b = secrets.randbelow(p)
    c = mul(a, b, p)
    return BeaverTriple(
        a_shares=additive_shares(a, n, p, seed=seed),
        b_shares=additive_shares(b, n, p, seed=None if seed is None else seed + 1),
        c_shares=additive_shares(c, n, p, seed=None if seed is None else seed + 2),
        p=p,
    )


def secure_multiply(
    x_shares: list[int], y_shares: list[int], triple: BeaverTriple,
) -> list[int]:
    """Compute shares of x·y given shares of x, y and a Beaver triple.

    Protocol:
      1. d = x − a;  e = y − b   (opened — everyone learns d and e)
      2. Result z = c + d·b + e·a + d·e
      Locally each party produces its own share of z; the first party also
      adds the constant d·e.
    """
    p = triple.p
    n = triple.n_parties()
    if len(x_shares) != n or len(y_shares) != n:
        raise ValueError("share count must match triple")

    # Open d and e
    d_shares = [sub(xi, ai, p) for xi, ai in zip(x_shares, triple.a_shares)]
    e_shares = [sub(yi, bi, p) for yi, bi in zip(y_shares, triple.b_shares)]
    d = reconstruct_additive(d_shares, p)
    e = reconstruct_additive(e_shares, p)

    # Each party computes z_i = c_i + d * b_i + e * a_i
    z = [add(add(triple.c_shares[i], mul(d, triple.b_shares[i], p), p),
             mul(e, triple.a_shares[i], p), p)
         for i in range(n)]
    # Add constant d·e to party 0 only
    z[0] = add(z[0], mul(d, e, p), p)
    return z
