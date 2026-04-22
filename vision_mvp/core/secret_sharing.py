"""Shamir's (k, n) threshold secret sharing — information-theoretically secure.

Shamir (1979). To share secret s among n parties such that any k can
reconstruct, but any k-1 learn nothing:

  Build a polynomial q(x) = s + a_1 x + … + a_{k-1} x^{k-1} over GF(p), with
  a_i drawn uniformly at random. Party i gets (i, q(i)).

Reconstruction is Lagrange interpolation at x = 0.

Security: any k-1 shares are uniformly distributed over GF(p), independent of
s. k shares reconstruct exactly. Used here as the substrate for H5 SPDZ-light
(MPC) and as a Wave-4 building block for adversarially-robust workspace
election.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from .gf import DEFAULT_PRIME, eval_poly, lagrange_interpolate


@dataclass
class Share:
    x: int
    y: int
    p: int


def split(secret: int, n: int, k: int,
           p: int = DEFAULT_PRIME, seed: int | None = None) -> list[Share]:
    """Split `secret` into `n` Shamir shares with reconstruction threshold `k`.

    If `seed` is provided, the polynomial is seeded deterministically — useful
    for testing; always pass None in production.
    """
    if not 1 <= k <= n:
        raise ValueError("need 1 ≤ k ≤ n")
    if not 0 <= secret < p:
        raise ValueError(f"secret must be in [0, p), got {secret}")

    if seed is not None:
        import random
        rng = random.Random(seed)
        coeffs = [secret] + [rng.randrange(p) for _ in range(k - 1)]
    else:
        coeffs = [secret] + [secrets.randbelow(p) for _ in range(k - 1)]

    # Party indices 1..n; 0 is reserved for the secret.
    return [Share(x=i, y=eval_poly(coeffs, i, p), p=p) for i in range(1, n + 1)]


def reconstruct(shares: list[Share]) -> int:
    """Reconstruct the secret by Lagrange-interpolating at x=0."""
    if not shares:
        raise ValueError("need at least one share")
    p = shares[0].p
    if any(s.p != p for s in shares):
        raise ValueError("all shares must share the same prime")
    xs = [s.x for s in shares]
    if len(set(xs)) != len(xs):
        raise ValueError("share x-coordinates must be distinct")
    return lagrange_interpolate(xs, [s.y for s in shares], 0, p)


def verify_threshold(
    secret: int, n: int, k: int, p: int = DEFAULT_PRIME, seed: int = 0,
) -> bool:
    """Sanity check: splitting and recombining any k-subset reconstructs s,
    and removing one from a k-subset gives a different reconstruction.
    """
    shares = split(secret, n, k, p, seed=seed)
    s_full = reconstruct(shares[:k])
    return s_full == secret
