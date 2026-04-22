"""Straggler-tolerant distributed computation via Reed-Solomon coding.

Lee, Lam, Pedarsani et al. (2018). To compute y = f(x_1, …, x_n) where each
x_i is handled by a separate worker, encode the n inputs into m ≥ n redundant
"coded" shards via an MDS (Reed-Solomon) code, send one shard to each worker,
and wait for *any* n to finish — the remaining m-n are stragglers and can
be dropped without loss.

For matrix multiplication f(A, b) = A b:
  - Split A row-wise into n blocks A_0, …, A_{n-1}
  - Create m coded blocks Ã_j = Σ_i g_{ji} A_i where g is an RS generator matrix
  - Worker j returns Ã_j b
  - Once any n workers return, decode via Lagrange interpolation

For 1-D data (our use: y_i = f(x_i, c) for constant c), same idea: the shards
themselves are evaluations of a degree-(n-1) polynomial at distinct points;
any n evaluations reconstruct the polynomial (and hence the shard values).

Uses `core/gf.py` for the finite-field arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gf import DEFAULT_PRIME, eval_poly, lagrange_interpolate


@dataclass
class CodedComputeReport:
    n_workers_needed: int
    m_workers: int
    n_stragglers_tolerated: int

    def summary(self) -> str:
        return (
            f"{self.n_workers_needed} of {self.m_workers} workers suffice "
            f"(tolerates {self.n_stragglers_tolerated} stragglers)"
        )


def encode_inputs(
    inputs: list[int], m: int, p: int = DEFAULT_PRIME,
) -> list[int]:
    """RS-encode n inputs as m ≥ n evaluations of the unique degree-(n-1)
    polynomial through (1, inputs[0]), …, (n, inputs[n-1]).

    Shard j (1-indexed) = poly(j). Shards 1..n reproduce the originals;
    shards n+1..m are redundancy that let the decoder tolerate up to (m − n)
    stragglers.
    """
    n = len(inputs)
    if m < n:
        raise ValueError("m must be ≥ n")
    xs = list(range(1, n + 1))
    ys = [v % p for v in inputs]
    return [lagrange_interpolate(xs, ys, j, p) for j in range(1, m + 1)]


def decode_outputs(
    received: list[tuple[int, int]],
    n: int,
    p: int = DEFAULT_PRIME,
) -> list[int]:
    """Given at least n (x, y) pairs, reconstruct the n input values."""
    if len(received) < n:
        raise ValueError(f"need at least {n} shards, got {len(received)}")
    # Take any n and interpolate at x=1..n to recover inputs
    picked = received[:n]
    xs = [x for x, _ in picked]
    ys = [y for _, y in picked]
    return [lagrange_interpolate(xs, ys, x_target, p) for x_target in range(1, n + 1)]


def coded_compute_report(n: int, m: int) -> CodedComputeReport:
    """Straggler tolerance summary."""
    return CodedComputeReport(
        n_workers_needed=n,
        m_workers=m,
        n_stragglers_tolerated=m - n,
    )
