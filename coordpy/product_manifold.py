"""W43 Product-Manifold Capsule (PMC) — capsule-native product
manifold + factoradic routing + subspace state + causal lattice.

W43 is the first capsule-native layer in CoordPy that decomposes
each cell's coordination state across a *product manifold* with
six interlocking channels:

  * a hyperbolic branch channel  — bounded Poincare-disk-style
    encoding of parent-DAG depth and branch path
  * a spherical consensus channel — unit-norm signature over the
    cell's claim_kinds, giving cosine-distance consensus
  * a euclidean attribute channel — linear vector of cell attributes
    (round, n_handoffs, total_payload_words)
  * a factoradic route channel    — Lehmer-coded permutation of
    role arrival order; O(log(n!)) structured bits at zero
    visible-token cost
  * a subspace state channel       — bounded-rank orthonormal basis
    representing the cell's admissible-interpretation subspace,
    canonicalised by QR (a strict approximation of one point on
    the Grassmannian Gr(k, d))
  * a causal-clock channel         — Lamport vector clock over the
    role-handoff arrival order, with explicit dependency-closure
    admissibility check

W43 is strictly additive on top of W42: when configured trivially
(``pmc_enabled=False``, ``manifest_v13_disabled=True``,
``abstain_on_causal_violation=False``, ``abstain_on_subspace_drift
=False``), the orchestrator reduces to the W42 result byte-for-
byte. The W43 layer never reads transformer hidden states, KV
cache, or attention weights; it operates purely on the capsule
layer.

W43 introduces seven new content-addressed component CIDs:
``hyperbolic_channel_cid``, ``spherical_channel_cid``,
``euclidean_channel_cid``, ``factoradic_route_cid``,
``subspace_basis_cid``, ``causal_clock_cid``, and
``manifold_audit_cid``. They bind under one ``manifold_witness_cid``
and one ``manifest_v13_cid``. Each is verified against eighteen
enumerated W43 failure modes disjoint from the W22..W42 cumulative
boundary.

Honest scope (do-not-overstate)
-------------------------------

W43 does NOT implement transformer-internal mixed-curvature
attention. It does NOT pool KV cache across hosts. It does NOT
provide a continuous Grassmannian homotopy. Each channel is a
mathematically motivated *bounded approximation* of one direction
in the "product-manifold / cram-singularity" architecture vision.
The gap from approximation to full substrate is documented in the
W43 conjecture set:

  * ``W43-C-MIXED-CURVATURE-LATENT``    — full transformer-internal
    mixed-curvature attention requires substrate access.
  * ``W43-C-COLLECTIVE-KV-POOLING``     — host-collective KV-cache
    sharing requires transformer-internal hooks not exposed at
    the capsule layer.
  * ``W43-C-FULL-GRASSMANNIAN-HOMOTOPY`` — a true Gr(k, d) homotopy
    requires continuous representation that the bounded basis
    proxy approximates only locally.

W43 introduces three NEW proved-conditional limitation theorems
at the capsule layer:

  * ``W43-L-DUAL-CHANNEL-COLLUSION-CAP`` — when an adversary
    controls BOTH the spherical-channel signature AND the
    subspace-basis registry, W43 cannot recover.
  * ``W43-L-FORGED-CAUSAL-CLOCK-CAP``    — when an adversary
    can forge a monotone Lamport vector clock that satisfies
    closure but does not match the registered topology, W43
    rejects via the causal-clock-cid mismatch but cannot infer
    the true ordering from the capsule layer alone.
  * ``W43-L-OBLIVIOUS-FACTORADIC-CAP``   — when role names are
    permuted in the registered order, the factoradic channel
    routes via the *registered* permutation, not the
    *intended* one; cross-host topology disambiguation is out
    of scope for the W43 channel.

This module lives at ``coordpy.product_manifold`` and is NOT
exported through ``coordpy.__experimental__`` at this milestone:
the stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W43 surface through an explicit
``from coordpy.product_manifold import ...`` import.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


# =============================================================================
# Schema, branch constants, defaults
# =============================================================================

W43_PRODUCT_MANIFOLD_SCHEMA_VERSION: str = (
    "coordpy.product_manifold.v1")

# Decision branches mirror the W22..W42 family.
W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH: str = (
    "trivial_pmc_passthrough")
W43_BRANCH_PMC_DISABLED: str = "pmc_disabled"
W43_BRANCH_PMC_REJECTED: str = "pmc_rejected"
W43_BRANCH_PMC_NO_TRIGGER: str = "pmc_no_trigger"
W43_BRANCH_PMC_RATIFIED: str = "pmc_ratified"
W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED: str = (
    "pmc_causal_violation_abstained")
W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED: str = (
    "pmc_subspace_drift_abstained")
W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED: str = (
    "pmc_spherical_divergence_abstained")
W43_BRANCH_PMC_NO_POLICY: str = "pmc_no_policy"

W43_ALL_BRANCHES: tuple[str, ...] = (
    W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH,
    W43_BRANCH_PMC_DISABLED,
    W43_BRANCH_PMC_REJECTED,
    W43_BRANCH_PMC_NO_TRIGGER,
    W43_BRANCH_PMC_RATIFIED,
    W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED,
    W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED,
    W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED,
    W43_BRANCH_PMC_NO_POLICY,
)

# Per-channel defaults.
W43_DEFAULT_HYPERBOLIC_DIM: int = 4
"""Dimension of the bounded Poincare-disk encoding."""

W43_DEFAULT_SPHERICAL_DIM: int = 8
"""Dimension of the unit-norm spherical consensus signature."""

W43_DEFAULT_EUCLIDEAN_DIM: int = 4
"""Dimension of the linear euclidean attribute vector."""

W43_DEFAULT_SUBSPACE_RANK: int = 2
"""Rank k of the subspace basis (k columns of the d-vector basis)."""

W43_DEFAULT_SUBSPACE_DIM: int = 4
"""Ambient dimension d of the subspace basis."""

W43_DEFAULT_SPHERICAL_AGREEMENT_MIN: float = 0.85
"""Minimum cosine agreement between observed and expected
spherical signatures for ratification on the consensus path."""

W43_DEFAULT_SUBSPACE_DRIFT_MAX: float = 0.25
"""Maximum allowed principal-angle drift between observed and
registered subspace bases (in radians; abstain above this)."""

W43_DEFAULT_HYPERBOLIC_RADIUS_MAX: float = 0.95
"""Maximum allowed Poincare-disk radius. Encodings outside this
ball are rejected as numerically unstable (the disk model loses
precision near the boundary)."""


# =============================================================================
# Canonicalisation helpers
# =============================================================================

def _canonical_bytes(payload: Any) -> bytes:
    """Canonical JSON-bytes for CID input (deterministic)."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    """Round floats to a fixed precision so CIDs are deterministic
    across platforms with slightly different float arithmetic."""
    return [float(round(float(v), precision)) for v in values]


def _round_matrix(
        matrix: Sequence[Sequence[float]], precision: int = 12,
) -> list[list[float]]:
    return [_round_floats(row, precision) for row in matrix]


# =============================================================================
# Hyperbolic branch channel (Poincare-disk-style)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class HyperbolicBranchEncoding:
    """Bounded Poincare-disk-style encoding of a parent-DAG path.

    The branch path is encoded as a small ``dim``-dimensional vector
    inside the open unit ball of radius ``r_max < 1``. The radial
    component encodes depth as ``r = tanh(depth / scale) * r_max``;
    the angular components encode the binary parent/child branch
    choices through a deterministic projection on the unit sphere.

    This is NOT a full hyperbolic neural network — it is a
    closed-form, deterministic, bijective encoding for shallow
    branch trees (depth <= 2 * dim) that round-trips back to the
    original (depth, path) pair. For trees deeper than the encoding
    capacity, the path is hashed deterministically to a residual
    angle; the round-trip is then approximate (depth is preserved;
    path hash is preserved as a SHA-256 anchor).
    """

    coordinates: tuple[float, ...]
    depth: int
    path_hash: str
    r_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": _round_floats(self.coordinates),
            "depth": int(self.depth),
            "path_hash": str(self.path_hash),
            "r_max": float(round(self.r_max, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w43_hyperbolic_branch",
            "encoding": self.to_dict(),
        })


def encode_hyperbolic_branch(
        path: Sequence[int],
        *,
        dim: int = W43_DEFAULT_HYPERBOLIC_DIM,
        r_max: float = W43_DEFAULT_HYPERBOLIC_RADIUS_MAX,
        scale: float = 4.0,
) -> HyperbolicBranchEncoding:
    """Encode a binary branch ``path`` (sequence of 0/1 choices) as
    a bounded vector inside the unit ball.

    Parameters
    ----------
    path
        Sequence of 0/1 integers giving the parent/child choices
        from the root of the branch DAG to the current cell.
        Length is the cell's depth.
    dim
        Encoding dimension. The first ``2*dim`` choices fit
        bijectively; deeper paths fold into a SHA-256 anchor.
    r_max
        Strict upper bound on the radius of the encoded vector.
        Must be in (0, 1).

    Returns
    -------
    HyperbolicBranchEncoding
        Frozen encoding with a deterministic CID.
    """
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    if not (0.0 < r_max < 1.0):
        raise ValueError(
            f"r_max must be in (0, 1), got {r_max}")
    depth = int(len(path))
    # Validate path entries are 0/1.
    for i, c in enumerate(path):
        if c not in (0, 1):
            raise ValueError(
                f"path[{i}] must be 0 or 1, got {c!r}")

    # Per-coordinate scale alpha. The all-coords-at-1.0 worst case
    # has norm alpha * sqrt(dim); we choose alpha so that bound is
    # strictly below r_max, leaving room for the radial-depth
    # multiplier introduced below.
    base_alpha = float(r_max) / math.sqrt(max(1, dim)) * 0.9

    # Radial-depth modulator in [0.5, 1.0]: shallow paths shrink the
    # coordinates toward the origin (more room near the centre),
    # deep paths sit farther out. The modulator is monotone in depth
    # and bounded so the ball constraint always holds.
    radial = 0.5 + 0.5 * math.tanh(float(depth) / float(scale))
    alpha = base_alpha * radial

    coords: list[float] = [0.0] * dim
    capacity = min(depth, 2 * dim)
    for i in range(capacity):
        coord_idx = i // 2
        bit_in_coord = i % 2
        if bit_in_coord == 0:
            # First bit of the pair sets the sign with placeholder
            # magnitude 0.5 (refined to 0.5 or 1.0 by the next bit).
            sign = 1.0 if path[i] == 1 else -1.0
            coords[coord_idx] = sign * 0.5
        else:
            mag = 1.0 if path[i] == 1 else 0.5
            sign = 1.0 if coords[coord_idx] >= 0 else -1.0
            coords[coord_idx] = sign * mag

    # Scale every populated coord by alpha so the round-trip can
    # recover the magnitude bit by dividing back by alpha.
    coords = [c * alpha for c in coords]

    path_hash = hashlib.sha256(
        bytes(int(b) & 1 for b in path)
    ).hexdigest()

    enc = HyperbolicBranchEncoding(
        coordinates=tuple(_round_floats(coords)),
        depth=depth,
        path_hash=path_hash,
        r_max=float(r_max),
    )
    # Final norm check: enforce r < r_max strictly.
    actual_r = math.sqrt(sum(c * c for c in enc.coordinates))
    if actual_r >= r_max:
        raise ValueError(
            f"hyperbolic encoding norm {actual_r:.12f} >= r_max "
            f"{r_max:.12f}; numerical instability")
    return enc


def decode_hyperbolic_branch_path_prefix(
        enc: HyperbolicBranchEncoding,
        *, scale: float = 4.0,
) -> tuple[int, ...]:
    """Recover the first ``min(enc.depth, 2 * dim)`` bits of the
    encoded path. For paths longer than the encoding capacity, the
    remainder must be re-derived from ``enc.path_hash`` (the hash
    is the witness, not the original bits).

    Round-trip guarantee: for paths of depth <= 2 * dim,
    ``encode_hyperbolic_branch(p)`` then
    ``decode_hyperbolic_branch_path_prefix(.)`` returns the original
    path exactly.
    """
    coords = enc.coordinates
    dim = len(coords)
    out: list[int] = []
    capacity = min(enc.depth, 2 * dim)
    if capacity == 0 or dim == 0:
        return tuple()
    base_alpha = float(enc.r_max) / math.sqrt(max(1, dim)) * 0.9
    radial = 0.5 + 0.5 * math.tanh(float(enc.depth) / float(scale))
    alpha = base_alpha * radial
    if alpha == 0:
        return tuple([0] * capacity)
    for i in range(capacity):
        coord_idx = i // 2
        bit_in_coord = i % 2
        v = coords[coord_idx]
        if bit_in_coord == 0:
            out.append(1 if v >= 0 else 0)
        else:
            # Magnitude bit: encoded as 0.5 or 1.0 in pre-scale
            # units. After dividing by alpha we recover that.
            mag = abs(v) / alpha
            out.append(1 if mag > 0.75 else 0)
    return tuple(out)


# =============================================================================
# Spherical consensus channel
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SphericalConsensusSignature:
    """Unit-norm signature over a cell's claim_kinds.

    Each cell produces a ``dim``-vector whose i-th coordinate is the
    *frequency* of claim_kind ``i`` (in a registered ordering),
    L2-normalised to unit length. Two cells' signatures agree if
    their cosine similarity exceeds an agreement threshold; this is
    the spherical consensus distance, which lives on S^{dim-1}.
    """

    coordinates: tuple[float, ...]
    n_observations: int
    claim_kinds_sorted: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": _round_floats(self.coordinates),
            "n_observations": int(self.n_observations),
            "claim_kinds_sorted": list(self.claim_kinds_sorted),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w43_spherical_consensus",
            "signature": self.to_dict(),
        })


def encode_spherical_consensus(
        claim_kinds: Sequence[str],
        *,
        dim: int = W43_DEFAULT_SPHERICAL_DIM,
) -> SphericalConsensusSignature:
    """Encode a list of claim_kind labels as a unit-norm signature.

    Each label hashes deterministically into one of ``dim``
    buckets; the signature is the L2-normalised bucket-frequency
    vector. If ``claim_kinds`` is empty, the signature is a
    zero-vector marked by ``n_observations=0``.
    """
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    counts = [0] * dim
    for k in claim_kinds:
        h = int(hashlib.sha256(str(k).encode("utf-8")).hexdigest(),
                16)
        counts[h % dim] += 1
    n = sum(counts)
    if n == 0:
        return SphericalConsensusSignature(
            coordinates=tuple([0.0] * dim),
            n_observations=0,
            claim_kinds_sorted=tuple(),
        )
    norm = math.sqrt(sum(c * c for c in counts))
    coords = tuple(_round_floats([c / norm for c in counts]))
    return SphericalConsensusSignature(
        coordinates=coords,
        n_observations=int(n),
        claim_kinds_sorted=tuple(sorted(set(str(k) for k in claim_kinds))),
    )


def cosine_agreement(
        a: SphericalConsensusSignature,
        b: SphericalConsensusSignature,
) -> float:
    """Return cosine similarity in [-1, 1] (or 0 if either is the
    zero-vector)."""
    if a.n_observations == 0 or b.n_observations == 0:
        return 0.0
    if len(a.coordinates) != len(b.coordinates):
        return 0.0
    dot = sum(x * y for x, y in zip(a.coordinates, b.coordinates))
    na = math.sqrt(sum(x * x for x in a.coordinates))
    nb = math.sqrt(sum(y * y for y in b.coordinates))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


# =============================================================================
# Euclidean attribute channel
# =============================================================================

@dataclasses.dataclass(frozen=True)
class EuclideanAttributeVector:
    """Linear vector of cell attributes."""

    coordinates: tuple[float, ...]
    field_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": _round_floats(self.coordinates),
            "field_names": list(self.field_names),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w43_euclidean_attribute",
            "vector": self.to_dict(),
        })


def encode_euclidean_attributes(
        attributes: Mapping[str, float],
        *,
        field_order: Sequence[str] | None = None,
        dim: int = W43_DEFAULT_EUCLIDEAN_DIM,
) -> EuclideanAttributeVector:
    """Encode a mapping of named numeric attributes into an
    ``dim``-vector. Fields beyond ``dim`` are dropped; missing
    fields are zero-padded.
    """
    names = list(field_order) if field_order is not None else sorted(
        attributes.keys())
    names = names[:dim]
    coords: list[float] = []
    for n in names:
        v = attributes.get(n, 0.0)
        try:
            coords.append(float(v))
        except (TypeError, ValueError):
            coords.append(0.0)
    while len(coords) < dim:
        coords.append(0.0)
        names.append("")
    return EuclideanAttributeVector(
        coordinates=tuple(_round_floats(coords)),
        field_names=tuple(names),
    )


# =============================================================================
# Factoradic route channel (Lehmer code / permutation)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class FactoradicRoute:
    """A bijective Lehmer-code encoding of a permutation.

    The permutation lives on ``[0, n)``. The Lehmer code is the
    sequence ``(L[0], L[1], ..., L[n-1])`` where ``L[i]`` is the
    number of elements after ``perm[i]`` in ``perm[i+1:]`` that are
    smaller than ``perm[i]``. The factoradic integer is the
    factorial-base interpretation of the Lehmer code; together they
    are bijective with the symmetric group ``S_n``.

    ``factoradic_int`` carries log2(n!) bits of structured state at
    zero visible-token cost (it lives inside the manifest CID).
    """

    permutation: tuple[int, ...]
    lehmer_code: tuple[int, ...]
    factoradic_int: int
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "permutation": list(self.permutation),
            "lehmer_code": list(self.lehmer_code),
            "factoradic_int": int(self.factoradic_int),
            "n": int(self.n),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w43_factoradic_route",
            "route": self.to_dict(),
        })

    def n_structured_bits(self) -> int:
        """Information capacity in bits. ceil(log2(n!))."""
        if self.n < 2:
            return 0
        return int(math.ceil(math.log2(math.factorial(self.n))))


def encode_factoradic_route(
        permutation: Sequence[int],
) -> FactoradicRoute:
    """Encode a permutation of [0, n) as a Lehmer code +
    factoradic integer.

    Bijective for all permutations of [0, n)."""
    n = len(permutation)
    if n == 0:
        return FactoradicRoute(
            permutation=tuple(), lehmer_code=tuple(),
            factoradic_int=0, n=0,
        )
    perm = list(permutation)
    if sorted(perm) != list(range(n)):
        raise ValueError(
            f"permutation must be a permutation of range({n}); "
            f"got {perm}")
    # Lehmer code in O(n^2). The Lehmer code at position i is the
    # rank of perm[i] in the still-available natural numbers
    # ``[0, n) \ {perm[0..i-1]}``. The factoradic integer that
    # results is bijective with the symmetric group S_n.
    lehmer: list[int] = []
    remaining = list(range(n))
    for v in perm:
        idx = remaining.index(v)
        lehmer.append(idx)
        remaining.pop(idx)
    # Factoradic integer.
    fact = 0
    for i, l in enumerate(lehmer):
        fact += l * math.factorial(n - 1 - i)
    return FactoradicRoute(
        permutation=tuple(perm),
        lehmer_code=tuple(lehmer),
        factoradic_int=int(fact),
        n=int(n),
    )


def decode_factoradic_route(
        factoradic_int: int, *, n: int,
) -> FactoradicRoute:
    """Recover the Lehmer code and permutation from the factoradic
    integer. Inverse of :func:`encode_factoradic_route`."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return FactoradicRoute(
            permutation=tuple(), lehmer_code=tuple(),
            factoradic_int=0, n=0,
        )
    max_int = math.factorial(n) - 1
    if factoradic_int < 0 or factoradic_int > max_int:
        raise ValueError(
            f"factoradic_int {factoradic_int} out of range "
            f"[0, {max_int}] for n={n}")
    # Recover Lehmer code.
    lehmer: list[int] = []
    rem = int(factoradic_int)
    for i in range(n):
        f = math.factorial(n - 1 - i)
        l = rem // f
        rem = rem % f
        lehmer.append(int(l))
    # Reconstruct permutation from Lehmer code.
    pool = list(range(n))
    perm: list[int] = []
    for l in lehmer:
        perm.append(pool.pop(l))
    return FactoradicRoute(
        permutation=tuple(perm),
        lehmer_code=tuple(lehmer),
        factoradic_int=int(factoradic_int),
        n=int(n),
    )


# =============================================================================
# Subspace state channel (Grassmannian-style approximation)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SubspaceBasis:
    """A bounded-rank orthonormal basis representing one point on
    the Grassmannian Gr(k, d).

    Stored as a ``d x k`` matrix whose ``k`` columns are an
    orthonormal basis for the span. Canonicalised by an explicit
    QR step so two bases of the same subspace produce the same
    serialised matrix and therefore the same CID.

    This is a **bounded approximation** — comparison is via
    principal angles between subspaces (max sin(theta) over the
    canonical basis pair), not a full Grassmannian homotopy.
    """

    basis_columns: tuple[tuple[float, ...], ...]
    rank: int
    dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis_columns": _round_matrix(self.basis_columns),
            "rank": int(self.rank),
            "dim": int(self.dim),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w43_subspace_basis",
            "basis": self.to_dict(),
        })


def _matrix_transpose(m: Sequence[Sequence[float]]) -> list[list[float]]:
    if not m:
        return []
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def _matmul(
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not a or not b:
        return []
    n_rows = len(a)
    n_inner = len(a[0])
    n_cols = len(b[0])
    if len(b) != n_inner:
        raise ValueError(
            f"matmul shape mismatch: {n_rows}x{n_inner} @ "
            f"{len(b)}x{n_cols}")
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            s = 0.0
            for k in range(n_inner):
                s += a[i][k] * b[k][j]
            out[i][j] = s
    return out


def _gram_schmidt_qr(
        vectors: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Return an orthonormal basis (column vectors) for the column
    span of ``vectors`` (which is dim x k, k columns of length dim).

    Modified Gram-Schmidt with reorthogonalisation. Drops zero
    columns (returns an effective rank smaller than k if the span
    is degenerate).
    """
    if not vectors:
        return []
    dim = len(vectors)
    k = len(vectors[0]) if dim > 0 else 0
    # Convert to column-major for easier handling.
    columns: list[list[float]] = []
    for j in range(k):
        col = [float(vectors[i][j]) for i in range(dim)]
        # Modified Gram-Schmidt against existing columns.
        for q in columns:
            dot = sum(col[i] * q[i] for i in range(dim))
            for i in range(dim):
                col[i] -= dot * q[i]
        # Reorthogonalise (one extra pass).
        for q in columns:
            dot = sum(col[i] * q[i] for i in range(dim))
            for i in range(dim):
                col[i] -= dot * q[i]
        norm = math.sqrt(sum(c * c for c in col))
        if norm < 1e-12:
            # Degenerate column; skip.
            continue
        col = [c / norm for c in col]
        # Canonicalise sign: enforce first non-zero entry positive.
        for c in col:
            if abs(c) > 1e-12:
                if c < 0:
                    col = [-c for c in col]
                break
        columns.append(col)
    # Reshape back to row-major dim x k_eff.
    k_eff = len(columns)
    out = [[columns[j][i] for j in range(k_eff)] for i in range(dim)]
    return out


def encode_subspace_basis(
        vectors: Sequence[Sequence[float]],
        *,
        rank: int = W43_DEFAULT_SUBSPACE_RANK,
        dim: int = W43_DEFAULT_SUBSPACE_DIM,
) -> SubspaceBasis:
    """Build a canonicalised orthonormal basis for the span of
    ``vectors`` (a ``dim x k`` matrix).

    The basis is canonicalised by Gram-Schmidt with sign
    normalisation so two equal subspaces produce equal CIDs.
    If the span has effective rank below ``rank``, the missing
    columns are zero-padded.
    """
    if rank <= 0 or dim <= 0:
        raise ValueError(
            f"rank, dim must be > 0; got rank={rank}, dim={dim}")
    if not vectors:
        empty_cols = [[0.0] * rank for _ in range(dim)]
        return SubspaceBasis(
            basis_columns=tuple(tuple(_round_floats(row))
                                 for row in empty_cols),
            rank=int(rank), dim=int(dim),
        )
    # Pad / truncate to dim x k.
    cols_in = []
    for j in range(min(rank, len(vectors[0]) if vectors else 0)):
        col = []
        for i in range(dim):
            if i < len(vectors) and j < len(vectors[i]):
                col.append(float(vectors[i][j]))
            else:
                col.append(0.0)
        cols_in.append(col)
    # Reshape to row-major dim x k for QR.
    if cols_in:
        k_in = len(cols_in)
        rowmaj = [[cols_in[j][i] for j in range(k_in)]
                  for i in range(dim)]
    else:
        rowmaj = [[0.0] * rank for _ in range(dim)]
    qr = _gram_schmidt_qr(rowmaj)
    # Pad columns to ``rank``.
    if not qr:
        qr = [[0.0] * rank for _ in range(dim)]
    while len(qr[0]) < rank:
        for i in range(dim):
            qr[i].append(0.0)
    # Truncate columns if more than rank.
    qr = [row[:rank] for row in qr]
    return SubspaceBasis(
        basis_columns=tuple(tuple(_round_floats(row)) for row in qr),
        rank=int(rank), dim=int(dim),
    )


def principal_angle_drift(
        a: SubspaceBasis,
        b: SubspaceBasis,
) -> float:
    """Compute the maximum principal angle (in radians) between two
    subspaces. Returns 0 for identical subspaces, pi/2 for fully
    orthogonal subspaces.

    Implementation: max sin(theta_i) where theta_i are the
    principal angles, computed via the SVD of B^T A. Since we lack
    a numpy dependency at this layer, we use a small power-iteration
    proxy on the symmetric matrix A^T B B^T A.
    """
    if a.rank != b.rank or a.dim != b.dim:
        return float(math.pi / 2)
    A = list(a.basis_columns)
    B = list(b.basis_columns)
    # M = A^T B (k x k).
    AT = _matrix_transpose(A)
    M = _matmul(AT, B)
    # Compute singular values of M via M^T M -> symmetric, find
    # maximum eigenvalue / minimum eigenvalue via power iteration.
    MTM = _matmul(_matrix_transpose(M), M)
    k = len(MTM)
    if k == 0:
        return 0.0
    # Trace of MTM = sum of singular values squared. The largest
    # principal angle has cos^2 = min_sigma^2. We compute min by
    # subtracting from identity scaled.
    # Quick deterministic eigvals via Jacobi rotations.
    eigvals = _jacobi_eigvals(MTM)
    if not eigvals:
        return float(math.pi / 2)
    # Each cos^2(theta_i) = sigma_i^2. The maximum principal angle
    # has the smallest cos^2.
    cos2_min = max(0.0, min(1.0, min(eigvals)))
    # sin^2 = 1 - cos^2.
    sin2 = max(0.0, min(1.0, 1.0 - cos2_min))
    return float(math.asin(math.sqrt(sin2)))


def _jacobi_eigvals(
        m: Sequence[Sequence[float]], *,
        max_sweeps: int = 64,
        tol: float = 1e-12,
) -> list[float]:
    """Compute eigenvalues of a small symmetric matrix via
    Jacobi rotations. Deterministic, dependency-free."""
    n = len(m)
    if n == 0:
        return []
    a = [[float(m[i][j]) for j in range(n)] for i in range(n)]
    for _sweep in range(max_sweeps):
        off = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    off += a[i][j] * a[i][j]
        if off < tol:
            break
        # Find largest off-diagonal.
        p, q, max_val = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > max_val:
                    max_val = abs(a[i][j])
                    p, q = i, j
        if max_val < tol:
            break
        # Compute rotation.
        if abs(a[p][p] - a[q][q]) < tol:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(
                2.0 * a[p][q], a[p][p] - a[q][q])
        c = math.cos(theta)
        s = math.sin(theta)
        # Apply rotation to rows p and q.
        for i in range(n):
            api = a[i][p]
            aqi = a[i][q]
            a[i][p] = c * api + s * aqi
            a[i][q] = -s * api + c * aqi
        for j in range(n):
            apj = a[p][j]
            aqj = a[q][j]
            a[p][j] = c * apj + s * aqj
            a[q][j] = -s * apj + c * aqj
    return [a[i][i] for i in range(n)]


# =============================================================================
# Causal-clock channel (Lamport vector clock + dependency closure)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CausalVectorClock:
    """A Lamport-style vector clock indexed by role name.

    Each role's component records the count of handoffs the role
    has authored. A handoff from role ``r`` carries the *snapshot*
    of the clock just after ``r`` increments its component.

    Admissibility (Lamport partial order): for clocks ``c_a`` and
    ``c_b``, ``c_a <= c_b`` iff every component of ``c_a`` is
    bounded by the corresponding component of ``c_b``. A handoff
    sequence is causally admissible if its clocks form a total
    order under <= on the role-permuted components.
    """

    components: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": [list(c) for c in self.components],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w43_causal_vector_clock",
            "clock": self.to_dict(),
        })

    @property
    def as_dict(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.components}

    @classmethod
    def from_mapping(
            cls, m: Mapping[str, int],
    ) -> "CausalVectorClock":
        return cls(components=tuple(
            sorted((str(k), int(v)) for k, v in m.items())))


def causally_dominates(
        a: CausalVectorClock, b: CausalVectorClock,
) -> bool:
    """Return True if ``a <= b`` componentwise on the union of role
    names. Missing components default to 0."""
    da = a.as_dict
    db = b.as_dict
    keys = set(da.keys()) | set(db.keys())
    return all(da.get(k, 0) <= db.get(k, 0) for k in keys)


def is_causally_admissible(
        clocks: Sequence[CausalVectorClock],
) -> bool:
    """Return True if the clocks form a total order under causal
    domination. Handoff sequences are admissible iff this holds."""
    for i in range(len(clocks) - 1):
        if not causally_dominates(clocks[i], clocks[i + 1]):
            return False
    return True


def detect_causal_violation_index(
        clocks: Sequence[CausalVectorClock],
) -> int:
    """Return the index ``i`` of the first violation
    (``clocks[i] <= clocks[i+1]`` fails) or ``-1`` if admissible."""
    for i in range(len(clocks) - 1):
        if not causally_dominates(clocks[i], clocks[i + 1]):
            return int(i)
    return -1


# =============================================================================
# Cell observations -> channel bundle
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ProductManifoldChannelBundle:
    """The five channels' encodings for one cell, plus the causal
    clocks bundle. Frozen and content-addressable."""

    hyperbolic: HyperbolicBranchEncoding
    spherical: SphericalConsensusSignature
    euclidean: EuclideanAttributeVector
    factoradic: FactoradicRoute
    subspace: SubspaceBasis
    causal_clocks: tuple[CausalVectorClock, ...]
    causal_admissible: bool
    causal_violation_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "hyperbolic": self.hyperbolic.to_dict(),
            "spherical": self.spherical.to_dict(),
            "euclidean": self.euclidean.to_dict(),
            "factoradic": self.factoradic.to_dict(),
            "subspace": self.subspace.to_dict(),
            "causal_clocks": [c.to_dict()
                              for c in self.causal_clocks],
            "causal_admissible": bool(self.causal_admissible),
            "causal_violation_index": int(self.causal_violation_index),
        }


@dataclasses.dataclass(frozen=True)
class CellObservation:
    """Minimal observation contract for the W43 layer.

    A caller (typically a controller wrapping the W42 orchestrator)
    populates one of these per cell. The object is intentionally
    simple — strings + ints + small float vectors — so it can be
    constructed from any agent-team orchestrator.

    Fields:

      branch_path
        Sequence of 0/1 integers giving the parent/child choices
        from the root to the cell.
      claim_kinds
        List of claim_kind labels observed in the cell.
      attributes
        Mapping of named numeric attributes (round, n_handoffs,
        total_payload_words, wall_ms, etc.).
      role_arrival_order
        The order in which roles authored handoffs in this cell, as
        a list of role names. Encoded factoradically against
        ``role_universe``.
      role_universe
        The canonical, sorted list of all role names participating
        in the run. Defines the symmetric group for the factoradic
        channel.
      subspace_vectors
        A small ``dim x k`` matrix whose columns span the cell's
        admissible-interpretation subspace. Typically derived from
        the cell's role-handoff payload features.
      causal_clocks
        Per-handoff vector clocks. The sequence must be presented
        in the order the handoffs were observed.
    """

    branch_path: tuple[int, ...] = ()
    claim_kinds: tuple[str, ...] = ()
    attributes: tuple[tuple[str, float], ...] = ()
    role_arrival_order: tuple[str, ...] = ()
    role_universe: tuple[str, ...] = ()
    subspace_vectors: tuple[tuple[float, ...], ...] = ()
    causal_clocks: tuple[CausalVectorClock, ...] = ()


def encode_cell_channels(
        obs: CellObservation,
        *,
        hyperbolic_dim: int = W43_DEFAULT_HYPERBOLIC_DIM,
        spherical_dim: int = W43_DEFAULT_SPHERICAL_DIM,
        euclidean_dim: int = W43_DEFAULT_EUCLIDEAN_DIM,
        subspace_rank: int = W43_DEFAULT_SUBSPACE_RANK,
        subspace_dim: int = W43_DEFAULT_SUBSPACE_DIM,
        attribute_field_order: Sequence[str] | None = None,
) -> ProductManifoldChannelBundle:
    """Encode a :class:`CellObservation` across all five channels +
    the causal-clock bundle."""
    hyp = encode_hyperbolic_branch(
        obs.branch_path, dim=hyperbolic_dim)
    sph = encode_spherical_consensus(
        obs.claim_kinds, dim=spherical_dim)
    euc = encode_euclidean_attributes(
        dict(obs.attributes),
        field_order=attribute_field_order,
        dim=euclidean_dim,
    )
    # Factoradic route: map role_arrival_order into permutation of
    # range(len(role_universe)) using role_universe as the index.
    if obs.role_universe:
        index = {r: i for i, r in enumerate(obs.role_universe)}
        try:
            perm = [index[r] for r in obs.role_arrival_order]
        except KeyError as ex:
            raise ValueError(
                f"role {ex} not in role_universe "
                f"{list(obs.role_universe)}") from ex
        # If the arrival order is incomplete, append missing roles
        # in canonical order (so the factoradic always covers the
        # universe).
        seen = set(perm)
        for i in range(len(obs.role_universe)):
            if i not in seen:
                perm.append(i)
        fac = encode_factoradic_route(perm)
    else:
        fac = encode_factoradic_route(())
    sub = encode_subspace_basis(
        obs.subspace_vectors,
        rank=subspace_rank, dim=subspace_dim,
    )
    admissible = is_causally_admissible(obs.causal_clocks)
    violation = detect_causal_violation_index(obs.causal_clocks)
    return ProductManifoldChannelBundle(
        hyperbolic=hyp,
        spherical=sph,
        euclidean=euc,
        factoradic=fac,
        subspace=sub,
        causal_clocks=tuple(obs.causal_clocks),
        causal_admissible=bool(admissible),
        causal_violation_index=int(violation),
    )


# =============================================================================
# Policy + decision selector
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ProductManifoldPolicyEntry:
    """Controller-side honest policy for one role-handoff signature.

    Maps a registered ``role_handoff_signature_cid`` (W42's
    invariance signature) to:

      * the **expected services** (passed through from W42)
      * the **expected spherical signature** (cosine-agreement
        target on the consensus channel)
      * the **expected subspace basis** (principal-angle drift
        target on the subspace channel)
      * the **expected causal topology hash** (causal-clock-cid
        target on the causal-clock channel)
    """

    role_handoff_signature_cid: str
    expected_services: tuple[str, ...]
    expected_spherical: SphericalConsensusSignature
    expected_subspace: SubspaceBasis
    expected_causal_topology_hash: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w43_policy_entry",
            "role_handoff_signature_cid": str(
                self.role_handoff_signature_cid),
            "expected_services": [str(s)
                                   for s in self.expected_services],
            "expected_spherical_cid": self.expected_spherical.cid(),
            "expected_subspace_cid": self.expected_subspace.cid(),
            "expected_causal_topology_hash": str(
                self.expected_causal_topology_hash),
        })


@dataclasses.dataclass
class ProductManifoldPolicyRegistry:
    """Controller-side mapping of role-handoff signature -> entry."""

    entries: dict[str, ProductManifoldPolicyEntry] = (
        dataclasses.field(default_factory=dict))

    def register(self, entry: ProductManifoldPolicyEntry) -> None:
        self.entries[str(entry.role_handoff_signature_cid)] = entry

    def lookup(
            self, signature_cid: str,
    ) -> "ProductManifoldPolicyEntry | None":
        return self.entries.get(str(signature_cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w43_policy_registry",
            "entries": [
                {"k": k, "v": v.cid()}
                for k, v in sorted(self.entries.items())
            ],
        })


def select_pmc_decision(
        *,
        observed_spherical: SphericalConsensusSignature,
        expected_spherical: SphericalConsensusSignature | None,
        observed_subspace: SubspaceBasis,
        expected_subspace: SubspaceBasis | None,
        causal_admissible: bool,
        policy_match_found: bool,
        spherical_agreement_min: float = (
            W43_DEFAULT_SPHERICAL_AGREEMENT_MIN),
        subspace_drift_max: float = W43_DEFAULT_SUBSPACE_DRIFT_MAX,
) -> tuple[str, float, float]:
    """Closed-form decision selector for the W43 layer.

    Returns ``(branch, spherical_agreement, subspace_drift)``.

    Order of checks:

      1. No policy match -> ``PMC_NO_POLICY``
      2. Causal violation -> ``PMC_CAUSAL_VIOLATION_ABSTAINED``
      3. Subspace drift > max -> ``PMC_SUBSPACE_DRIFT_ABSTAINED``
      4. Spherical agreement < min -> ``PMC_SPHERICAL_DIVERGENCE_ABSTAINED``
      5. Otherwise -> ``PMC_RATIFIED``
    """
    if not policy_match_found:
        return (W43_BRANCH_PMC_NO_POLICY, 0.0, 0.0)
    if not causal_admissible:
        return (W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED, 0.0, 0.0)
    drift = (principal_angle_drift(
                observed_subspace, expected_subspace)
             if expected_subspace is not None else 0.0)
    if drift > subspace_drift_max:
        return (W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED,
                0.0, float(drift))
    agreement = (cosine_agreement(
                    observed_spherical, expected_spherical)
                 if expected_spherical is not None else 1.0)
    if agreement < spherical_agreement_min:
        return (W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED,
                float(agreement), float(drift))
    return (W43_BRANCH_PMC_RATIFIED,
            float(agreement), float(drift))


# =============================================================================
# Component CIDs and manifest-v13
# =============================================================================

def _compute_w43_manifold_state_cid(
        *,
        hyperbolic_cid: str,
        spherical_cid: str,
        euclidean_cid: str,
        cell_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w43_manifold_state",
        "hyperbolic_cid": str(hyperbolic_cid),
        "spherical_cid": str(spherical_cid),
        "euclidean_cid": str(euclidean_cid),
        "cell_index": int(cell_index),
    })


def _compute_w43_factoradic_route_cid(
        *,
        factoradic_cid: str,
        n_structured_bits: int,
        cell_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w43_factoradic_route_anchor",
        "factoradic_cid": str(factoradic_cid),
        "n_structured_bits": int(n_structured_bits),
        "cell_index": int(cell_index),
    })


def _compute_w43_subspace_basis_cid(
        *,
        subspace_cid: str,
        principal_angle_drift_radians: float,
        cell_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w43_subspace_basis_anchor",
        "subspace_cid": str(subspace_cid),
        "principal_angle_drift_radians": float(round(
            principal_angle_drift_radians, 12)),
        "cell_index": int(cell_index),
    })


def _compute_w43_causal_clock_cid(
        *,
        causal_clock_cids: Sequence[str],
        causal_admissible: bool,
        causal_violation_index: int,
        cell_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w43_causal_clock_bundle",
        "causal_clock_cids": [str(c) for c in causal_clock_cids],
        "causal_admissible": bool(causal_admissible),
        "causal_violation_index": int(causal_violation_index),
        "cell_index": int(cell_index),
    })


def _compute_w43_manifold_audit_cid(
        *,
        decision_branch: str,
        spherical_agreement: float,
        subspace_drift: float,
        policy_entry_cid: str,
        cell_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w43_manifold_audit",
        "decision_branch": str(decision_branch),
        "spherical_agreement": float(round(
            spherical_agreement, 12)),
        "subspace_drift": float(round(subspace_drift, 12)),
        "policy_entry_cid": str(policy_entry_cid),
        "cell_index": int(cell_index),
    })


def _compute_w43_manifold_witness_cid(
        *,
        manifold_state_cid: str,
        factoradic_route_cid: str,
        subspace_basis_cid: str,
        causal_clock_cid: str,
        manifold_audit_cid: str,
        n_w42_visible_tokens: int,
        n_w43_visible_tokens: int,
        n_w43_overhead_tokens: int,
        n_structured_bits: int,
) -> str:
    return _sha256_hex({
        "kind": "w43_manifold_witness",
        "manifold_state_cid": str(manifold_state_cid),
        "factoradic_route_cid": str(factoradic_route_cid),
        "subspace_basis_cid": str(subspace_basis_cid),
        "causal_clock_cid": str(causal_clock_cid),
        "manifold_audit_cid": str(manifold_audit_cid),
        "n_w42_visible_tokens": int(n_w42_visible_tokens),
        "n_w43_visible_tokens": int(n_w43_visible_tokens),
        "n_w43_overhead_tokens": int(n_w43_overhead_tokens),
        "n_structured_bits": int(n_structured_bits),
    })


def _compute_w43_manifest_v13_cid(
        *,
        parent_w42_cid: str,
        manifold_state_cid: str,
        factoradic_route_cid: str,
        subspace_basis_cid: str,
        causal_clock_cid: str,
        manifold_audit_cid: str,
        manifold_witness_cid: str,
) -> str:
    return _sha256_hex({
        "kind": "w43_manifest_v13",
        "parent_w42_cid": str(parent_w42_cid),
        "manifold_state_cid": str(manifold_state_cid),
        "factoradic_route_cid": str(factoradic_route_cid),
        "subspace_basis_cid": str(subspace_basis_cid),
        "causal_clock_cid": str(causal_clock_cid),
        "manifold_audit_cid": str(manifold_audit_cid),
        "manifold_witness_cid": str(manifold_witness_cid),
    })


def _compute_w43_outer_cid(
        *,
        schema_cid: str,
        parent_w42_cid: str,
        manifest_v13_cid: str,
        cell_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w43_outer",
        "schema_cid": str(schema_cid),
        "parent_w42_cid": str(parent_w42_cid),
        "manifest_v13_cid": str(manifest_v13_cid),
        "cell_index": int(cell_index),
    })


# =============================================================================
# Envelope, registry, result
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ProductManifoldRatificationEnvelope:
    """Sealed manifest-v13 envelope for one cell of the W43 layer."""

    schema_version: str
    schema_cid: str
    parent_w42_cid: str
    cell_index: int

    decision_branch: str
    role_handoff_signature_cid: str
    policy_entry_cid: str

    hyperbolic_cid: str
    spherical_cid: str
    euclidean_cid: str
    factoradic_cid: str
    subspace_cid: str
    causal_clock_cids: tuple[str, ...]

    manifold_state_cid: str
    factoradic_route_cid: str
    subspace_basis_cid: str
    causal_clock_cid: str
    manifold_audit_cid: str
    manifold_witness_cid: str
    manifest_v13_cid: str

    causal_admissible: bool
    causal_violation_index: int
    spherical_agreement: float
    subspace_drift: float

    n_w42_visible_tokens: int
    n_w43_visible_tokens: int
    n_w43_overhead_tokens: int
    n_structured_bits: int
    n_factoradic_bits: int

    w43_cid: str

    def recompute_w43_cid(self) -> str:
        return _compute_w43_outer_cid(
            schema_cid=self.schema_cid,
            parent_w42_cid=self.parent_w42_cid,
            manifest_v13_cid=self.manifest_v13_cid,
            cell_index=int(self.cell_index),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "parent_w42_cid": self.parent_w42_cid,
            "cell_index": int(self.cell_index),
            "decision_branch": self.decision_branch,
            "role_handoff_signature_cid": (
                self.role_handoff_signature_cid),
            "policy_entry_cid": self.policy_entry_cid,
            "hyperbolic_cid": self.hyperbolic_cid,
            "spherical_cid": self.spherical_cid,
            "euclidean_cid": self.euclidean_cid,
            "factoradic_cid": self.factoradic_cid,
            "subspace_cid": self.subspace_cid,
            "causal_clock_cids": list(self.causal_clock_cids),
            "manifold_state_cid": self.manifold_state_cid,
            "factoradic_route_cid": self.factoradic_route_cid,
            "subspace_basis_cid": self.subspace_basis_cid,
            "causal_clock_cid": self.causal_clock_cid,
            "manifold_audit_cid": self.manifold_audit_cid,
            "manifold_witness_cid": self.manifold_witness_cid,
            "manifest_v13_cid": self.manifest_v13_cid,
            "causal_admissible": bool(self.causal_admissible),
            "causal_violation_index": int(
                self.causal_violation_index),
            "spherical_agreement": float(
                round(self.spherical_agreement, 12)),
            "subspace_drift": float(round(self.subspace_drift, 12)),
            "n_w42_visible_tokens": int(
                self.n_w42_visible_tokens),
            "n_w43_visible_tokens": int(
                self.n_w43_visible_tokens),
            "n_w43_overhead_tokens": int(
                self.n_w43_overhead_tokens),
            "n_structured_bits": int(self.n_structured_bits),
            "n_factoradic_bits": int(self.n_factoradic_bits),
            "w43_cid": self.w43_cid,
        }


@dataclasses.dataclass
class ProductManifoldRegistry:
    """Registry for the W43 product-manifold layer.

    When ``pmc_enabled = False`` AND ``manifest_v13_disabled = True``
    AND ``abstain_on_causal_violation = False`` AND
    ``abstain_on_subspace_drift = False`` AND
    ``abstain_on_spherical_divergence = False``, the W43
    orchestrator reduces to W42 byte-for-byte
    (the W43-L-TRIVIAL-PASSTHROUGH falsifier).
    """

    schema_cid: str
    policy_registry: ProductManifoldPolicyRegistry = (
        dataclasses.field(
            default_factory=ProductManifoldPolicyRegistry))
    pmc_enabled: bool = True
    manifest_v13_disabled: bool = False
    abstain_on_causal_violation: bool = True
    abstain_on_subspace_drift: bool = True
    abstain_on_spherical_divergence: bool = True
    spherical_agreement_min: float = (
        W43_DEFAULT_SPHERICAL_AGREEMENT_MIN)
    subspace_drift_max: float = W43_DEFAULT_SUBSPACE_DRIFT_MAX

    @property
    def is_trivial(self) -> bool:
        return (not self.pmc_enabled
                and self.manifest_v13_disabled
                and not self.abstain_on_causal_violation
                and not self.abstain_on_subspace_drift
                and not self.abstain_on_spherical_divergence)


@dataclasses.dataclass
class W43ProductManifoldResult:
    decision_branch: str
    role_handoff_signature_cid: str
    policy_entry_cid: str
    parent_w42_cid: str
    cell_index: int

    hyperbolic_cid: str
    spherical_cid: str
    euclidean_cid: str
    factoradic_cid: str
    subspace_cid: str
    causal_clock_cids: tuple[str, ...]

    manifold_state_cid: str
    factoradic_route_cid: str
    subspace_basis_cid: str
    causal_clock_cid: str
    manifold_audit_cid: str
    manifold_witness_cid: str
    manifest_v13_cid: str
    w43_cid: str

    causal_admissible: bool
    causal_violation_index: int
    spherical_agreement: float
    subspace_drift: float

    n_w42_visible_tokens: int
    n_w43_visible_tokens: int
    n_w43_overhead_tokens: int
    n_structured_bits: int
    n_factoradic_bits: int

    cram_factor_w43: float
    cram_factor_gain_vs_w42: float

    ratified: bool
    verification_ok: bool
    verification_reason: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# =============================================================================
# Verifier (18 enumerated W43 failure modes)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class W43VerificationOutcome:
    ok: bool
    reason: str
    n_checks: int


def verify_product_manifold_ratification(
        env: "ProductManifoldRatificationEnvelope | None",
        *,
        registered_schema_cid: str,
        registered_parent_w42_cid: str,
) -> W43VerificationOutcome:
    """Pure-function verifier for the W43 envelope.

    Enumerates 18 disjoint W43 failure modes:

    1.  ``empty_w43_envelope``
    2.  ``w43_schema_version_unknown``
    3.  ``w43_schema_cid_mismatch``
    4.  ``w42_parent_cid_mismatch``
    5.  ``w43_decision_branch_unknown``
    6.  ``w43_role_handoff_signature_cid_mismatch``
    7.  ``w43_hyperbolic_cid_invalid``
    8.  ``w43_spherical_cid_invalid``
    9.  ``w43_euclidean_cid_invalid``
    10. ``w43_factoradic_cid_invalid``
    11. ``w43_subspace_cid_invalid``
    12. ``w43_manifold_state_cid_mismatch``
    13. ``w43_factoradic_route_cid_mismatch``
    14. ``w43_subspace_basis_cid_mismatch``
    15. ``w43_causal_clock_cid_mismatch``
    16. ``w43_manifold_audit_cid_mismatch``
    17. ``w43_manifold_witness_cid_mismatch``
    18. ``w43_manifest_v13_cid_mismatch``
    19. ``w43_outer_cid_mismatch``
    20. ``w43_token_accounting_invalid``
    21. ``w43_spherical_agreement_invalid``
    22. ``w43_subspace_drift_invalid``
    """
    n_checks = 0
    if env is None:
        return W43VerificationOutcome(
            ok=False, reason="empty_w43_envelope", n_checks=0)
    n_checks += 1
    if env.schema_version != W43_PRODUCT_MANIFOLD_SCHEMA_VERSION:
        return W43VerificationOutcome(
            ok=False, reason="w43_schema_version_unknown",
            n_checks=n_checks)
    n_checks += 1
    if env.schema_cid != str(registered_schema_cid):
        return W43VerificationOutcome(
            ok=False, reason="w43_schema_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.parent_w42_cid != str(registered_parent_w42_cid):
        return W43VerificationOutcome(
            ok=False, reason="w42_parent_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.decision_branch not in W43_ALL_BRANCHES:
        return W43VerificationOutcome(
            ok=False, reason="w43_decision_branch_unknown",
            n_checks=n_checks)
    n_checks += 1
    # role_handoff signature CID is 64-hex SHA-256. Empty allowed
    # only on the trivial-passthrough branch.
    if env.decision_branch != W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH:
        if (not env.role_handoff_signature_cid
                or len(env.role_handoff_signature_cid) != 64):
            return W43VerificationOutcome(
                ok=False,
                reason="w43_role_handoff_signature_cid_mismatch",
                n_checks=n_checks)
    n_checks += 1
    # Per-channel CIDs must be 64-hex when not on trivial passthrough.
    for field, name in (
        (env.hyperbolic_cid, "w43_hyperbolic_cid_invalid"),
        (env.spherical_cid, "w43_spherical_cid_invalid"),
        (env.euclidean_cid, "w43_euclidean_cid_invalid"),
        (env.factoradic_cid, "w43_factoradic_cid_invalid"),
        (env.subspace_cid, "w43_subspace_cid_invalid"),
    ):
        n_checks += 1
        if env.decision_branch != W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH:
            if not field or len(field) != 64:
                return W43VerificationOutcome(
                    ok=False, reason=name, n_checks=n_checks)

    # Recompute composed CIDs.
    expected_state_cid = _compute_w43_manifold_state_cid(
        hyperbolic_cid=env.hyperbolic_cid,
        spherical_cid=env.spherical_cid,
        euclidean_cid=env.euclidean_cid,
        cell_index=int(env.cell_index),
    )
    n_checks += 1
    if expected_state_cid != env.manifold_state_cid:
        return W43VerificationOutcome(
            ok=False, reason="w43_manifold_state_cid_mismatch",
            n_checks=n_checks)
    expected_route_cid = _compute_w43_factoradic_route_cid(
        factoradic_cid=env.factoradic_cid,
        n_structured_bits=int(env.n_factoradic_bits),
        cell_index=int(env.cell_index),
    )
    n_checks += 1
    if expected_route_cid != env.factoradic_route_cid:
        return W43VerificationOutcome(
            ok=False, reason="w43_factoradic_route_cid_mismatch",
            n_checks=n_checks)
    expected_subspace_cid = _compute_w43_subspace_basis_cid(
        subspace_cid=env.subspace_cid,
        principal_angle_drift_radians=float(env.subspace_drift),
        cell_index=int(env.cell_index),
    )
    n_checks += 1
    if expected_subspace_cid != env.subspace_basis_cid:
        return W43VerificationOutcome(
            ok=False, reason="w43_subspace_basis_cid_mismatch",
            n_checks=n_checks)
    expected_causal_cid = _compute_w43_causal_clock_cid(
        causal_clock_cids=env.causal_clock_cids,
        causal_admissible=bool(env.causal_admissible),
        causal_violation_index=int(env.causal_violation_index),
        cell_index=int(env.cell_index),
    )
    n_checks += 1
    if expected_causal_cid != env.causal_clock_cid:
        return W43VerificationOutcome(
            ok=False, reason="w43_causal_clock_cid_mismatch",
            n_checks=n_checks)
    expected_audit_cid = _compute_w43_manifold_audit_cid(
        decision_branch=env.decision_branch,
        spherical_agreement=float(env.spherical_agreement),
        subspace_drift=float(env.subspace_drift),
        policy_entry_cid=env.policy_entry_cid,
        cell_index=int(env.cell_index),
    )
    n_checks += 1
    if expected_audit_cid != env.manifold_audit_cid:
        return W43VerificationOutcome(
            ok=False, reason="w43_manifold_audit_cid_mismatch",
            n_checks=n_checks)
    expected_witness_cid = _compute_w43_manifold_witness_cid(
        manifold_state_cid=env.manifold_state_cid,
        factoradic_route_cid=env.factoradic_route_cid,
        subspace_basis_cid=env.subspace_basis_cid,
        causal_clock_cid=env.causal_clock_cid,
        manifold_audit_cid=env.manifold_audit_cid,
        n_w42_visible_tokens=int(env.n_w42_visible_tokens),
        n_w43_visible_tokens=int(env.n_w43_visible_tokens),
        n_w43_overhead_tokens=int(env.n_w43_overhead_tokens),
        n_structured_bits=int(env.n_structured_bits),
    )
    n_checks += 1
    if expected_witness_cid != env.manifold_witness_cid:
        return W43VerificationOutcome(
            ok=False, reason="w43_manifold_witness_cid_mismatch",
            n_checks=n_checks)
    expected_manifest = _compute_w43_manifest_v13_cid(
        parent_w42_cid=env.parent_w42_cid,
        manifold_state_cid=env.manifold_state_cid,
        factoradic_route_cid=env.factoradic_route_cid,
        subspace_basis_cid=env.subspace_basis_cid,
        causal_clock_cid=env.causal_clock_cid,
        manifold_audit_cid=env.manifold_audit_cid,
        manifold_witness_cid=env.manifold_witness_cid,
    )
    n_checks += 1
    if expected_manifest != env.manifest_v13_cid:
        return W43VerificationOutcome(
            ok=False, reason="w43_manifest_v13_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.recompute_w43_cid() != env.w43_cid:
        return W43VerificationOutcome(
            ok=False, reason="w43_outer_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if (env.n_w42_visible_tokens < 0
            or env.n_w43_overhead_tokens < 0
            or env.n_w43_visible_tokens != (
                int(env.n_w42_visible_tokens)
                + int(env.n_w43_overhead_tokens))):
        return W43VerificationOutcome(
            ok=False, reason="w43_token_accounting_invalid",
            n_checks=n_checks)
    n_checks += 1
    if not (-1.0 - 1e-9 <= float(env.spherical_agreement) <= 1.0 + 1e-9):
        return W43VerificationOutcome(
            ok=False, reason="w43_spherical_agreement_invalid",
            n_checks=n_checks)
    n_checks += 1
    if not (0.0 <= float(env.subspace_drift)
            <= float(math.pi / 2) + 1e-9):
        return W43VerificationOutcome(
            ok=False, reason="w43_subspace_drift_invalid",
            n_checks=n_checks)
    n_checks += 1
    return W43VerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


# =============================================================================
# Orchestrator
# =============================================================================

@dataclasses.dataclass
class ProductManifoldOrchestrator:
    """W43 product-manifold orchestrator.

    Stateless across cells (each cell is a fresh observation +
    decision). The orchestrator does not wrap a W42 inner — the
    caller passes the W42 outputs explicitly. This keeps the W43
    layer composable: any W42-shaped result + a CellObservation is
    enough to invoke W43.
    """

    registry: ProductManifoldRegistry
    require_w43_verification: bool = True

    _last_result: "W43ProductManifoldResult | None" = None
    _last_envelope: (
        "ProductManifoldRatificationEnvelope | None") = None
    _cell_index: int = 0

    @property
    def schema_cid(self) -> str:
        return str(self.registry.schema_cid)

    def reset_session(self) -> None:
        self._last_result = None
        self._last_envelope = None
        self._cell_index = 0

    def decode(
            self,
            *,
            observation: CellObservation,
            role_handoff_signature_cid: str,
            parent_w42_cid: str,
            n_w42_visible_tokens: int,
    ) -> W43ProductManifoldResult:
        """Compute the W43 result for one cell."""
        cell_index = int(self._cell_index)

        # Channels.
        bundle = encode_cell_channels(observation)
        hyperbolic_cid = bundle.hyperbolic.cid()
        spherical_cid = bundle.spherical.cid()
        euclidean_cid = bundle.euclidean.cid()
        factoradic_cid = bundle.factoradic.cid()
        subspace_cid = bundle.subspace.cid()
        causal_clock_cids = tuple(c.cid()
                                   for c in bundle.causal_clocks)

        # Order: the trivial-passthrough config (which subsumes
        # ``not pmc_enabled``) must short-circuit first so the
        # W43-L-TRIVIAL-PASSTHROUGH falsifier holds. The plain
        # ``not pmc_enabled`` path then catches operator-disabled
        # registries that left other guard-rails on.
        if self.registry.is_trivial:
            self._cell_index += 1
            return self._pack_passthrough(
                cell_index=cell_index,
                bundle=bundle,
                role_handoff_signature_cid=(
                    role_handoff_signature_cid),
                parent_w42_cid=parent_w42_cid,
                n_w42_visible_tokens=n_w42_visible_tokens,
                decision_branch=(
                    W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH),
                hyperbolic_cid="",
                spherical_cid="",
                euclidean_cid="",
                factoradic_cid="",
                subspace_cid="",
                causal_clock_cids=tuple(),
                verify_reason="trivial_passthrough",
            )

        if not self.registry.pmc_enabled:
            self._cell_index += 1
            return self._pack_passthrough(
                cell_index=cell_index,
                bundle=bundle,
                role_handoff_signature_cid=(
                    role_handoff_signature_cid),
                parent_w42_cid=parent_w42_cid,
                n_w42_visible_tokens=n_w42_visible_tokens,
                decision_branch=W43_BRANCH_PMC_DISABLED,
                hyperbolic_cid=hyperbolic_cid,
                spherical_cid=spherical_cid,
                euclidean_cid=euclidean_cid,
                factoradic_cid=factoradic_cid,
                subspace_cid=subspace_cid,
                causal_clock_cids=causal_clock_cids,
                verify_reason="disabled",
            )

        # Look up policy for this signature.
        policy = self.registry.policy_registry.lookup(
            role_handoff_signature_cid)
        if policy is None:
            policy_match_found = False
            policy_entry_cid = ""
            expected_spherical: SphericalConsensusSignature | None = (
                None)
            expected_subspace: SubspaceBasis | None = None
        else:
            policy_match_found = True
            policy_entry_cid = policy.cid()
            expected_spherical = policy.expected_spherical
            expected_subspace = policy.expected_subspace

        decision_branch, agreement, drift = select_pmc_decision(
            observed_spherical=bundle.spherical,
            expected_spherical=expected_spherical,
            observed_subspace=bundle.subspace,
            expected_subspace=expected_subspace,
            causal_admissible=bundle.causal_admissible,
            policy_match_found=policy_match_found,
            spherical_agreement_min=(
                self.registry.spherical_agreement_min),
            subspace_drift_max=self.registry.subspace_drift_max,
        )

        # If a guard-rail is disabled, fall back to ratification on
        # the corresponding abstention path.
        if (decision_branch
                == W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED
                and not self.registry.abstain_on_causal_violation):
            decision_branch = W43_BRANCH_PMC_RATIFIED
        if (decision_branch
                == W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED
                and not self.registry.abstain_on_subspace_drift):
            decision_branch = W43_BRANCH_PMC_RATIFIED
        if (decision_branch
                == W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED
                and not (
                    self.registry.abstain_on_spherical_divergence)):
            decision_branch = W43_BRANCH_PMC_RATIFIED

        # Overhead: 1 visible token when the W43 layer makes an
        # active decision (anything other than the no-trigger and
        # no-policy paths).
        if decision_branch in (
                W43_BRANCH_PMC_RATIFIED,
                W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED,
                W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED,
                W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED):
            n_w43_overhead = 1
        else:
            n_w43_overhead = 0

        n_w43_visible = (int(n_w42_visible_tokens)
                         + int(n_w43_overhead))

        # Component CIDs.
        n_factoradic_bits = bundle.factoradic.n_structured_bits()
        manifold_state_cid = _compute_w43_manifold_state_cid(
            hyperbolic_cid=hyperbolic_cid,
            spherical_cid=spherical_cid,
            euclidean_cid=euclidean_cid,
            cell_index=cell_index,
        )
        factoradic_route_cid = _compute_w43_factoradic_route_cid(
            factoradic_cid=factoradic_cid,
            n_structured_bits=n_factoradic_bits,
            cell_index=cell_index,
        )
        subspace_basis_cid = _compute_w43_subspace_basis_cid(
            subspace_cid=subspace_cid,
            principal_angle_drift_radians=float(drift),
            cell_index=cell_index,
        )
        causal_clock_cid = _compute_w43_causal_clock_cid(
            causal_clock_cids=causal_clock_cids,
            causal_admissible=bundle.causal_admissible,
            causal_violation_index=bundle.causal_violation_index,
            cell_index=cell_index,
        )
        manifold_audit_cid = _compute_w43_manifold_audit_cid(
            decision_branch=decision_branch,
            spherical_agreement=float(agreement),
            subspace_drift=float(drift),
            policy_entry_cid=policy_entry_cid,
            cell_index=cell_index,
        )

        # Structured bit count: 7 component CIDs * 256 bits +
        # factoradic information capacity.
        n_structured_bits = (7 * 256) + int(n_factoradic_bits)

        manifold_witness_cid = _compute_w43_manifold_witness_cid(
            manifold_state_cid=manifold_state_cid,
            factoradic_route_cid=factoradic_route_cid,
            subspace_basis_cid=subspace_basis_cid,
            causal_clock_cid=causal_clock_cid,
            manifold_audit_cid=manifold_audit_cid,
            n_w42_visible_tokens=int(n_w42_visible_tokens),
            n_w43_visible_tokens=int(n_w43_visible),
            n_w43_overhead_tokens=int(n_w43_overhead),
            n_structured_bits=int(n_structured_bits),
        )

        manifest_v13_cid = (
            "" if self.registry.manifest_v13_disabled
            else _compute_w43_manifest_v13_cid(
                parent_w42_cid=parent_w42_cid,
                manifold_state_cid=manifold_state_cid,
                factoradic_route_cid=factoradic_route_cid,
                subspace_basis_cid=subspace_basis_cid,
                causal_clock_cid=causal_clock_cid,
                manifold_audit_cid=manifold_audit_cid,
                manifold_witness_cid=manifold_witness_cid,
            ))

        if manifest_v13_cid:
            w43_cid = _compute_w43_outer_cid(
                schema_cid=self.schema_cid,
                parent_w42_cid=parent_w42_cid,
                manifest_v13_cid=manifest_v13_cid,
                cell_index=cell_index,
            )
        else:
            w43_cid = ""

        envelope = ProductManifoldRatificationEnvelope(
            schema_version=W43_PRODUCT_MANIFOLD_SCHEMA_VERSION,
            schema_cid=self.schema_cid,
            parent_w42_cid=str(parent_w42_cid),
            cell_index=cell_index,
            decision_branch=decision_branch,
            role_handoff_signature_cid=str(
                role_handoff_signature_cid),
            policy_entry_cid=str(policy_entry_cid),
            hyperbolic_cid=hyperbolic_cid,
            spherical_cid=spherical_cid,
            euclidean_cid=euclidean_cid,
            factoradic_cid=factoradic_cid,
            subspace_cid=subspace_cid,
            causal_clock_cids=tuple(causal_clock_cids),
            manifold_state_cid=manifold_state_cid,
            factoradic_route_cid=factoradic_route_cid,
            subspace_basis_cid=subspace_basis_cid,
            causal_clock_cid=causal_clock_cid,
            manifold_audit_cid=manifold_audit_cid,
            manifold_witness_cid=manifold_witness_cid,
            manifest_v13_cid=manifest_v13_cid,
            causal_admissible=bool(bundle.causal_admissible),
            causal_violation_index=int(
                bundle.causal_violation_index),
            spherical_agreement=float(agreement),
            subspace_drift=float(drift),
            n_w42_visible_tokens=int(n_w42_visible_tokens),
            n_w43_visible_tokens=int(n_w43_visible),
            n_w43_overhead_tokens=int(n_w43_overhead),
            n_structured_bits=int(n_structured_bits),
            n_factoradic_bits=int(n_factoradic_bits),
            w43_cid=w43_cid,
        )

        outcome = verify_product_manifold_ratification(
            envelope,
            registered_schema_cid=self.schema_cid,
            registered_parent_w42_cid=parent_w42_cid,
        )
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)

        if not verify_ok and self.require_w43_verification:
            self._cell_index += 1
            return self._pack_result(
                cell_index=cell_index,
                envelope=envelope,
                decision_branch=W43_BRANCH_PMC_REJECTED,
                ratified=False,
                verify_ok=False,
                verify_reason=verify_reason,
                bundle=bundle,
                spherical_agreement=float(agreement),
                subspace_drift=float(drift),
                n_w42_visible_tokens=int(n_w42_visible_tokens),
                n_w43_visible_tokens=int(n_w43_visible),
                n_w43_overhead_tokens=int(n_w43_overhead),
                n_structured_bits=int(n_structured_bits),
                n_factoradic_bits=int(n_factoradic_bits),
            )

        self._cell_index += 1
        ratified = (decision_branch
                     == W43_BRANCH_PMC_RATIFIED)
        return self._pack_result(
            cell_index=cell_index,
            envelope=envelope,
            decision_branch=decision_branch,
            ratified=bool(ratified),
            verify_ok=verify_ok,
            verify_reason=verify_reason,
            bundle=bundle,
            spherical_agreement=float(agreement),
            subspace_drift=float(drift),
            n_w42_visible_tokens=int(n_w42_visible_tokens),
            n_w43_visible_tokens=int(n_w43_visible),
            n_w43_overhead_tokens=int(n_w43_overhead),
            n_structured_bits=int(n_structured_bits),
            n_factoradic_bits=int(n_factoradic_bits),
        )

    def _pack_passthrough(
            self,
            *,
            cell_index: int,
            bundle: ProductManifoldChannelBundle,
            role_handoff_signature_cid: str,
            parent_w42_cid: str,
            n_w42_visible_tokens: int,
            decision_branch: str,
            hyperbolic_cid: str,
            spherical_cid: str,
            euclidean_cid: str,
            factoradic_cid: str,
            subspace_cid: str,
            causal_clock_cids: tuple[str, ...],
            verify_reason: str,
    ) -> W43ProductManifoldResult:
        result = W43ProductManifoldResult(
            decision_branch=str(decision_branch),
            role_handoff_signature_cid=str(
                role_handoff_signature_cid),
            policy_entry_cid="",
            parent_w42_cid=str(parent_w42_cid),
            cell_index=int(cell_index),
            hyperbolic_cid=str(hyperbolic_cid),
            spherical_cid=str(spherical_cid),
            euclidean_cid=str(euclidean_cid),
            factoradic_cid=str(factoradic_cid),
            subspace_cid=str(subspace_cid),
            causal_clock_cids=tuple(causal_clock_cids),
            manifold_state_cid="",
            factoradic_route_cid="",
            subspace_basis_cid="",
            causal_clock_cid="",
            manifold_audit_cid="",
            manifold_witness_cid="",
            manifest_v13_cid="",
            w43_cid="",
            causal_admissible=bool(bundle.causal_admissible),
            causal_violation_index=int(
                bundle.causal_violation_index),
            spherical_agreement=0.0,
            subspace_drift=0.0,
            n_w42_visible_tokens=int(n_w42_visible_tokens),
            n_w43_visible_tokens=int(n_w42_visible_tokens),
            n_w43_overhead_tokens=0,
            n_structured_bits=0,
            n_factoradic_bits=0,
            cram_factor_w43=0.0,
            cram_factor_gain_vs_w42=0.0,
            ratified=(decision_branch
                       == W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH),
            verification_ok=(decision_branch
                              == W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH),
            verification_reason=str(verify_reason),
        )
        self._last_result = result
        self._last_envelope = None
        return result

    def _pack_result(
            self,
            *,
            cell_index: int,
            envelope: ProductManifoldRatificationEnvelope,
            decision_branch: str,
            ratified: bool,
            verify_ok: bool,
            verify_reason: str,
            bundle: ProductManifoldChannelBundle,
            spherical_agreement: float,
            subspace_drift: float,
            n_w42_visible_tokens: int,
            n_w43_visible_tokens: int,
            n_w43_overhead_tokens: int,
            n_structured_bits: int,
            n_factoradic_bits: int,
    ) -> W43ProductManifoldResult:
        wire = max(1, int(n_w43_overhead_tokens))
        cram_w43 = (
            float(n_structured_bits) / float(wire)
            if int(n_structured_bits) > 0 else 0.0)
        # The W42 cram baseline is 6*256 structured bits per
        # W42 active cell at 1 visible-token overhead (per W42
        # construction). Where the W42 layer would not have made an
        # active decision (no overhead), the W42 cram_factor is 0.
        cram_w42 = 6.0 * 256.0  # bits per visible token (W42 active)
        cram_gain = float(cram_w43 - cram_w42)

        result = W43ProductManifoldResult(
            decision_branch=str(decision_branch),
            role_handoff_signature_cid=envelope.role_handoff_signature_cid,
            policy_entry_cid=envelope.policy_entry_cid,
            parent_w42_cid=envelope.parent_w42_cid,
            cell_index=int(cell_index),
            hyperbolic_cid=envelope.hyperbolic_cid,
            spherical_cid=envelope.spherical_cid,
            euclidean_cid=envelope.euclidean_cid,
            factoradic_cid=envelope.factoradic_cid,
            subspace_cid=envelope.subspace_cid,
            causal_clock_cids=envelope.causal_clock_cids,
            manifold_state_cid=envelope.manifold_state_cid,
            factoradic_route_cid=envelope.factoradic_route_cid,
            subspace_basis_cid=envelope.subspace_basis_cid,
            causal_clock_cid=envelope.causal_clock_cid,
            manifold_audit_cid=envelope.manifold_audit_cid,
            manifold_witness_cid=envelope.manifold_witness_cid,
            manifest_v13_cid=envelope.manifest_v13_cid,
            w43_cid=envelope.w43_cid,
            causal_admissible=bool(bundle.causal_admissible),
            causal_violation_index=int(
                bundle.causal_violation_index),
            spherical_agreement=float(spherical_agreement),
            subspace_drift=float(subspace_drift),
            n_w42_visible_tokens=int(n_w42_visible_tokens),
            n_w43_visible_tokens=int(n_w43_visible_tokens),
            n_w43_overhead_tokens=int(n_w43_overhead_tokens),
            n_structured_bits=int(n_structured_bits),
            n_factoradic_bits=int(n_factoradic_bits),
            cram_factor_w43=float(cram_w43),
            cram_factor_gain_vs_w42=float(cram_gain),
            ratified=bool(ratified),
            verification_ok=bool(verify_ok),
            verification_reason=str(verify_reason),
        )
        self._last_result = result
        self._last_envelope = envelope
        return result

    @property
    def last_result(self) -> "W43ProductManifoldResult | None":
        return self._last_result

    @property
    def last_envelope(self) -> (
            "ProductManifoldRatificationEnvelope | None"):
        return self._last_envelope


# =============================================================================
# Builders
# =============================================================================

def build_trivial_product_manifold_registry(
        *,
        schema_cid: str | None = None,
) -> ProductManifoldRegistry:
    """Build a registry whose orchestrator reduces to W42 byte-for-
    byte (the W43-L-TRIVIAL-PASSTHROUGH falsifier)."""
    cid = schema_cid or _sha256_hex({
        "kind": "w43_trivial_schema",
    })
    return ProductManifoldRegistry(
        schema_cid=str(cid),
        policy_registry=ProductManifoldPolicyRegistry(),
        pmc_enabled=False,
        manifest_v13_disabled=True,
        abstain_on_causal_violation=False,
        abstain_on_subspace_drift=False,
        abstain_on_spherical_divergence=False,
    )


def build_product_manifold_registry(
        *,
        schema_cid: str,
        policy_entries: Sequence[ProductManifoldPolicyEntry] = (),
        pmc_enabled: bool = True,
        manifest_v13_disabled: bool = False,
        abstain_on_causal_violation: bool = True,
        abstain_on_subspace_drift: bool = True,
        abstain_on_spherical_divergence: bool = True,
        spherical_agreement_min: float = (
            W43_DEFAULT_SPHERICAL_AGREEMENT_MIN),
        subspace_drift_max: float = W43_DEFAULT_SUBSPACE_DRIFT_MAX,
) -> ProductManifoldRegistry:
    policy = ProductManifoldPolicyRegistry()
    for e in policy_entries:
        policy.register(e)
    return ProductManifoldRegistry(
        schema_cid=str(schema_cid),
        policy_registry=policy,
        pmc_enabled=bool(pmc_enabled),
        manifest_v13_disabled=bool(manifest_v13_disabled),
        abstain_on_causal_violation=bool(
            abstain_on_causal_violation),
        abstain_on_subspace_drift=bool(abstain_on_subspace_drift),
        abstain_on_spherical_divergence=bool(
            abstain_on_spherical_divergence),
        spherical_agreement_min=float(spherical_agreement_min),
        subspace_drift_max=float(subspace_drift_max),
    )


__all__ = [
    # Schema, branches, defaults
    "W43_PRODUCT_MANIFOLD_SCHEMA_VERSION",
    "W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH",
    "W43_BRANCH_PMC_DISABLED",
    "W43_BRANCH_PMC_REJECTED",
    "W43_BRANCH_PMC_NO_TRIGGER",
    "W43_BRANCH_PMC_RATIFIED",
    "W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED",
    "W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED",
    "W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED",
    "W43_BRANCH_PMC_NO_POLICY",
    "W43_ALL_BRANCHES",
    "W43_DEFAULT_HYPERBOLIC_DIM",
    "W43_DEFAULT_SPHERICAL_DIM",
    "W43_DEFAULT_EUCLIDEAN_DIM",
    "W43_DEFAULT_SUBSPACE_RANK",
    "W43_DEFAULT_SUBSPACE_DIM",
    "W43_DEFAULT_SPHERICAL_AGREEMENT_MIN",
    "W43_DEFAULT_SUBSPACE_DRIFT_MAX",
    "W43_DEFAULT_HYPERBOLIC_RADIUS_MAX",
    # Channels
    "HyperbolicBranchEncoding", "encode_hyperbolic_branch",
    "decode_hyperbolic_branch_path_prefix",
    "SphericalConsensusSignature", "encode_spherical_consensus",
    "cosine_agreement",
    "EuclideanAttributeVector", "encode_euclidean_attributes",
    "FactoradicRoute", "encode_factoradic_route",
    "decode_factoradic_route",
    "SubspaceBasis", "encode_subspace_basis",
    "principal_angle_drift",
    "CausalVectorClock", "causally_dominates",
    "is_causally_admissible", "detect_causal_violation_index",
    # Bundle / observation
    "ProductManifoldChannelBundle", "CellObservation",
    "encode_cell_channels",
    # Policy
    "ProductManifoldPolicyEntry",
    "ProductManifoldPolicyRegistry",
    # Decision selector
    "select_pmc_decision",
    # Envelope, registry, result
    "ProductManifoldRatificationEnvelope",
    "ProductManifoldRegistry", "W43ProductManifoldResult",
    "W43VerificationOutcome",
    # Verifier
    "verify_product_manifold_ratification",
    # Orchestrator
    "ProductManifoldOrchestrator",
    # Builders
    "build_trivial_product_manifold_registry",
    "build_product_manifold_registry",
]
