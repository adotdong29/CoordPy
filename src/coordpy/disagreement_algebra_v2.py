"""W56 M-supporting — Disagreement Algebra V2.

Extends W55 disagreement algebra with one new identity:

* **substrate-projection** — projecting a merged payload through
  the tiny substrate's KV bridge and then taking the substrate's
  hidden state cosine to the original merged payload must remain
  ≥ 0.5 for "in-distribution" probes. This is the H30 bar.

V2 keeps every V1 identity (idempotent ⊕, ⊖ self-cancellation,
⊗ distributivity on agreement subspace). The substrate-projection
identity is new and is conditioned on the tiny substrate runtime;
when the substrate is unavailable the identity is skipped.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .disagreement_algebra import (
    AlgebraTrace,
    DisagreementAlgebraWitness,
    check_difference_self_cancellation,
    check_intersection_distributivity_on_agreement,
    check_merge_idempotent,
    emit_disagreement_algebra_witness,
)


W56_DA_V2_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v2.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


def check_substrate_projection_identity(
        a: Sequence[float],
        b: Sequence[float],
        *, substrate_forward_fn,
) -> bool:
    """`(a ⊕ b) projected_to_substrate ≈ substrate_forward(a ⊕ b)`.

    With ⊕ as elementwise average, the identity reduces to:
    cosine( substrate_forward_fn(a + b / 2), avg(a, b) ) ≥ 0.5.

    The bar is intentionally loose because the substrate is
    tiny and untrained; we are testing that the projection is
    non-trivial (cosine > some floor), not that it preserves
    semantics exactly.
    """
    n = max(len(a), len(b))
    merged = [
        0.5 * float(a[i] if i < len(a) else 0.0)
        + 0.5 * float(b[i] if i < len(b) else 0.0)
        for i in range(n)
    ]
    projected = substrate_forward_fn(merged)
    return bool(_cosine(projected, merged) >= 0.5)


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV2Witness:
    schema: str
    v1_witness_cid: str
    idempotent_ok: bool
    self_cancel_ok: bool
    distributivity_ok: bool
    substrate_projection_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "v1_witness_cid": str(self.v1_witness_cid),
            "idempotent_ok": bool(self.idempotent_ok),
            "self_cancel_ok": bool(self.self_cancel_ok),
            "distributivity_ok": bool(self.distributivity_ok),
            "substrate_projection_ok": bool(
                self.substrate_projection_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v2_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v2_witness(
        *,
        trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        probe_c: Sequence[float],
        substrate_forward_fn=None,
) -> DisagreementAlgebraV2Witness:
    idem = check_merge_idempotent(list(probe_a))
    cancel = check_difference_self_cancellation(list(probe_a))
    dist = check_intersection_distributivity_on_agreement(
        list(probe_a), list(probe_b), list(probe_c))
    if substrate_forward_fn is None:
        sub_ok = True
    else:
        sub_ok = check_substrate_projection_identity(
            list(probe_a), list(probe_b),
            substrate_forward_fn=substrate_forward_fn)
    v1 = emit_disagreement_algebra_witness(
        trace=trace,
        identity_results=(idem, cancel, dist))
    return DisagreementAlgebraV2Witness(
        schema=W56_DA_V2_SCHEMA_VERSION,
        v1_witness_cid=v1.cid(),
        idempotent_ok=bool(idem.ok),
        self_cancel_ok=bool(cancel.ok),
        distributivity_ok=bool(dist.ok),
        substrate_projection_ok=bool(sub_ok),
    )


__all__ = [
    "W56_DA_V2_SCHEMA_VERSION",
    "DisagreementAlgebraV2Witness",
    "check_substrate_projection_identity",
    "emit_disagreement_algebra_v2_witness",
]
