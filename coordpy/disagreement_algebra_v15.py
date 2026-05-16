"""W69 M16 — Disagreement Algebra V15.

Strictly extends W68's ``coordpy.disagreement_algebra_v14``. V14
added the agent-replacement equivalence identity. V15 adds:

* **Multi-branch-rejoin equivalence identity** — two probes are
  multi-branch-rejoin-equivalent iff argmax preserved AND symmetric
  KL ≤ ``mbr_floor`` AND the multi-branch-rejoin fingerprint
  matches.
* **Multi-branch-rejoin falsifier** — triggers when mbr > floor or
  argmax flips or fingerprint mismatches.

V15 reduces to V14 byte-for-byte when no multi-branch-rejoin
oracle is provided.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v14 import (
    symmetric_kl,
)
from .tiny_substrate_v3 import _sha256_hex


W69_DA_V15_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v15.v1")


def check_multi_branch_rejoin_equivalence_identity(
        *, mbr_oracle: Callable[[], tuple[bool, float, bool]],
        mbr_floor: float = 0.20,
) -> bool:
    """Calls the oracle to get
    ``(argmax_ok, kl_value, fingerprint_match)``."""
    try:
        argmax_ok, kl, fp_ok = mbr_oracle()
    except Exception:
        return False
    return (bool(argmax_ok) and bool(fp_ok)
            and float(kl) <= float(mbr_floor))


def multi_branch_rejoin_equivalence_falsifier(
        *, mbr_oracle: Callable[[], tuple[bool, float, bool]],
        mbr_floor: float = 0.20,
) -> bool:
    """Returns True (falsifier triggers) iff equivalence fails."""
    try:
        argmax_ok, kl, fp_ok = mbr_oracle()
    except Exception:
        return True
    return (
        (not bool(argmax_ok))
        or (not bool(fp_ok))
        or (float(kl) > float(mbr_floor)))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV15Witness:
    schema: str
    inner_v14_witness_cid: str
    multi_branch_rejoin_equiv_ok: bool
    multi_branch_rejoin_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v14_witness_cid": str(
                self.inner_v14_witness_cid),
            "multi_branch_rejoin_equiv_ok": bool(
                self.multi_branch_rejoin_equiv_ok),
            "multi_branch_rejoin_falsifier_ok": bool(
                self.multi_branch_rejoin_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v15_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v15_witness(
        *, trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        inner_v14_witness_cid: str = "",
        mbr_oracle: Callable[[], tuple[bool, float, bool]] | None = None,
        mbr_floor: float = 0.20,
) -> DisagreementAlgebraV15Witness:
    if mbr_oracle is None:
        # No-op: V15 reduces to V14 byte-for-byte.
        return DisagreementAlgebraV15Witness(
            schema=W69_DA_V15_SCHEMA_VERSION,
            inner_v14_witness_cid=str(inner_v14_witness_cid),
            multi_branch_rejoin_equiv_ok=True,
            multi_branch_rejoin_falsifier_ok=False,
        )
    equiv = check_multi_branch_rejoin_equivalence_identity(
        mbr_oracle=mbr_oracle, mbr_floor=float(mbr_floor))
    fal = multi_branch_rejoin_equivalence_falsifier(
        mbr_oracle=mbr_oracle, mbr_floor=float(mbr_floor))
    return DisagreementAlgebraV15Witness(
        schema=W69_DA_V15_SCHEMA_VERSION,
        inner_v14_witness_cid=str(inner_v14_witness_cid),
        multi_branch_rejoin_equiv_ok=bool(equiv),
        multi_branch_rejoin_falsifier_ok=bool(fal),
    )


__all__ = [
    "W69_DA_V15_SCHEMA_VERSION",
    "check_multi_branch_rejoin_equivalence_identity",
    "multi_branch_rejoin_equivalence_falsifier",
    "DisagreementAlgebraV15Witness",
    "emit_disagreement_algebra_v15_witness",
]
