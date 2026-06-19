"""W68 M17 — Disagreement Algebra V14.

Strictly extends W67's ``coordpy.disagreement_algebra_v13``. V13
added the Bregman-equivalence identity. V14 adds:

* **Agent-replacement equivalence identity** — two probes are
  agent-replacement-equivalent iff argmax preserved AND the
  symmetric KL divergence ≤ ``ar_floor`` AND the agent fingerprint
  hashes agree under the agent-replacement permutation.
* **Agent-replacement falsifier** — triggers when ar > floor.

V14 reduces to V13 byte-for-byte when no agent-replacement oracle
is provided.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v13 import (
    bregman_divergence_kl,
    emit_disagreement_algebra_v13_witness,
)
from .tiny_substrate_v3 import _sha256_hex


W68_DA_V14_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v14.v1")


def symmetric_kl(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """Symmetric KL divergence (0.5 * KL(a||b) + 0.5 * KL(b||a))."""
    if not a or not b:
        return 0.0
    return float(
        0.5 * bregman_divergence_kl(a, b)
        + 0.5 * bregman_divergence_kl(b, a))


def check_agent_replacement_equivalence_identity(
        *, ar_oracle: Callable[[], tuple[bool, float, bool]],
        ar_floor: float = 0.20,
) -> bool:
    """Calls the oracle to get
    ``(argmax_ok, kl_value, fingerprint_match)``."""
    try:
        argmax_ok, kl, fp_ok = ar_oracle()
    except Exception:
        return False
    return (bool(argmax_ok) and bool(fp_ok)
            and float(kl) <= float(ar_floor))


def agent_replacement_equivalence_falsifier(
        *, ar_oracle: Callable[[], tuple[bool, float, bool]],
        ar_floor: float = 0.20,
) -> bool:
    """Returns True (falsifier triggers) iff equivalence fails."""
    try:
        argmax_ok, kl, fp_ok = ar_oracle()
    except Exception:
        return True
    return (
        (not bool(argmax_ok))
        or (not bool(fp_ok))
        or (float(kl) > float(ar_floor)))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV14Witness:
    schema: str
    inner_v13_witness_cid: str
    agent_replacement_equiv_ok: bool
    agent_replacement_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v13_witness_cid": str(
                self.inner_v13_witness_cid),
            "agent_replacement_equiv_ok": bool(
                self.agent_replacement_equiv_ok),
            "agent_replacement_falsifier_ok": bool(
                self.agent_replacement_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v14_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v14_witness(
        *, trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        probe_c: Sequence[float],
        ar_oracle: (
            Callable[[], tuple[bool, float, bool]] | None) = None,
        ar_falsifier_oracle: (
            Callable[[], tuple[bool, float, bool]] | None) = None,
        ar_floor: float = 0.20,
        **v13_kwargs: Any,
) -> DisagreementAlgebraV14Witness:
    v13 = emit_disagreement_algebra_v13_witness(
        trace=trace, probe_a=probe_a, probe_b=probe_b,
        probe_c=probe_c, **v13_kwargs)
    ar_ok = True
    if ar_oracle is not None:
        ar_ok = bool(
            check_agent_replacement_equivalence_identity(
                ar_oracle=ar_oracle, ar_floor=float(ar_floor)))
    ar_fals_ok = True
    if ar_falsifier_oracle is not None:
        ar_fals_ok = bool(
            agent_replacement_equivalence_falsifier(
                ar_oracle=ar_falsifier_oracle,
                ar_floor=float(ar_floor)))
    return DisagreementAlgebraV14Witness(
        schema=W68_DA_V14_SCHEMA_VERSION,
        inner_v13_witness_cid=str(v13.cid()),
        agent_replacement_equiv_ok=bool(ar_ok),
        agent_replacement_falsifier_ok=bool(ar_fals_ok),
    )


__all__ = [
    "W68_DA_V14_SCHEMA_VERSION",
    "symmetric_kl",
    "check_agent_replacement_equivalence_identity",
    "agent_replacement_equivalence_falsifier",
    "DisagreementAlgebraV14Witness",
    "emit_disagreement_algebra_v14_witness",
]
