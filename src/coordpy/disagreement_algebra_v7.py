"""W61 M17 — Disagreement Algebra V7.

Strictly extends W60's ``coordpy.disagreement_algebra_v6``. V7 adds
one new identity:

* **attention-pattern-equivalence identity** — steering the
  substrate via attention-steering V5 to match a reference
  attention pattern produces a top-K position set whose Jaccard
  with the reference is ≥ ``jaccard_floor``. The identity holds
  iff Jaccard ≥ floor AND the substrate's logit argmax is
  preserved.

V7 strictly extends V6: when ``attention_pattern_oracle = None``,
V7 reduces to V6 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v6 import (
    DisagreementAlgebraV6Witness,
    emit_disagreement_algebra_v6_witness,
)


W61_DA_V7_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v7.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def check_attention_pattern_equivalence_identity(
        *, attention_pattern_oracle: Callable[[], tuple[bool, float]],
        jaccard_floor: float = 0.5,
) -> bool:
    """Oracle returns ``(argmax_preserved_bool, top_k_jaccard_float)``.
    Identity holds iff argmax_preserved AND jaccard ≥ floor.
    """
    try:
        argmax_ok, jaccard = attention_pattern_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(jaccard) >= float(jaccard_floor))


def attention_pattern_equivalence_falsifier(
        *, attention_pattern_oracle: Callable[[], tuple[bool, float]],
) -> bool:
    """Falsifier: identity should fail when argmax_preserved=False
    or jaccard is below floor. Returns True if the oracle reports a
    legitimate failure (the identity correctly predicts non-
    equivalence)."""
    try:
        argmax_ok, jaccard = attention_pattern_oracle()
    except Exception:
        return True
    return (not bool(argmax_ok)) or float(jaccard) < 0.5


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV7Witness:
    schema: str
    inner_v6_witness_cid: str
    attention_pattern_equiv_ok: bool
    attention_pattern_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v6_witness_cid": str(
                self.inner_v6_witness_cid),
            "attention_pattern_equiv_ok": bool(
                self.attention_pattern_equiv_ok),
            "attention_pattern_falsifier_ok": bool(
                self.attention_pattern_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v7_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v7_witness(
        *,
        trace: AlgebraTrace,
        probe_a: Sequence[float], probe_b: Sequence[float],
        probe_c: Sequence[float],
        substrate_forward_fn: Any = None,
        hidden_state_projector: Any = None,
        steering_oracle: Any = None,
        cache_reuse_oracle: Any = None,
        retrieval_replay_oracle: Any = None,
        replay_controller_oracle: Any = None,
        attention_pattern_oracle: Any = None,
        attention_pattern_falsifier_oracle: Any = None,
) -> DisagreementAlgebraV7Witness:
    v6 = emit_disagreement_algebra_v6_witness(
        trace=trace, probe_a=probe_a, probe_b=probe_b,
        probe_c=probe_c,
        substrate_forward_fn=substrate_forward_fn,
        hidden_state_projector=hidden_state_projector,
        steering_oracle=steering_oracle,
        cache_reuse_oracle=cache_reuse_oracle,
        retrieval_replay_oracle=retrieval_replay_oracle,
        replay_controller_oracle=replay_controller_oracle)
    ap_ok = True
    if attention_pattern_oracle is not None:
        ap_ok = bool(
            check_attention_pattern_equivalence_identity(
                attention_pattern_oracle=(
                    attention_pattern_oracle)))
    falsifier_ok = True
    if attention_pattern_falsifier_oracle is not None:
        falsifier_ok = bool(
            attention_pattern_equivalence_falsifier(
                attention_pattern_oracle=(
                    attention_pattern_falsifier_oracle)))
    return DisagreementAlgebraV7Witness(
        schema=W61_DA_V7_SCHEMA_VERSION,
        inner_v6_witness_cid=str(v6.cid()),
        attention_pattern_equiv_ok=bool(ap_ok),
        attention_pattern_falsifier_ok=bool(falsifier_ok),
    )


__all__ = [
    "W61_DA_V7_SCHEMA_VERSION",
    "DisagreementAlgebraV7Witness",
    "check_attention_pattern_equivalence_identity",
    "attention_pattern_equivalence_falsifier",
    "emit_disagreement_algebra_v7_witness",
]
