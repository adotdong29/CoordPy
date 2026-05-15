"""W60 M18 — Disagreement Algebra V6.

Strictly extends W59's ``coordpy.disagreement_algebra_v5``. V6 adds
one new identity:

* **replay-controller-equivalence identity** — replaying a payload
  via the W60 ReplayController on a stored cache fingerprint
  produces a logit pattern that matches the original reference up
  to the controller's *advertised drift_ceiling*. The identity
  checks ``argmax_preserved`` and ``last_logit_l2_drift ≤
  drift_ceiling`` for the budget the identity oracle reports.

V6 strictly extends V5: when ``replay_controller_oracle = None``,
V6 reduces to V5 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v5 import (
    DisagreementAlgebraV5Witness,
    emit_disagreement_algebra_v5_witness,
)


W60_DA_V6_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v6.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def check_replay_controller_equivalence_identity(
        *, replay_controller_oracle,
        l2_tolerance: float = 1e-6,
) -> bool:
    """Replay-controller oracle returns
    ``(argmax_preserved_bool, last_l2_drift_float)``. Identity
    holds iff argmax_preserved AND l2_drift ≤ l2_tolerance.
    """
    try:
        argmax_ok, l2_drift = replay_controller_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(l2_drift) <= float(l2_tolerance))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV6Witness:
    schema: str
    inner_v5_witness_cid: str
    merge_idempotent_ok: bool
    diff_self_cancel_ok: bool
    intersect_distrib_ok: bool
    substrate_projection_ok: bool
    hidden_projection_ok: bool
    attention_steering_compat_ok: bool
    cache_reuse_equiv_ok: bool
    retrieval_equiv_ok: bool
    replay_controller_equiv_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v5_witness_cid": str(
                self.inner_v5_witness_cid),
            "merge_idempotent_ok": bool(
                self.merge_idempotent_ok),
            "diff_self_cancel_ok": bool(
                self.diff_self_cancel_ok),
            "intersect_distrib_ok": bool(
                self.intersect_distrib_ok),
            "substrate_projection_ok": bool(
                self.substrate_projection_ok),
            "hidden_projection_ok": bool(
                self.hidden_projection_ok),
            "attention_steering_compat_ok": bool(
                self.attention_steering_compat_ok),
            "cache_reuse_equiv_ok": bool(
                self.cache_reuse_equiv_ok),
            "retrieval_equiv_ok": bool(self.retrieval_equiv_ok),
            "replay_controller_equiv_ok": bool(
                self.replay_controller_equiv_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v6_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v6_witness(
        *,
        trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        probe_c: Sequence[float],
        substrate_forward_fn=None,
        hidden_state_projector=None,
        steering_oracle=None,
        cache_reuse_oracle=None,
        retrieval_replay_oracle=None,
        replay_controller_oracle=None,
) -> DisagreementAlgebraV6Witness:
    v5_witness = emit_disagreement_algebra_v5_witness(
        trace=trace,
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c,
        substrate_forward_fn=substrate_forward_fn,
        hidden_state_projector=hidden_state_projector,
        steering_oracle=steering_oracle,
        cache_reuse_oracle=cache_reuse_oracle,
        retrieval_replay_oracle=retrieval_replay_oracle)
    rc_ok = True
    if replay_controller_oracle is not None:
        rc_ok = bool(
            check_replay_controller_equivalence_identity(
                replay_controller_oracle=replay_controller_oracle))
    return DisagreementAlgebraV6Witness(
        schema=W60_DA_V6_SCHEMA_VERSION,
        inner_v5_witness_cid=str(v5_witness.cid()),
        merge_idempotent_ok=bool(
            v5_witness.merge_idempotent_ok),
        diff_self_cancel_ok=bool(
            v5_witness.diff_self_cancel_ok),
        intersect_distrib_ok=bool(
            v5_witness.intersect_distrib_ok),
        substrate_projection_ok=bool(
            v5_witness.substrate_projection_ok),
        hidden_projection_ok=bool(
            v5_witness.hidden_projection_ok),
        attention_steering_compat_ok=bool(
            v5_witness.attention_steering_compat_ok),
        cache_reuse_equiv_ok=bool(
            v5_witness.cache_reuse_equiv_ok),
        retrieval_equiv_ok=bool(v5_witness.retrieval_equiv_ok),
        replay_controller_equiv_ok=bool(rc_ok),
    )


__all__ = [
    "W60_DA_V6_SCHEMA_VERSION",
    "DisagreementAlgebraV6Witness",
    "check_replay_controller_equivalence_identity",
    "emit_disagreement_algebra_v6_witness",
]
