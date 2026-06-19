"""W59 M16 — Disagreement Algebra V5.

Strictly extends W58's ``coordpy.disagreement_algebra_v4``. V5
adds one new identity:

* **retrieval-equivalence identity** — replaying a payload via
  the cache-controller-V2 retrieval head on a stored cache
  fingerprint produces a logit pattern that matches the original
  reference up to the controller-V2's *advertised tolerance*. The
  identity checks ``argmax_preserved`` and
  ``last_logit_l2_drift ≤ tol`` for a tolerance budget the
  identity oracle reports.

V5 strictly extends V4: when ``retrieval_replay_oracle = None``,
V5 reduces to V4 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .disagreement_algebra import (
    AlgebraTrace,
    check_difference_self_cancellation,
    check_intersection_distributivity_on_agreement,
    check_merge_idempotent,
)


W59_DA_V5_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v5.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def check_retrieval_equivalence_identity(
        *, retrieval_replay_oracle,
        l2_tolerance: float = 1e-6,
) -> bool:
    """Retrieval-replay oracle returns
    ``(argmax_preserved_bool, last_l2_drift_float)``. Identity
    holds iff argmax_preserved AND l2_drift ≤ l2_tolerance.
    """
    try:
        argmax_ok, l2_drift = retrieval_replay_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(l2_drift) <= float(l2_tolerance))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV5Witness:
    schema: str
    inner_trace_cid: str
    merge_idempotent_ok: bool
    diff_self_cancel_ok: bool
    intersect_distrib_ok: bool
    substrate_projection_ok: bool
    hidden_projection_ok: bool
    attention_steering_compat_ok: bool
    cache_reuse_equiv_ok: bool
    retrieval_equiv_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_trace_cid": str(self.inner_trace_cid),
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
            "retrieval_equiv_ok": bool(
                self.retrieval_equiv_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v5_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v5_witness(
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
) -> DisagreementAlgebraV5Witness:
    from .disagreement_algebra_v4 import (
        check_cache_reuse_equivalence_identity,
    )
    merge_res = check_merge_idempotent(probe_a)
    diff_res = check_difference_self_cancellation(probe_a)
    inter_res = check_intersection_distributivity_on_agreement(
        probe_a, probe_b, probe_c)
    merge_ok = bool(merge_res.ok)
    diff_ok = bool(diff_res.ok)
    inter_ok = bool(inter_res.ok)
    sub_ok = True
    if substrate_forward_fn is not None:
        from .disagreement_algebra_v2 import (
            check_substrate_projection_identity,
        )
        try:
            sub_ok = bool(check_substrate_projection_identity(
                probe_a, probe_b,
                substrate_forward_fn=substrate_forward_fn))
        except Exception:
            sub_ok = False
    hid_ok = True
    if hidden_state_projector is not None:
        from .disagreement_algebra_v3 import (
            check_hidden_state_projection_identity,
        )
        hid_ok = bool(check_hidden_state_projection_identity(
            probe_a, probe_b,
            hidden_state_projector=hidden_state_projector))
    steer_ok = True
    if steering_oracle is not None:
        from .disagreement_algebra_v3 import (
            check_attention_steering_compatibility,
        )
        steer_ok = bool(check_attention_steering_compatibility(
            probe_a, steering_oracle=steering_oracle))
    cache_ok = True
    if cache_reuse_oracle is not None:
        cache_ok = bool(
            check_cache_reuse_equivalence_identity(
                cache_reuse_oracle=cache_reuse_oracle))
    ret_ok = True
    if retrieval_replay_oracle is not None:
        ret_ok = bool(
            check_retrieval_equivalence_identity(
                retrieval_replay_oracle=retrieval_replay_oracle))
    return DisagreementAlgebraV5Witness(
        schema=W59_DA_V5_SCHEMA_VERSION,
        inner_trace_cid=str(trace.cid()),
        merge_idempotent_ok=bool(merge_ok),
        diff_self_cancel_ok=bool(diff_ok),
        intersect_distrib_ok=bool(inter_ok),
        substrate_projection_ok=bool(sub_ok),
        hidden_projection_ok=bool(hid_ok),
        attention_steering_compat_ok=bool(steer_ok),
        cache_reuse_equiv_ok=bool(cache_ok),
        retrieval_equiv_ok=bool(ret_ok),
    )


__all__ = [
    "W59_DA_V5_SCHEMA_VERSION",
    "DisagreementAlgebraV5Witness",
    "check_retrieval_equivalence_identity",
    "emit_disagreement_algebra_v5_witness",
]
