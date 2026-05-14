"""W58 M17 — Disagreement Algebra V4.

Strictly extends W57's ``coordpy.disagreement_algebra_v3``. V4
adds one new identity:

* **cache-reuse-equivalence identity** — replaying a payload
  through prefix-state reuse with a matching cache fingerprint
  produces an *identical* logit pattern (within float64
  precision) to the corresponding full recompute. The identity
  checks ``max_abs(diff) < 1e-9`` between
  ``forward(reuse_path)`` and ``forward(full_recompute)`` over a
  shared follow-up sequence.

V4 strictly extends V3: when ``cache_reuse_oracle = None``, V4
reduces to V3 byte-for-byte.
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


W58_DA_V4_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v4.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def check_cache_reuse_equivalence_identity(
        *, cache_reuse_oracle,
        threshold: float = 1e-9,
) -> bool:
    """Cache-reuse oracle returns ``(max_abs_diff, matches_flag)``.
    Identity holds iff ``matches_flag`` is True or
    ``max_abs_diff < threshold``.
    """
    try:
        max_abs, matches = cache_reuse_oracle()
    except Exception:
        return False
    if bool(matches):
        return True
    return bool(float(max_abs) < float(threshold))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV4Witness:
    schema: str
    inner_trace_cid: str
    merge_idempotent_ok: bool
    diff_self_cancel_ok: bool
    intersect_distrib_ok: bool
    substrate_projection_ok: bool
    hidden_projection_ok: bool
    attention_steering_compat_ok: bool
    cache_reuse_equiv_ok: bool

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v4_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v4_witness(
        *,
        trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        probe_c: Sequence[float],
        substrate_forward_fn=None,
        hidden_state_projector=None,
        steering_oracle=None,
        cache_reuse_oracle=None,
) -> DisagreementAlgebraV4Witness:
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
    return DisagreementAlgebraV4Witness(
        schema=W58_DA_V4_SCHEMA_VERSION,
        inner_trace_cid=str(trace.cid()),
        merge_idempotent_ok=bool(merge_ok),
        diff_self_cancel_ok=bool(diff_ok),
        intersect_distrib_ok=bool(inter_ok),
        substrate_projection_ok=bool(sub_ok),
        hidden_projection_ok=bool(hid_ok),
        attention_steering_compat_ok=bool(steer_ok),
        cache_reuse_equiv_ok=bool(cache_ok),
    )


__all__ = [
    "W58_DA_V4_SCHEMA_VERSION",
    "DisagreementAlgebraV4Witness",
    "check_cache_reuse_equivalence_identity",
    "emit_disagreement_algebra_v4_witness",
]
