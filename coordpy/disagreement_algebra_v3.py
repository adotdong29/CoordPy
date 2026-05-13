"""W57 M15 — Disagreement Algebra V3.

Extends W56 V2 with two new identities:

* **hidden-projection identity** — projecting a payload through
  the substrate's *intermediate-layer* hidden state (via the
  hidden-state bridge) and then taking the cosine to the original
  payload should remain ≥ 0.3 on in-distribution probes.
* **attention-steering compatibility** — steering attention
  toward a payload via the attention-steering bridge should
  produce an attention shift (mean KL > 0) without flipping the
  causal mask sign — i.e., the steering changes *which keys*
  receive weight but not *the temporal order*.

V3 strictly extends V2: all V1+V2 identities still hold. New
identities are conditioned on the W57 hidden-state bridge and
attention-steering bridge being available.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .disagreement_algebra import (
    AlgebraTrace,
    check_difference_self_cancellation,
    check_intersection_distributivity_on_agreement,
    check_merge_idempotent,
)


W57_DA_V3_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v3.v1")


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


def check_hidden_state_projection_identity(
        a: Sequence[float],
        b: Sequence[float],
        *, hidden_state_projector,
        threshold: float = 0.3,
) -> bool:
    """Project ``avg(a, b)`` through the hidden-state bridge,
    compare cosine to the original average; must be ≥ threshold.
    """
    n = max(len(a), len(b))
    avg = [
        0.5 * (float(a[i] if i < len(a) else 0.0)
               + float(b[i] if i < len(b) else 0.0))
        for i in range(int(n))
    ]
    try:
        proj = list(hidden_state_projector(avg))
    except Exception:
        return False
    return bool(_cosine(proj, avg) >= float(threshold))


def check_attention_steering_compatibility(
        carrier: Sequence[float],
        *, steering_oracle,
) -> bool:
    """Steering oracle returns ``(mean_kl, max_shift,
    causal_ok)``. Compatibility: mean_kl > 0 and causal_ok.
    """
    try:
        kl, shift, causal = steering_oracle(carrier)
    except Exception:
        return False
    return bool(float(kl) > 0.0 and bool(causal))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV3Witness:
    schema: str
    inner_trace_cid: str
    merge_idempotent_ok: bool
    diff_self_cancel_ok: bool
    intersect_distrib_ok: bool
    substrate_projection_ok: bool
    hidden_projection_ok: bool
    attention_steering_compat_ok: bool

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v3_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v3_witness(
        *,
        trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        probe_c: Sequence[float],
        substrate_forward_fn=None,
        hidden_state_projector=None,
        steering_oracle=None,
) -> DisagreementAlgebraV3Witness:
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
        hid_ok = bool(check_hidden_state_projection_identity(
            probe_a, probe_b,
            hidden_state_projector=hidden_state_projector))
    steer_ok = True
    if steering_oracle is not None:
        steer_ok = bool(check_attention_steering_compatibility(
            probe_a, steering_oracle=steering_oracle))
    return DisagreementAlgebraV3Witness(
        schema=W57_DA_V3_SCHEMA_VERSION,
        inner_trace_cid=str(trace.cid()),
        merge_idempotent_ok=bool(merge_ok),
        diff_self_cancel_ok=bool(diff_ok),
        intersect_distrib_ok=bool(inter_ok),
        substrate_projection_ok=bool(sub_ok),
        hidden_projection_ok=bool(hid_ok),
        attention_steering_compat_ok=bool(steer_ok),
    )


__all__ = [
    "W57_DA_V3_SCHEMA_VERSION",
    "DisagreementAlgebraV3Witness",
    "check_hidden_state_projection_identity",
    "check_attention_steering_compatibility",
    "emit_disagreement_algebra_v3_witness",
]
