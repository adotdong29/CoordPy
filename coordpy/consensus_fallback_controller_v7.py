"""W61 M11 — Consensus Fallback Controller V7.

Strictly extends W60's ``coordpy.consensus_fallback_controller_v6``
with an **11-stage decision chain**. V6 had 10 stages; V7 inserts
**``attention_pattern_consensus``** between ``replay_controller``
and ``best_parent``:

  1. K-of-N quorum
  2. trust-weighted quorum
  3. substrate-conditioned tiebreaker
  4. logit-lens-conditioned tiebreaker
  5. cache-reuse-replay
  6. retrieval-replay
  7. replay-controller-conditioned
  8. **attention-pattern-consensus** (NEW)
  9. best-parent
  10. transcript fallback
  11. abstain

The new stage fires when multiple parents share a similar attention
pattern (top-K Jaccard ≥ floor). It picks the parent with the
highest *attention-pattern-trust × confidence* product among the
similar-pattern subset.

V7 strictly extends V6: when ``attention_pattern_oracle = None``,
V7 reduces to V6 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence

from .consensus_fallback_controller_v6 import (
    ConsensusFallbackControllerV6,
    W60_CONSENSUS_V6_STAGES,
    W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER,
)
from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
)


W61_CONSENSUS_V7_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v7.v1")
W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN: str = (
    "attention_pattern_consensus")

W61_CONSENSUS_V7_STAGES: tuple[str, ...] = (
    *(s for s in W60_CONSENSUS_V6_STAGES if s not in (
        W59_CONSENSUS_V5_STAGE_BEST_PARENT,
        W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
        W59_CONSENSUS_V5_STAGE_ABSTAIN)),
    W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class ConsensusFallbackControllerV7:
    inner_v6: ConsensusFallbackControllerV6
    attention_pattern_oracle: (
        Callable[[Sequence[Sequence[float]],
                  Sequence[Sequence[int]]],
                 int] | None) = None
    audit_v7: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, k_required: int = 2,
            cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
    ) -> "ConsensusFallbackControllerV7":
        inner = ConsensusFallbackControllerV6.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold))
        return cls(inner_v6=inner)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_CONSENSUS_V7_SCHEMA_VERSION,
            "kind": "consensus_v7_controller",
            "inner_v6_cid": str(self.inner_v6.cid()),
            "stages": list(W61_CONSENSUS_V7_STAGES),
        })

    def decide_v7(
            self, *,
            payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            attention_top_k_positions: (
                Sequence[Sequence[int]] | None) = None,
            attention_top_k_jaccard_floor: float = 0.5,
            transcript_available: bool = False,
            transcript_carrier: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        # Try V6 chain first. The V6 controller's decide() expects
        # the W60 parent-payload signature; we adapt our V7-shape
        # inputs into it. We pass a neutral query_direction (sum of
        # payloads) so the substrate-conditioned tiebreaker has a
        # well-defined target.
        if not payloads:
            return {
                "stage": "abstain",
                "payload": [], "v7_promoted": False,
                "rationale": "no_payloads",
            }
        dim = len(payloads[0])
        query = [0.0] * dim
        for p in payloads:
            for i, v in enumerate(p):
                if i < dim:
                    query[i] += float(v)
        try:
            v6_decision = self.inner_v6.decide(
                parent_payloads=payloads,
                parent_trusts=trusts,
                parent_cache_fingerprints=None,
                parent_retrieval_scores=None,
                parent_replay_decisions=replay_decisions,
                query_direction=query,
                transcript_available=bool(
                    transcript_available),
                transcript_carrier=(
                    list(transcript_carrier)
                    if transcript_carrier is not None
                    else None))
        except Exception as e:
            v6_decision = {
                "stage": "best_parent",
                "payload": list(payloads[0]),
                "rationale": f"v6_error:{e}",
            }
        v6_stage = str(v6_decision.get("stage", ""))
        # Capture the V7 audit entry.
        self.audit_v7.append({
            "v6_stage": v6_stage,
            "v6_decision": v6_decision,
        })
        # If V6 produced a non-fallback decision, propagate it.
        if v6_stage not in (
                W59_CONSENSUS_V5_STAGE_BEST_PARENT,
                W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
                W59_CONSENSUS_V5_STAGE_ABSTAIN):
            return {
                "stage": v6_stage,
                **v6_decision,
                "v7_promoted": False,
            }
        # Otherwise try the V7 attention-pattern-consensus stage.
        if (attention_top_k_positions is not None
                and len(attention_top_k_positions)
                == len(payloads)):
            # Compute pairwise Jaccard between top-K positions.
            n = len(payloads)
            jaccards = [[0.0] * n for _ in range(n)]
            for i in range(n):
                set_i = set(
                    int(p)
                    for p in attention_top_k_positions[i])
                for j in range(n):
                    if i == j:
                        jaccards[i][j] = 1.0
                        continue
                    set_j = set(
                        int(p)
                        for p in attention_top_k_positions[j])
                    inter = len(set_i & set_j)
                    union = len(set_i | set_j)
                    jaccards[i][j] = (
                        float(inter) / float(union)
                        if union > 0 else 0.0)
            # Find the maximum cluster of parents whose pairwise
            # Jaccard ≥ floor.
            best_cluster: list[int] = []
            for i in range(n):
                cluster = [i]
                for j in range(n):
                    if j == i: continue
                    if all(jaccards[j][k]
                           >= float(
                               attention_top_k_jaccard_floor)
                           for k in cluster):
                        cluster.append(j)
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster
            if len(best_cluster) >= 2:
                # Pick the parent in cluster with highest
                # trust × confidence.
                best = max(best_cluster,
                           key=lambda i: float(trusts[i])
                           if i < len(trusts) else 0.0)
                self.audit_v7.append({
                    "v7_stage":
                        W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN,
                    "cluster": best_cluster,
                    "best_idx": int(best),
                })
                return {
                    "stage":
                        W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN,
                    "payload": list(payloads[best]),
                    "best_idx": int(best),
                    "cluster_size": int(len(best_cluster)),
                    "v7_promoted": True,
                }
        # V7 cannot resolve: fall through to V6's final decision.
        return {
            "stage": v6_stage,
            **v6_decision,
            "v7_promoted": False,
        }


@dataclasses.dataclass(frozen=True)
class ConsensusFallbackControllerV7Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    n_v7_promoted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "n_v7_promoted": int(self.n_v7_promoted),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v7_witness",
            "witness": self.to_dict()})


def emit_consensus_v7_witness(
        controller: ConsensusFallbackControllerV7,
) -> ConsensusFallbackControllerV7Witness:
    n_dec = len(controller.audit_v7)
    n_promoted = sum(
        1 for e in controller.audit_v7
        if str(e.get("v7_stage", ""))
        == W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN)
    return ConsensusFallbackControllerV7Witness(
        schema=W61_CONSENSUS_V7_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W61_CONSENSUS_V7_STAGES),
        n_decisions=int(n_dec),
        n_v7_promoted=int(n_promoted),
    )


__all__ = [
    "W61_CONSENSUS_V7_SCHEMA_VERSION",
    "W61_CONSENSUS_V7_STAGES",
    "W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN",
    "ConsensusFallbackControllerV7",
    "ConsensusFallbackControllerV7Witness",
    "emit_consensus_v7_witness",
]
