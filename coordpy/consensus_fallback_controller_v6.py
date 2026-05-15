"""W60 M11 — Consensus Fallback Controller V6.

Strictly extends W59's ``coordpy.consensus_fallback_controller_v5``
with a **10-stage decision chain**:

  1. K-of-N quorum
  2. trust-weighted quorum
  3. substrate-conditioned tiebreaker
  4. logit-lens-conditioned tiebreaker
  5. cache-reuse-replay
  6. retrieval-replay (V5)
  7. **replay-controller-conditioned** tiebreaker (NEW — picks the
     parent whose ReplayController decision aligns with the
     substrate's CRC + flop-saving signal)
  8. best-parent
  9. transcript fallback
  10. abstain

The new replay-controller stage runs *after* retrieval-replay:
retrieval-replay covers "we know which parent's controller-V2
retrieval score on this query produced this payload"; replay-
controller covers "we know which parent's W60 ReplayController
chose REUSE successfully (CRC passed and drift below ceiling) on
this exact carrier".

V6 strictly extends V5: when ``replay_controller_oracle = None``,
V6 reduces to V5 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence

from .consensus_fallback_controller_v5 import (
    ConsensusFallbackControllerV5,
    W59_CONSENSUS_V5_STAGES,
    W59_CONSENSUS_V5_STAGE_K_OF_N,
    W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED,
    W59_CONSENSUS_V5_STAGE_SUBSTRATE,
    W59_CONSENSUS_V5_STAGE_LOGIT_LENS,
    W59_CONSENSUS_V5_STAGE_CACHE_REUSE,
    W59_CONSENSUS_V5_STAGE_RETRIEVAL,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    _cosine,
)


W60_CONSENSUS_V6_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v6.v1")
W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER: str = (
    "replay_controller_conditioned")

W60_CONSENSUS_V6_STAGES: tuple[str, ...] = (
    W59_CONSENSUS_V5_STAGE_K_OF_N,
    W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED,
    W59_CONSENSUS_V5_STAGE_SUBSTRATE,
    W59_CONSENSUS_V5_STAGE_LOGIT_LENS,
    W59_CONSENSUS_V5_STAGE_CACHE_REUSE,
    W59_CONSENSUS_V5_STAGE_RETRIEVAL,
    W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER,
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
class ConsensusFallbackControllerV6:
    inner_v5: ConsensusFallbackControllerV5
    replay_controller_oracle: (
        Callable[[Sequence[Sequence[float]],
                  Sequence[float],
                  Sequence[str]],
                 int] | None) = None
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            k_required: int = 2,
            cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
    ) -> "ConsensusFallbackControllerV6":
        inner = ConsensusFallbackControllerV5(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold))
        return cls(inner_v5=inner)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W60_CONSENSUS_V6_SCHEMA_VERSION,
            "kind": "consensus_v6_controller",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "audit": list(self.audit),
        })

    def decide(
            self, *,
            parent_payloads: Sequence[Sequence[float]],
            parent_trusts: Sequence[float],
            parent_cache_fingerprints: (
                Sequence[tuple[int, ...]] | None) = None,
            parent_retrieval_scores: (
                Sequence[float] | None) = None,
            parent_replay_decisions: (
                Sequence[str] | None) = None,
            query_direction: Sequence[float],
            transcript_payload: Sequence[float],
    ) -> dict[str, Any]:
        n = len(parent_payloads)
        if n == 0:
            self.audit.append(
                {"stage": W59_CONSENSUS_V5_STAGE_ABSTAIN,
                 "reason": "no_parents"})
            return {
                "decision_stage": W59_CONSENSUS_V5_STAGE_ABSTAIN,
                "selected_index": -1,
                "payload": [0.0] * len(query_direction),
            }
        # Try V5 stages 1-6 first via the inner V5 controller.
        # We replicate stages 1-6 of V5 inline to keep audit
        # ordering coherent.
        cos = [_cosine(p, query_direction)
                for p in parent_payloads]
        agree = [i for i, c in enumerate(cos)
                  if c >= float(self.inner_v5.cosine_floor)]
        # Stage 1.
        if len(agree) >= int(self.inner_v5.k_required):
            idx = max(agree, key=lambda i: cos[i])
            self.audit.append({
                "stage": W59_CONSENSUS_V5_STAGE_K_OF_N,
                "selected_index": int(idx),
            })
            return {
                "decision_stage": W59_CONSENSUS_V5_STAGE_K_OF_N,
                "selected_index": int(idx),
                "payload": list(parent_payloads[idx]),
            }
        # Stage 2.
        tot = sum(float(t) for t, c
                    in zip(parent_trusts, cos)
                    if c >= float(self.inner_v5.cosine_floor))
        if tot >= float(self.inner_v5.trust_threshold):
            cand = [i for i, c in enumerate(cos)
                     if c >= float(self.inner_v5.cosine_floor)]
            idx = max(cand, key=lambda i:
                        cos[i] * parent_trusts[i])
            self.audit.append({
                "stage": W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED,
                "selected_index": int(idx),
            })
            return {
                "decision_stage": (
                    W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED),
                "selected_index": int(idx),
                "payload": list(parent_payloads[idx]),
            }
        # Stages 3-6: defer to inner V5.
        v5_res = self.inner_v5.decide(
            parent_payloads=parent_payloads,
            parent_trusts=parent_trusts,
            parent_cache_fingerprints=parent_cache_fingerprints,
            parent_retrieval_scores=parent_retrieval_scores,
            query_direction=query_direction,
            transcript_payload=[])
        if v5_res["decision_stage"] not in (
                W59_CONSENSUS_V5_STAGE_BEST_PARENT,
                W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
                W59_CONSENSUS_V5_STAGE_ABSTAIN,
        ):
            self.audit.append({
                "stage": v5_res["decision_stage"],
                "selected_index": int(
                    v5_res["selected_index"]),
            })
            return v5_res
        # Stage 7 — replay-controller-conditioned tiebreaker.
        if (self.replay_controller_oracle is not None
                and parent_replay_decisions is not None):
            try:
                idx = int(self.replay_controller_oracle(
                    parent_payloads, query_direction,
                    parent_replay_decisions))
                if 0 <= idx < n:
                    self.audit.append({
                        "stage": (
                            W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER),
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage": (
                            W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER),
                        "selected_index": int(idx),
                        "payload": list(
                            parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stages 8-10: best-parent / transcript / abstain.
        best = max(range(n),
                    key=lambda i: parent_trusts[i] * cos[i])
        if cos[best] >= float(self.inner_v5.cosine_floor):
            self.audit.append({
                "stage": W59_CONSENSUS_V5_STAGE_BEST_PARENT,
                "selected_index": int(best),
            })
            return {
                "decision_stage": (
                    W59_CONSENSUS_V5_STAGE_BEST_PARENT),
                "selected_index": int(best),
                "payload": list(parent_payloads[best]),
            }
        if list(transcript_payload):
            self.audit.append({
                "stage": (
                    W59_CONSENSUS_V5_STAGE_TRANSCRIPT),
            })
            return {
                "decision_stage": (
                    W59_CONSENSUS_V5_STAGE_TRANSCRIPT),
                "selected_index": -1,
                "payload": list(transcript_payload),
            }
        self.audit.append({
            "stage": W59_CONSENSUS_V5_STAGE_ABSTAIN,
            "reason": "all_paths_below_floor",
        })
        return {
            "decision_stage": W59_CONSENSUS_V5_STAGE_ABSTAIN,
            "selected_index": -1,
            "payload": [0.0] * len(query_direction),
        }


@dataclasses.dataclass(frozen=True)
class ConsensusFallbackControllerV6Witness:
    schema: str
    controller_cid: str
    n_audit_entries: int
    last_stage: str
    n_stages_in_chain: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_audit_entries": int(self.n_audit_entries),
            "last_stage": str(self.last_stage),
            "n_stages_in_chain": int(self.n_stages_in_chain),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v6_witness",
            "witness": self.to_dict()})


def emit_consensus_v6_witness(
        controller: ConsensusFallbackControllerV6,
) -> ConsensusFallbackControllerV6Witness:
    last_stage = (
        str(controller.audit[-1]["stage"])
        if controller.audit
        else W59_CONSENSUS_V5_STAGE_ABSTAIN)
    return ConsensusFallbackControllerV6Witness(
        schema=W60_CONSENSUS_V6_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_audit_entries=int(len(controller.audit)),
        last_stage=str(last_stage),
        n_stages_in_chain=int(len(W60_CONSENSUS_V6_STAGES)),
    )


__all__ = [
    "W60_CONSENSUS_V6_SCHEMA_VERSION",
    "W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER",
    "W60_CONSENSUS_V6_STAGES",
    "ConsensusFallbackControllerV6",
    "ConsensusFallbackControllerV6Witness",
    "emit_consensus_v6_witness",
]
