"""W59 M10 — Consensus Fallback Controller V5.

Strictly extends W58's ``coordpy.consensus_fallback_controller_v4``
with a **9-stage decision chain**:

  1. K-of-N quorum
  2. trust-weighted quorum
  3. substrate-conditioned tiebreaker
  4. logit-lens-conditioned tiebreaker
  5. cache-reuse-replay (from V4)
  6. **retrieval-replay** (NEW — picks the parent whose payload
     matches the cache controller V2 retrieval score on the
     stored cache fingerprint)
  7. best-parent
  8. transcript fallback
  9. abstain

The new retrieval-replay stage runs *after* the cache-reuse stage:
cache-reuse covers "we know which run literally produced this
payload"; retrieval-replay covers "we know which parent's
controller-V2 retrieval score on this query produced this
payload". The latter is strictly weaker (it doesn't require
byte-identical cache contents, only retrieval-score agreement)
and is the appropriate fall-through when cache-reuse fails.

V5 strictly extends V4: when ``retrieval_replay_oracle = None``,
V5 reduces to V4 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence


W59_CONSENSUS_V5_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v5.v1")

W59_CONSENSUS_V5_STAGE_K_OF_N: str = "k_of_n"
W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED: str = "trust_weighted"
W59_CONSENSUS_V5_STAGE_SUBSTRATE: str = "substrate_conditioned"
W59_CONSENSUS_V5_STAGE_LOGIT_LENS: str = "logit_lens_conditioned"
W59_CONSENSUS_V5_STAGE_CACHE_REUSE: str = "cache_reuse_replay"
W59_CONSENSUS_V5_STAGE_RETRIEVAL: str = "retrieval_replay"
W59_CONSENSUS_V5_STAGE_BEST_PARENT: str = "best_parent"
W59_CONSENSUS_V5_STAGE_TRANSCRIPT: str = "transcript"
W59_CONSENSUS_V5_STAGE_ABSTAIN: str = "abstain"


W59_CONSENSUS_V5_STAGES: tuple[str, ...] = (
    W59_CONSENSUS_V5_STAGE_K_OF_N,
    W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED,
    W59_CONSENSUS_V5_STAGE_SUBSTRATE,
    W59_CONSENSUS_V5_STAGE_LOGIT_LENS,
    W59_CONSENSUS_V5_STAGE_CACHE_REUSE,
    W59_CONSENSUS_V5_STAGE_RETRIEVAL,
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


@dataclasses.dataclass
class ConsensusFallbackControllerV5:
    k_required: int = 2
    cosine_floor: float = 0.6
    trust_threshold: float = 0.5
    substrate_oracle: (
        Callable[[Sequence[Sequence[float]],
                  Sequence[float]],
                 int] | None) = None
    logit_lens_oracle: (
        Callable[[Sequence[Sequence[float]],
                  Sequence[float]],
                 int] | None) = None
    cache_reuse_oracle: (
        Callable[[Sequence[Sequence[float]],
                  Sequence[float],
                  Sequence[tuple[int, ...]]],
                 int] | None) = None
    retrieval_replay_oracle: (
        Callable[[Sequence[Sequence[float]],
                  Sequence[float],
                  Sequence[float]],
                 int] | None) = None
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W59_CONSENSUS_V5_SCHEMA_VERSION,
            "kind": "consensus_v5_controller",
            "k_required": int(self.k_required),
            "cosine_floor": float(round(self.cosine_floor, 12)),
            "trust_threshold": float(round(
                self.trust_threshold, 12)),
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
        cos = [_cosine(p, query_direction)
                for p in parent_payloads]
        # Stage 1 — K-of-N.
        agree = [i for i, c in enumerate(cos)
                  if c >= float(self.cosine_floor)]
        if len(agree) >= int(self.k_required):
            self.audit.append({
                "stage": W59_CONSENSUS_V5_STAGE_K_OF_N,
                "agree": list(agree),
                "cos": [float(round(c, 9)) for c in cos],
            })
            idx = max(agree, key=lambda i: cos[i])
            return {
                "decision_stage": W59_CONSENSUS_V5_STAGE_K_OF_N,
                "selected_index": int(idx),
                "payload": list(parent_payloads[idx]),
            }
        # Stage 2 — trust-weighted quorum.
        tot_trust = sum(float(t) for t, c
                         in zip(parent_trusts, cos)
                         if c >= float(self.cosine_floor))
        if tot_trust >= float(self.trust_threshold):
            self.audit.append({
                "stage": W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED,
                "tot_trust": float(round(tot_trust, 9)),
            })
            cand = [i for i, c in enumerate(cos)
                     if c >= float(self.cosine_floor)]
            idx = max(cand,
                       key=lambda i: cos[i] * parent_trusts[i])
            return {
                "decision_stage":
                    W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED,
                "selected_index": int(idx),
                "payload": list(parent_payloads[idx]),
            }
        # Stage 3 — substrate-conditioned tiebreaker.
        if self.substrate_oracle is not None:
            try:
                idx = int(self.substrate_oracle(
                    parent_payloads, query_direction))
                if 0 <= idx < n:
                    self.audit.append({
                        "stage":
                            W59_CONSENSUS_V5_STAGE_SUBSTRATE,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W59_CONSENSUS_V5_STAGE_SUBSTRATE,
                        "selected_index": int(idx),
                        "payload": list(parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stage 4 — logit-lens-conditioned tiebreaker.
        if self.logit_lens_oracle is not None:
            try:
                idx = int(self.logit_lens_oracle(
                    parent_payloads, query_direction))
                if 0 <= idx < n:
                    self.audit.append({
                        "stage":
                            W59_CONSENSUS_V5_STAGE_LOGIT_LENS,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W59_CONSENSUS_V5_STAGE_LOGIT_LENS,
                        "selected_index": int(idx),
                        "payload": list(parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stage 5 — cache-reuse-replay tiebreaker.
        if (self.cache_reuse_oracle is not None
                and parent_cache_fingerprints is not None):
            try:
                idx = int(self.cache_reuse_oracle(
                    parent_payloads, query_direction,
                    parent_cache_fingerprints))
                if 0 <= idx < n:
                    self.audit.append({
                        "stage":
                            W59_CONSENSUS_V5_STAGE_CACHE_REUSE,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W59_CONSENSUS_V5_STAGE_CACHE_REUSE,
                        "selected_index": int(idx),
                        "payload": list(parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stage 6 — retrieval-replay tiebreaker (NEW).
        if (self.retrieval_replay_oracle is not None
                and parent_retrieval_scores is not None):
            try:
                idx = int(self.retrieval_replay_oracle(
                    parent_payloads, query_direction,
                    parent_retrieval_scores))
                if 0 <= idx < n:
                    self.audit.append({
                        "stage":
                            W59_CONSENSUS_V5_STAGE_RETRIEVAL,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W59_CONSENSUS_V5_STAGE_RETRIEVAL,
                        "selected_index": int(idx),
                        "payload": list(parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stage 7 — best-parent.
        best = max(range(n),
                    key=lambda i: parent_trusts[i] * cos[i])
        if cos[best] >= float(self.cosine_floor):
            self.audit.append({
                "stage":
                    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
                "selected_index": int(best),
            })
            return {
                "decision_stage":
                    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
                "selected_index": int(best),
                "payload": list(parent_payloads[best]),
            }
        # Stage 8 — transcript fallback.
        if list(transcript_payload):
            self.audit.append({
                "stage":
                    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            })
            return {
                "decision_stage":
                    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
                "selected_index": -1,
                "payload": list(transcript_payload),
            }
        # Stage 9 — abstain.
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
class ConsensusFallbackControllerV5Witness:
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
            "kind": "consensus_v5_witness",
            "witness": self.to_dict()})


def emit_consensus_v5_witness(
        controller: ConsensusFallbackControllerV5,
) -> ConsensusFallbackControllerV5Witness:
    last_stage = (
        str(controller.audit[-1]["stage"])
        if controller.audit
        else W59_CONSENSUS_V5_STAGE_ABSTAIN)
    return ConsensusFallbackControllerV5Witness(
        schema=W59_CONSENSUS_V5_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_audit_entries=int(len(controller.audit)),
        last_stage=str(last_stage),
        n_stages_in_chain=int(len(W59_CONSENSUS_V5_STAGES)),
    )


__all__ = [
    "W59_CONSENSUS_V5_SCHEMA_VERSION",
    "W59_CONSENSUS_V5_STAGE_K_OF_N",
    "W59_CONSENSUS_V5_STAGE_TRUST_WEIGHTED",
    "W59_CONSENSUS_V5_STAGE_SUBSTRATE",
    "W59_CONSENSUS_V5_STAGE_LOGIT_LENS",
    "W59_CONSENSUS_V5_STAGE_CACHE_REUSE",
    "W59_CONSENSUS_V5_STAGE_RETRIEVAL",
    "W59_CONSENSUS_V5_STAGE_BEST_PARENT",
    "W59_CONSENSUS_V5_STAGE_TRANSCRIPT",
    "W59_CONSENSUS_V5_STAGE_ABSTAIN",
    "W59_CONSENSUS_V5_STAGES",
    "ConsensusFallbackControllerV5",
    "ConsensusFallbackControllerV5Witness",
    "emit_consensus_v5_witness",
]
