"""W58 M11 — Consensus Fallback Controller V4.

Strictly extends W57's ``coordpy.consensus_fallback_controller_v3``
with an **8-stage decision chain**:

  1. K-of-N quorum
  2. trust-weighted quorum
  3. substrate-conditioned tiebreaker
  4. logit-lens-conditioned tiebreaker
  5. **cache-reuse-replay** (NEW — picks the parent whose payload
       was reproduced under prefix-state reuse with a matching
       cache fingerprint)
  6. best-parent
  7. transcript fallback
  8. abstain

The cache-reuse-replay stage takes a separate oracle. The oracle
receives ``(parent_payloads, query_direction, parent_cache_fingerprints)``
and returns the index of the parent whose cache fingerprint
matches the expected one. This is the substrate-level "we know
which run *literally* produced this payload" tiebreaker.

V4 strictly extends V3: when ``cache_reuse_oracle = None``, V4
reduces to V3 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence


W58_CONSENSUS_V4_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v4.v1")

W58_CONSENSUS_V4_STAGE_K_OF_N: str = "k_of_n"
W58_CONSENSUS_V4_STAGE_TRUST_WEIGHTED: str = "trust_weighted"
W58_CONSENSUS_V4_STAGE_SUBSTRATE: str = "substrate_conditioned"
W58_CONSENSUS_V4_STAGE_LOGIT_LENS: str = "logit_lens_conditioned"
W58_CONSENSUS_V4_STAGE_CACHE_REUSE: str = "cache_reuse_replay"
W58_CONSENSUS_V4_STAGE_BEST_PARENT: str = "best_parent"
W58_CONSENSUS_V4_STAGE_TRANSCRIPT: str = "transcript"
W58_CONSENSUS_V4_STAGE_ABSTAIN: str = "abstain"


W58_CONSENSUS_V4_STAGES: tuple[str, ...] = (
    W58_CONSENSUS_V4_STAGE_K_OF_N,
    W58_CONSENSUS_V4_STAGE_TRUST_WEIGHTED,
    W58_CONSENSUS_V4_STAGE_SUBSTRATE,
    W58_CONSENSUS_V4_STAGE_LOGIT_LENS,
    W58_CONSENSUS_V4_STAGE_CACHE_REUSE,
    W58_CONSENSUS_V4_STAGE_BEST_PARENT,
    W58_CONSENSUS_V4_STAGE_TRANSCRIPT,
    W58_CONSENSUS_V4_STAGE_ABSTAIN,
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
class ConsensusFallbackControllerV4:
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
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_CONSENSUS_V4_SCHEMA_VERSION,
            "kind": "consensus_v4_controller",
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
            query_direction: Sequence[float],
            transcript_payload: Sequence[float],
    ) -> dict[str, Any]:
        n = len(parent_payloads)
        if n == 0:
            self.audit.append(
                {"stage": W58_CONSENSUS_V4_STAGE_ABSTAIN,
                 "reason": "no_parents"})
            return {
                "decision_stage": W58_CONSENSUS_V4_STAGE_ABSTAIN,
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
                "stage": W58_CONSENSUS_V4_STAGE_K_OF_N,
                "agree": list(agree),
                "cos": [float(round(c, 9)) for c in cos],
            })
            idx = max(agree, key=lambda i: cos[i])
            return {
                "decision_stage": W58_CONSENSUS_V4_STAGE_K_OF_N,
                "selected_index": int(idx),
                "payload": list(parent_payloads[idx]),
            }
        # Stage 2 — trust-weighted quorum.
        tot_trust = sum(float(t) for t, c
                         in zip(parent_trusts, cos)
                         if c >= float(self.cosine_floor))
        if tot_trust >= float(self.trust_threshold):
            self.audit.append({
                "stage": W58_CONSENSUS_V4_STAGE_TRUST_WEIGHTED,
                "tot_trust": float(round(tot_trust, 9)),
            })
            cand = [i for i, c in enumerate(cos)
                     if c >= float(self.cosine_floor)]
            idx = max(cand,
                       key=lambda i: cos[i] * parent_trusts[i])
            return {
                "decision_stage":
                    W58_CONSENSUS_V4_STAGE_TRUST_WEIGHTED,
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
                        "stage": W58_CONSENSUS_V4_STAGE_SUBSTRATE,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W58_CONSENSUS_V4_STAGE_SUBSTRATE,
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
                            W58_CONSENSUS_V4_STAGE_LOGIT_LENS,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W58_CONSENSUS_V4_STAGE_LOGIT_LENS,
                        "selected_index": int(idx),
                        "payload": list(parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stage 5 — cache-reuse-replay tiebreaker (NEW).
        if (self.cache_reuse_oracle is not None
                and parent_cache_fingerprints is not None):
            try:
                idx = int(self.cache_reuse_oracle(
                    parent_payloads, query_direction,
                    parent_cache_fingerprints))
                if 0 <= idx < n:
                    self.audit.append({
                        "stage":
                            W58_CONSENSUS_V4_STAGE_CACHE_REUSE,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W58_CONSENSUS_V4_STAGE_CACHE_REUSE,
                        "selected_index": int(idx),
                        "payload": list(parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stage 6 — best-parent.
        best = max(range(n),
                    key=lambda i: parent_trusts[i] * cos[i])
        if cos[best] >= float(self.cosine_floor):
            self.audit.append({
                "stage": W58_CONSENSUS_V4_STAGE_BEST_PARENT,
                "selected_index": int(best),
            })
            return {
                "decision_stage":
                    W58_CONSENSUS_V4_STAGE_BEST_PARENT,
                "selected_index": int(best),
                "payload": list(parent_payloads[best]),
            }
        # Stage 7 — transcript fallback.
        if list(transcript_payload):
            self.audit.append({
                "stage": W58_CONSENSUS_V4_STAGE_TRANSCRIPT,
            })
            return {
                "decision_stage":
                    W58_CONSENSUS_V4_STAGE_TRANSCRIPT,
                "selected_index": -1,
                "payload": list(transcript_payload),
            }
        # Stage 8 — abstain.
        self.audit.append({
            "stage": W58_CONSENSUS_V4_STAGE_ABSTAIN,
            "reason": "all_paths_below_floor",
        })
        return {
            "decision_stage": W58_CONSENSUS_V4_STAGE_ABSTAIN,
            "selected_index": -1,
            "payload": [0.0] * len(query_direction),
        }


@dataclasses.dataclass(frozen=True)
class ConsensusFallbackControllerV4Witness:
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
            "kind": "consensus_v4_witness",
            "witness": self.to_dict()})


def emit_consensus_v4_witness(
        controller: ConsensusFallbackControllerV4,
) -> ConsensusFallbackControllerV4Witness:
    last_stage = (
        str(controller.audit[-1]["stage"])
        if controller.audit
        else W58_CONSENSUS_V4_STAGE_ABSTAIN)
    return ConsensusFallbackControllerV4Witness(
        schema=W58_CONSENSUS_V4_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_audit_entries=int(len(controller.audit)),
        last_stage=str(last_stage),
        n_stages_in_chain=int(len(W58_CONSENSUS_V4_STAGES)),
    )


__all__ = [
    "W58_CONSENSUS_V4_SCHEMA_VERSION",
    "W58_CONSENSUS_V4_STAGE_K_OF_N",
    "W58_CONSENSUS_V4_STAGE_TRUST_WEIGHTED",
    "W58_CONSENSUS_V4_STAGE_SUBSTRATE",
    "W58_CONSENSUS_V4_STAGE_LOGIT_LENS",
    "W58_CONSENSUS_V4_STAGE_CACHE_REUSE",
    "W58_CONSENSUS_V4_STAGE_BEST_PARENT",
    "W58_CONSENSUS_V4_STAGE_TRANSCRIPT",
    "W58_CONSENSUS_V4_STAGE_ABSTAIN",
    "W58_CONSENSUS_V4_STAGES",
    "ConsensusFallbackControllerV4",
    "ConsensusFallbackControllerV4Witness",
    "emit_consensus_v4_witness",
]
