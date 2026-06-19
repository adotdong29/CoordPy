"""W57 M9 — Consensus Fallback Controller V3.

Extends W56 ``ConsensusFallbackControllerV2`` with a **7-stage
decision chain**:

  1. K-of-N quorum                          (W55)
  2. trust-weighted quorum                  (W55)
  3. substrate-conditioned tiebreaker       (W56)
  4. **logits-conditioned tiebreaker** (NEW)
  5. best-parent                            (W55)
  6. transcript fallback                    (W55)
  7. abstain                                (W55)

The new logits-conditioned stage uses the substrate V2's logit
lens at an intermediate layer as a *secondary* tiebreaker oracle.
The standard substrate-conditioned stage compares final-layer
logit perturbations; the logits-conditioned stage compares
*per-layer* unembedding logits at a chosen intermediate layer.
This lets the controller break a tie when the final logits are
flat but an earlier layer has a clear preference.

V3 strictly extends V2: when ``logit_lens_oracle = None``, V3
reduces to V2 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence


W57_CONSENSUS_V3_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v3.v1")

W57_CONSENSUS_V3_STAGE_K_OF_N: str = "k_of_n"
W57_CONSENSUS_V3_STAGE_TRUST_WEIGHTED: str = "trust_weighted"
W57_CONSENSUS_V3_STAGE_SUBSTRATE: str = "substrate_conditioned"
W57_CONSENSUS_V3_STAGE_LOGIT_LENS: str = "logit_lens_conditioned"
W57_CONSENSUS_V3_STAGE_BEST_PARENT: str = "best_parent"
W57_CONSENSUS_V3_STAGE_TRANSCRIPT: str = "transcript"
W57_CONSENSUS_V3_STAGE_ABSTAIN: str = "abstain"


W57_CONSENSUS_V3_STAGES: tuple[str, ...] = (
    W57_CONSENSUS_V3_STAGE_K_OF_N,
    W57_CONSENSUS_V3_STAGE_TRUST_WEIGHTED,
    W57_CONSENSUS_V3_STAGE_SUBSTRATE,
    W57_CONSENSUS_V3_STAGE_LOGIT_LENS,
    W57_CONSENSUS_V3_STAGE_BEST_PARENT,
    W57_CONSENSUS_V3_STAGE_TRANSCRIPT,
    W57_CONSENSUS_V3_STAGE_ABSTAIN,
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
class ConsensusFallbackControllerV3:
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
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W57_CONSENSUS_V3_SCHEMA_VERSION,
            "kind": "consensus_v3_controller",
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
            query_direction: Sequence[float],
            transcript_payload: Sequence[float],
    ) -> dict[str, Any]:
        n = len(parent_payloads)
        if n == 0:
            self.audit.append(
                {"stage": W57_CONSENSUS_V3_STAGE_ABSTAIN,
                 "reason": "no_parents"})
            return {
                "decision_stage": W57_CONSENSUS_V3_STAGE_ABSTAIN,
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
                "stage": W57_CONSENSUS_V3_STAGE_K_OF_N,
                "agree": list(agree),
                "cos": [float(round(c, 9)) for c in cos],
            })
            idx = max(agree, key=lambda i: cos[i])
            return {
                "decision_stage": W57_CONSENSUS_V3_STAGE_K_OF_N,
                "selected_index": int(idx),
                "payload": list(parent_payloads[idx]),
            }
        # Stage 2 — trust-weighted quorum.
        tot_trust = sum(float(t) for t, c in zip(parent_trusts, cos)
                         if c >= float(self.cosine_floor))
        if tot_trust >= float(self.trust_threshold):
            self.audit.append({
                "stage": W57_CONSENSUS_V3_STAGE_TRUST_WEIGHTED,
                "tot_trust": float(round(tot_trust, 9)),
            })
            cand = [i for i, c in enumerate(cos)
                     if c >= float(self.cosine_floor)]
            idx = max(cand, key=lambda i: cos[i] * parent_trusts[i])
            return {
                "decision_stage":
                    W57_CONSENSUS_V3_STAGE_TRUST_WEIGHTED,
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
                        "stage": W57_CONSENSUS_V3_STAGE_SUBSTRATE,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W57_CONSENSUS_V3_STAGE_SUBSTRATE,
                        "selected_index": int(idx),
                        "payload": list(parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stage 4 — logit-lens-conditioned tiebreaker (NEW).
        if self.logit_lens_oracle is not None:
            try:
                idx = int(self.logit_lens_oracle(
                    parent_payloads, query_direction))
                if 0 <= idx < n:
                    self.audit.append({
                        "stage":
                            W57_CONSENSUS_V3_STAGE_LOGIT_LENS,
                        "selected_index": int(idx),
                    })
                    return {
                        "decision_stage":
                            W57_CONSENSUS_V3_STAGE_LOGIT_LENS,
                        "selected_index": int(idx),
                        "payload": list(parent_payloads[idx]),
                    }
            except Exception:
                pass
        # Stage 5 — best-parent.
        best = max(range(n),
                    key=lambda i: parent_trusts[i] * cos[i])
        if cos[best] >= float(self.cosine_floor):
            self.audit.append({
                "stage": W57_CONSENSUS_V3_STAGE_BEST_PARENT,
                "selected_index": int(best),
            })
            return {
                "decision_stage":
                    W57_CONSENSUS_V3_STAGE_BEST_PARENT,
                "selected_index": int(best),
                "payload": list(parent_payloads[best]),
            }
        # Stage 6 — transcript fallback.
        if list(transcript_payload):
            self.audit.append({
                "stage": W57_CONSENSUS_V3_STAGE_TRANSCRIPT,
            })
            return {
                "decision_stage":
                    W57_CONSENSUS_V3_STAGE_TRANSCRIPT,
                "selected_index": -1,
                "payload": list(transcript_payload),
            }
        # Stage 7 — abstain.
        self.audit.append({
            "stage": W57_CONSENSUS_V3_STAGE_ABSTAIN,
            "reason": "all_paths_below_floor",
        })
        return {
            "decision_stage": W57_CONSENSUS_V3_STAGE_ABSTAIN,
            "selected_index": -1,
            "payload": [0.0] * len(query_direction),
        }


@dataclasses.dataclass(frozen=True)
class ConsensusFallbackControllerV3Witness:
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
            "kind": "consensus_v3_witness",
            "witness": self.to_dict()})


def emit_consensus_v3_witness(
        controller: ConsensusFallbackControllerV3,
) -> ConsensusFallbackControllerV3Witness:
    last_stage = (
        str(controller.audit[-1]["stage"])
        if controller.audit
        else W57_CONSENSUS_V3_STAGE_ABSTAIN)
    return ConsensusFallbackControllerV3Witness(
        schema=W57_CONSENSUS_V3_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_audit_entries=int(len(controller.audit)),
        last_stage=str(last_stage),
        n_stages_in_chain=int(len(W57_CONSENSUS_V3_STAGES)),
    )


__all__ = [
    "W57_CONSENSUS_V3_SCHEMA_VERSION",
    "W57_CONSENSUS_V3_STAGE_K_OF_N",
    "W57_CONSENSUS_V3_STAGE_TRUST_WEIGHTED",
    "W57_CONSENSUS_V3_STAGE_SUBSTRATE",
    "W57_CONSENSUS_V3_STAGE_LOGIT_LENS",
    "W57_CONSENSUS_V3_STAGE_BEST_PARENT",
    "W57_CONSENSUS_V3_STAGE_TRANSCRIPT",
    "W57_CONSENSUS_V3_STAGE_ABSTAIN",
    "W57_CONSENSUS_V3_STAGES",
    "ConsensusFallbackControllerV3",
    "ConsensusFallbackControllerV3Witness",
    "emit_consensus_v3_witness",
]
