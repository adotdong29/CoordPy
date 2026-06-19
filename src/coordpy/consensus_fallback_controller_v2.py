"""W56 M7 — Consensus Fallback Controller V2.

Extends W55 TWCC with a **6-stage decision chain**:

  1. K-of-N quorum
  2. trust-weighted quorum
  3. *substrate-conditioned* tiebreaker (NEW)
  4. best-parent
  5. transcript fallback
  6. abstain

The substrate-conditioned stage uses the tiny substrate's forward
as a tiebreaker oracle when capsule-layer consensus is split. The
substrate is queried with a packed representation of each
candidate parent capsule's payload; the candidate whose injection
maximises the substrate logit cosine to the consensus query
direction is preferred.

V2 strictly extends V1 (W55 TWCC): when ``substrate_oracle`` is
``None``, the controller reduces to W55's 5-stage chain
byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence


W56_CONSENSUS_V2_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v2.v1")

W56_CONSENSUS_V2_STAGE_K_OF_N: str = "k_of_n"
W56_CONSENSUS_V2_STAGE_TRUST_WEIGHTED: str = "trust_weighted"
W56_CONSENSUS_V2_STAGE_SUBSTRATE: str = "substrate_conditioned"
W56_CONSENSUS_V2_STAGE_BEST_PARENT: str = "best_parent"
W56_CONSENSUS_V2_STAGE_TRANSCRIPT: str = "transcript"
W56_CONSENSUS_V2_STAGE_ABSTAIN: str = "abstain"


W56_CONSENSUS_V2_STAGES: tuple[str, ...] = (
    W56_CONSENSUS_V2_STAGE_K_OF_N,
    W56_CONSENSUS_V2_STAGE_TRUST_WEIGHTED,
    W56_CONSENSUS_V2_STAGE_SUBSTRATE,
    W56_CONSENSUS_V2_STAGE_BEST_PARENT,
    W56_CONSENSUS_V2_STAGE_TRANSCRIPT,
    W56_CONSENSUS_V2_STAGE_ABSTAIN,
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
class ConsensusFallbackControllerV2:
    """6-stage decision chain.

    Each ``decide`` call records *every stage attempted* in the
    audit trail (not just the chosen stage), matching the W55
    TWCC content-addressing pattern.
    """

    k_required: int = 2
    cosine_floor: float = 0.6
    trust_threshold: float = 0.5
    substrate_oracle: (
        Callable[[Sequence[Sequence[float]],
                  Sequence[float]],
                 int] | None) = None
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W56_CONSENSUS_V2_SCHEMA_VERSION,
            "k_required": int(self.k_required),
            "cosine_floor": float(round(self.cosine_floor, 12)),
            "trust_threshold": float(round(
                self.trust_threshold, 12)),
            "audit": list(self.audit),
        })

    def decide(
            self,
            *,
            parent_payloads: Sequence[Sequence[float]],
            parent_trusts: Sequence[float],
            query_direction: Sequence[float],
            transcript_payload: Sequence[float],
    ) -> dict[str, Any]:
        if not parent_payloads:
            self.audit.append({
                "schema": W56_CONSENSUS_V2_SCHEMA_VERSION,
                "stage": W56_CONSENSUS_V2_STAGE_ABSTAIN,
                "reason": "no_parents",
            })
            return {
                "stage": W56_CONSENSUS_V2_STAGE_ABSTAIN,
                "decision": [0.0] * len(query_direction),
                "abstain": True,
            }

        # --- Stage 1: K-of-N quorum on payload cosine. ---
        agreeing: list[int] = []
        for i, p in enumerate(parent_payloads):
            if _cosine(p, query_direction) >= float(
                    self.cosine_floor):
                agreeing.append(int(i))
        stages_attempted: list[str] = []
        stages_attempted.append(W56_CONSENSUS_V2_STAGE_K_OF_N)
        if len(agreeing) >= int(self.k_required):
            merged = self._merge(
                [parent_payloads[i] for i in agreeing])
            entry = {
                "schema": W56_CONSENSUS_V2_SCHEMA_VERSION,
                "stage": W56_CONSENSUS_V2_STAGE_K_OF_N,
                "stages_attempted": list(stages_attempted),
                "n_agreeing": len(agreeing),
                "result_cid": _sha256_hex(
                    {"merged": list(merged)}),
            }
            self.audit.append(entry)
            return {
                "stage": W56_CONSENSUS_V2_STAGE_K_OF_N,
                "decision": list(merged),
                "abstain": False,
            }

        # --- Stage 2: trust-weighted quorum. ---
        stages_attempted.append(
            W56_CONSENSUS_V2_STAGE_TRUST_WEIGHTED)
        trust_total = sum(
            float(t) for t in parent_trusts if t is not None)
        if (trust_total >= float(self.trust_threshold)
                and len(parent_payloads) >= 2):
            merged = self._trust_weighted_merge(
                parent_payloads, parent_trusts)
            entry = {
                "schema": W56_CONSENSUS_V2_SCHEMA_VERSION,
                "stage": W56_CONSENSUS_V2_STAGE_TRUST_WEIGHTED,
                "stages_attempted": list(stages_attempted),
                "trust_total": float(round(trust_total, 12)),
                "result_cid": _sha256_hex(
                    {"merged": list(merged)}),
            }
            self.audit.append(entry)
            return {
                "stage": W56_CONSENSUS_V2_STAGE_TRUST_WEIGHTED,
                "decision": list(merged),
                "abstain": False,
            }

        # --- Stage 3 (NEW): substrate-conditioned tiebreaker. ---
        stages_attempted.append(W56_CONSENSUS_V2_STAGE_SUBSTRATE)
        if (self.substrate_oracle is not None
                and len(parent_payloads) >= 2):
            try:
                idx = int(self.substrate_oracle(
                    list(parent_payloads),
                    list(query_direction)))
            except Exception:
                idx = -1
            if 0 <= idx < len(parent_payloads):
                entry = {
                    "schema": W56_CONSENSUS_V2_SCHEMA_VERSION,
                    "stage": W56_CONSENSUS_V2_STAGE_SUBSTRATE,
                    "stages_attempted": list(stages_attempted),
                    "selected_parent_index": int(idx),
                    "result_cid": _sha256_hex(
                        {"selected": list(
                            parent_payloads[idx])}),
                }
                self.audit.append(entry)
                return {
                    "stage": (
                        W56_CONSENSUS_V2_STAGE_SUBSTRATE),
                    "decision": list(parent_payloads[idx]),
                    "abstain": False,
                }

        # --- Stage 4: best parent (highest trust × cosine). ---
        stages_attempted.append(
            W56_CONSENSUS_V2_STAGE_BEST_PARENT)
        scores = [
            float(t) * float(_cosine(p, query_direction))
            for p, t in zip(parent_payloads, parent_trusts)
        ]
        best = int(max(range(len(scores)),
                        key=lambda i: scores[i]))
        if scores[best] >= 0.1:
            entry = {
                "schema": W56_CONSENSUS_V2_SCHEMA_VERSION,
                "stage": W56_CONSENSUS_V2_STAGE_BEST_PARENT,
                "stages_attempted": list(stages_attempted),
                "selected_parent_index": int(best),
                "score": float(round(scores[best], 12)),
                "result_cid": _sha256_hex(
                    {"selected": list(parent_payloads[best])}),
            }
            self.audit.append(entry)
            return {
                "stage": W56_CONSENSUS_V2_STAGE_BEST_PARENT,
                "decision": list(parent_payloads[best]),
                "abstain": False,
            }

        # --- Stage 5: transcript fallback. ---
        stages_attempted.append(W56_CONSENSUS_V2_STAGE_TRANSCRIPT)
        if any(abs(float(x)) > 1e-9 for x in transcript_payload):
            entry = {
                "schema": W56_CONSENSUS_V2_SCHEMA_VERSION,
                "stage": W56_CONSENSUS_V2_STAGE_TRANSCRIPT,
                "stages_attempted": list(stages_attempted),
                "result_cid": _sha256_hex(
                    {"transcript": list(transcript_payload)}),
            }
            self.audit.append(entry)
            return {
                "stage": W56_CONSENSUS_V2_STAGE_TRANSCRIPT,
                "decision": list(transcript_payload),
                "abstain": False,
            }

        # --- Stage 6: abstain. ---
        stages_attempted.append(W56_CONSENSUS_V2_STAGE_ABSTAIN)
        entry = {
            "schema": W56_CONSENSUS_V2_SCHEMA_VERSION,
            "stage": W56_CONSENSUS_V2_STAGE_ABSTAIN,
            "stages_attempted": list(stages_attempted),
        }
        self.audit.append(entry)
        return {
            "stage": W56_CONSENSUS_V2_STAGE_ABSTAIN,
            "decision": [0.0] * len(query_direction),
            "abstain": True,
        }

    @staticmethod
    def _merge(payloads: Sequence[Sequence[float]]) -> list[float]:
        if not payloads:
            return []
        d = max(len(p) for p in payloads)
        out = [0.0] * d
        for p in payloads:
            for i in range(d):
                out[i] += float(p[i] if i < len(p) else 0.0)
        return [float(x / float(len(payloads))) for x in out]

    @staticmethod
    def _trust_weighted_merge(
            payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
    ) -> list[float]:
        d = max(len(p) for p in payloads)
        z = sum(float(t) for t in trusts) or 1.0
        out = [0.0] * d
        for p, t in zip(payloads, trusts):
            w = float(t) / float(z)
            for i in range(d):
                out[i] += w * float(p[i] if i < len(p) else 0.0)
        return out


@dataclasses.dataclass(frozen=True)
class ConsensusFallbackControllerV2Witness:
    schema: str
    controller_cid: str
    stages_attempted_per_decide: tuple[tuple[str, ...], ...]
    final_stage_per_decide: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages_attempted_per_decide": [
                list(s)
                for s in self.stages_attempted_per_decide],
            "final_stage_per_decide": list(
                self.final_stage_per_decide),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v2_witness",
            "witness": self.to_dict()})


def emit_consensus_v2_witness(
        controller: ConsensusFallbackControllerV2,
) -> ConsensusFallbackControllerV2Witness:
    stages_per: list[tuple[str, ...]] = []
    final_per: list[str] = []
    for entry in controller.audit:
        stages_per.append(tuple(
            entry.get("stages_attempted", []) or [
                entry.get("stage", "")]))
        final_per.append(str(entry.get("stage", "")))
    return ConsensusFallbackControllerV2Witness(
        schema=W56_CONSENSUS_V2_SCHEMA_VERSION,
        controller_cid=controller.cid(),
        stages_attempted_per_decide=tuple(stages_per),
        final_stage_per_decide=tuple(final_per),
    )


__all__ = [
    "W56_CONSENSUS_V2_SCHEMA_VERSION",
    "W56_CONSENSUS_V2_STAGE_K_OF_N",
    "W56_CONSENSUS_V2_STAGE_TRUST_WEIGHTED",
    "W56_CONSENSUS_V2_STAGE_SUBSTRATE",
    "W56_CONSENSUS_V2_STAGE_BEST_PARENT",
    "W56_CONSENSUS_V2_STAGE_TRANSCRIPT",
    "W56_CONSENSUS_V2_STAGE_ABSTAIN",
    "W56_CONSENSUS_V2_STAGES",
    "ConsensusFallbackControllerV2",
    "ConsensusFallbackControllerV2Witness",
    "emit_consensus_v2_witness",
]
