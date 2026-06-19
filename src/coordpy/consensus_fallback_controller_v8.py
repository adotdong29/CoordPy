"""W62 M11 — Consensus Fallback Controller V8.

Strictly extends W61's ``coordpy.consensus_fallback_controller_v7``.
V7 had an 11-stage chain that inserted ``attention_pattern_consensus``
between ``replay_controller`` and ``best_parent``. V8 adds a
**12th stage**:

  trained_repair

placed between ``attention_pattern_consensus`` and ``best_parent``.
``trained_repair`` fires when:

* the V62 cache controller V5's trained repair head is available,
* AND ≥ 1 parent's CRC fingerprint indicated detected corruption,
* AND the repair head's predicted repair amount exceeds a threshold,

in which case the V8 stage returns the repaired payload as the
consensus output.

Honest scope
------------

* The repair amount is an *additive* correction; it does NOT
  un-corrupt the raw cached state.
  ``W62-L-CONSENSUS-V8-REPAIR-STAGE-SYNTHETIC-CAP`` documents.
* The repair threshold is a single scalar; not learned.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v7 import (
    ConsensusFallbackControllerV7,
    W61_CONSENSUS_V7_SCHEMA_VERSION,
    W61_CONSENSUS_V7_STAGES,
    W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN,
)
from .tiny_substrate_v3 import _sha256_hex


W62_CONSENSUS_V8_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v8.v1")
W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR: str = "trained_repair"


def _build_v8_stages() -> tuple[str, ...]:
    out = []
    inserted = False
    for s in W61_CONSENSUS_V7_STAGES:
        out.append(s)
        if (not inserted and s
                == W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN):
            out.append(W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR)
            inserted = True
    if not inserted:
        # Fall back: insert before best_parent.
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out) - 1)
        out.insert(
            idx, W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR)
    return tuple(out)


W62_CONSENSUS_V8_STAGES: tuple[str, ...] = _build_v8_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV8:
    inner_v7: ConsensusFallbackControllerV7
    repair_amount_threshold: float = 0.1
    audit_v8: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, k_required: int = 2,
            cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            repair_amount_threshold: float = 0.1,
    ) -> "ConsensusFallbackControllerV8":
        inner = ConsensusFallbackControllerV7.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold))
        return cls(
            inner_v7=inner,
            repair_amount_threshold=float(
                repair_amount_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W62_CONSENSUS_V8_SCHEMA_VERSION,
            "kind": "consensus_v8_controller",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "stages": list(W62_CONSENSUS_V8_STAGES),
            "repair_amount_threshold": float(round(
                self.repair_amount_threshold, 12)),
        })

    def decide_v8(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            attention_top_k_positions: (
                Sequence[Sequence[int]] | None) = None,
            attention_top_k_jaccard_floor: float = 0.5,
            transcript_available: bool = False,
            transcript_carrier: Sequence[float] | None = None,
            corruption_detected_per_parent: (
                Sequence[bool] | None) = None,
            repair_amount: float = 0.0,
            repaired_payload: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        # Stage 1..11 — try V7 first.
        v7_out = self.inner_v7.decide_v7(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            attention_top_k_positions=attention_top_k_positions,
            attention_top_k_jaccard_floor=float(
                attention_top_k_jaccard_floor),
            transcript_available=bool(transcript_available),
            transcript_carrier=transcript_carrier)
        # If V7 picked best_parent / transcript / abstain, try the
        # V8 repair stage first.
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        v7_stage = str(v7_out.get("stage", ""))
        any_corrupted = bool(
            corruption_detected_per_parent is not None
            and any(corruption_detected_per_parent))
        repair_above_threshold = bool(
            float(repair_amount)
            >= float(self.repair_amount_threshold))
        if (v7_stage in terminal_stages
                and any_corrupted
                and repair_above_threshold
                and repaired_payload is not None):
            self.audit_v8.append({
                "stage": W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR,
                "repair_amount": float(round(
                    repair_amount, 12)),
                "v7_terminal_stage": str(v7_stage),
            })
            return {
                "stage": W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR,
                "payload": [float(x)
                              for x in repaired_payload],
                "v8_promoted": True,
                "rationale": "trained_repair_applied",
            }
        self.audit_v8.append({
            "stage": v7_stage, "v8_promoted": False})
        return v7_out


@dataclasses.dataclass(frozen=True)
class ConsensusV8Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    repair_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "repair_stage_fired": int(self.repair_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v8_witness",
            "witness": self.to_dict()})


def emit_consensus_v8_witness(
        controller: ConsensusFallbackControllerV8,
) -> ConsensusV8Witness:
    fired = sum(
        1 for e in controller.audit_v8
        if str(e.get("stage", ""))
            == W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR)
    return ConsensusV8Witness(
        schema=W62_CONSENSUS_V8_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W62_CONSENSUS_V8_STAGES),
        n_decisions=int(len(controller.audit_v8)),
        repair_stage_fired=int(fired),
    )


__all__ = [
    "W62_CONSENSUS_V8_SCHEMA_VERSION",
    "W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR",
    "W62_CONSENSUS_V8_STAGES",
    "ConsensusFallbackControllerV8",
    "ConsensusV8Witness",
    "emit_consensus_v8_witness",
]
