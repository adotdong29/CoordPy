"""W63 M14 — Consensus Fallback Controller V9.

Strictly extends W62's ``coordpy.consensus_fallback_controller_v8``.
V8 had a 12-stage chain inserting ``trained_repair`` between
``attention_pattern_consensus`` and ``best_parent``. V9 adds a
**13th stage**:

  hidden_wins_arbiter

placed between ``trained_repair`` and ``best_parent``.
``hidden_wins_arbiter`` fires when:

* the W63 replay controller V4's three-way bridge classifier
  predicts ``hidden_wins`` for ≥ 1 parent,
* AND the parent's hidden-wins margin is positive,

in which case the V9 stage returns the hidden-wins parent's
payload as the consensus output.

Honest scope
------------

* The hidden-wins arbiter relies on the three-way classifier's
  prediction, which is itself fit on synthetic supervision.
* The arbiter does NOT prove that hidden bridges beat KV bridges
  on real models or workloads.
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
from .consensus_fallback_controller_v8 import (
    ConsensusFallbackControllerV8,
    W62_CONSENSUS_V8_SCHEMA_VERSION,
    W62_CONSENSUS_V8_STAGES,
    W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR,
)
from .tiny_substrate_v3 import _sha256_hex


W63_CONSENSUS_V9_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v9.v1")
W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER: str = (
    "hidden_wins_arbiter")


def _build_v9_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W62_CONSENSUS_V8_STAGES:
        out.append(s)
        if (not inserted and s
                == W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR):
            out.append(
                W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out) - 1)
        out.insert(
            idx, W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER)
    return tuple(out)


W63_CONSENSUS_V9_STAGES: tuple[str, ...] = _build_v9_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV9:
    inner_v8: ConsensusFallbackControllerV8
    hidden_wins_margin_threshold: float = 0.05
    audit_v9: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, k_required: int = 2,
            cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            repair_amount_threshold: float = 0.1,
            hidden_wins_margin_threshold: float = 0.05,
    ) -> "ConsensusFallbackControllerV9":
        inner = ConsensusFallbackControllerV8.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            repair_amount_threshold=float(
                repair_amount_threshold))
        return cls(
            inner_v8=inner,
            hidden_wins_margin_threshold=float(
                hidden_wins_margin_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W63_CONSENSUS_V9_SCHEMA_VERSION,
            "kind": "consensus_v9_controller",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "stages": list(W63_CONSENSUS_V9_STAGES),
            "hidden_wins_margin_threshold": float(round(
                self.hidden_wins_margin_threshold, 12)),
        })

    def decide_v9(
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
            hidden_wins_margins_per_parent: (
                Sequence[float] | None) = None,
            three_way_predictions_per_parent: (
                Sequence[str] | None) = None,
    ) -> dict[str, Any]:
        v8_out = self.inner_v8.decide_v8(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            attention_top_k_positions=attention_top_k_positions,
            attention_top_k_jaccard_floor=float(
                attention_top_k_jaccard_floor),
            transcript_available=bool(transcript_available),
            transcript_carrier=transcript_carrier,
            corruption_detected_per_parent=(
                corruption_detected_per_parent),
            repair_amount=float(repair_amount),
            repaired_payload=repaired_payload)
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        v8_stage = str(v8_out.get("stage", ""))
        any_hidden_wins = False
        winning_idx: int | None = None
        if (hidden_wins_margins_per_parent is not None
                and three_way_predictions_per_parent is not None):
            for i, (m, p) in enumerate(zip(
                    hidden_wins_margins_per_parent,
                    three_way_predictions_per_parent)):
                if (str(p) == "hidden_wins" and float(m)
                        >= float(
                            self.hidden_wins_margin_threshold)):
                    any_hidden_wins = True
                    winning_idx = int(i)
                    break
        if (v8_stage in terminal_stages
                and any_hidden_wins
                and winning_idx is not None
                and winning_idx < len(payloads)):
            self.audit_v9.append({
                "stage":
                    W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER,
                "v8_terminal_stage": str(v8_stage),
                "winning_parent": int(winning_idx),
            })
            return {
                "stage":
                    W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER,
                "payload": [
                    float(x) for x in payloads[winning_idx]],
                "v9_promoted": True,
                "rationale": "hidden_wins_arbiter_applied",
            }
        self.audit_v9.append({
            "stage": v8_stage, "v9_promoted": False})
        return v8_out


@dataclasses.dataclass(frozen=True)
class ConsensusV9Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    hidden_wins_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "hidden_wins_stage_fired": int(
                self.hidden_wins_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v9_witness",
            "witness": self.to_dict()})


def emit_consensus_v9_witness(
        controller: ConsensusFallbackControllerV9,
) -> ConsensusV9Witness:
    fired = sum(
        1 for e in controller.audit_v9
        if str(e.get("stage", ""))
            == W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER)
    return ConsensusV9Witness(
        schema=W63_CONSENSUS_V9_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W63_CONSENSUS_V9_STAGES),
        n_decisions=int(len(controller.audit_v9)),
        hidden_wins_stage_fired=int(fired),
    )


__all__ = [
    "W63_CONSENSUS_V9_SCHEMA_VERSION",
    "W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER",
    "W63_CONSENSUS_V9_STAGES",
    "ConsensusFallbackControllerV9",
    "ConsensusV9Witness",
    "emit_consensus_v9_witness",
]
