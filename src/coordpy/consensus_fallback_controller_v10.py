"""W64 M13 — Consensus Fallback Controller V10.

Strictly extends W63's ``coordpy.consensus_fallback_controller_v9``.
V9 had a 13-stage chain with ``hidden_wins_arbiter`` between
``trained_repair`` and ``best_parent``. V10 adds a **14th stage**:

  replay_dominance_primary_arbiter

placed between ``hidden_wins_arbiter`` and ``best_parent``.
``replay_dominance_primary_arbiter`` fires when:

* the W64 replay controller V5's four-way bridge classifier
  predicts ``replay_wins`` for ≥ 1 parent,
* AND the parent's replay-dominance-primary score is positive,

in which case the V10 stage returns the replay-dominance-primary
parent's payload as the consensus output.

Honest scope (W64)
------------------

* The replay-dominance-primary arbiter relies on the four-way
  classifier's prediction, which is itself fit on synthetic
  supervision.
* The arbiter does NOT prove that replay bridges beat the others
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
from .consensus_fallback_controller_v9 import (
    ConsensusFallbackControllerV9,
    W63_CONSENSUS_V9_SCHEMA_VERSION,
    W63_CONSENSUS_V9_STAGES,
    W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER,
)
from .tiny_substrate_v3 import _sha256_hex


W64_CONSENSUS_V10_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v10.v1")
W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER: str = (
    "replay_dominance_primary_arbiter")


def _build_v10_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W63_CONSENSUS_V9_STAGES:
        out.append(s)
        if (not inserted and s
                == W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER):
            out.append(
                W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out) - 1)
        out.insert(
            idx,
            W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER)
    return tuple(out)


W64_CONSENSUS_V10_STAGES: tuple[str, ...] = _build_v10_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV10:
    inner_v9: ConsensusFallbackControllerV9
    replay_dominance_primary_threshold: float = 0.05
    audit_v10: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, k_required: int = 2,
            cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            repair_amount_threshold: float = 0.1,
            hidden_wins_margin_threshold: float = 0.05,
            replay_dominance_primary_threshold: float = 0.05,
    ) -> "ConsensusFallbackControllerV10":
        inner = ConsensusFallbackControllerV9.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            repair_amount_threshold=float(
                repair_amount_threshold),
            hidden_wins_margin_threshold=float(
                hidden_wins_margin_threshold))
        return cls(
            inner_v9=inner,
            replay_dominance_primary_threshold=float(
                replay_dominance_primary_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W64_CONSENSUS_V10_SCHEMA_VERSION,
            "kind": "consensus_v10_controller",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "stages": list(W64_CONSENSUS_V10_STAGES),
            "replay_dominance_primary_threshold": float(round(
                self.replay_dominance_primary_threshold, 12)),
        })

    def decide_v10(
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
            replay_dominance_primary_scores_per_parent: (
                Sequence[float] | None) = None,
            four_way_predictions_per_parent: (
                Sequence[str] | None) = None,
    ) -> dict[str, Any]:
        v9_out = self.inner_v9.decide_v9(
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
            repaired_payload=repaired_payload,
            hidden_wins_margins_per_parent=(
                hidden_wins_margins_per_parent),
            three_way_predictions_per_parent=(
                three_way_predictions_per_parent))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        v9_stage = str(v9_out.get("stage", ""))
        any_replay_wins = False
        winning_idx: int | None = None
        if (replay_dominance_primary_scores_per_parent is not None
                and four_way_predictions_per_parent is not None):
            for i, (m, p) in enumerate(zip(
                    replay_dominance_primary_scores_per_parent,
                    four_way_predictions_per_parent)):
                if (str(p) == "replay_wins" and float(m)
                        >= float(
                            self.replay_dominance_primary_threshold)):
                    any_replay_wins = True
                    winning_idx = int(i)
                    break
        if (v9_stage in terminal_stages
                and any_replay_wins
                and winning_idx is not None
                and winning_idx < len(payloads)):
            self.audit_v10.append({
                "stage":
                    W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER,
                "v9_terminal_stage": str(v9_stage),
                "winning_parent": int(winning_idx),
            })
            return {
                "stage":
                    W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER,
                "payload": [
                    float(x) for x in payloads[winning_idx]],
                "v10_promoted": True,
                "rationale": (
                    "replay_dominance_primary_arbiter_applied"),
            }
        self.audit_v10.append({
            "stage": v9_stage, "v10_promoted": False})
        return v9_out


@dataclasses.dataclass(frozen=True)
class ConsensusV10Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    replay_dominance_primary_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "replay_dominance_primary_stage_fired": int(
                self.replay_dominance_primary_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v10_witness",
            "witness": self.to_dict()})


def emit_consensus_v10_witness(
        controller: ConsensusFallbackControllerV10,
) -> ConsensusV10Witness:
    fired = sum(
        1 for e in controller.audit_v10
        if str(e.get("stage", ""))
            == W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER)
    return ConsensusV10Witness(
        schema=W64_CONSENSUS_V10_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W64_CONSENSUS_V10_STAGES),
        n_decisions=int(len(controller.audit_v10)),
        replay_dominance_primary_stage_fired=int(fired),
    )


__all__ = [
    "W64_CONSENSUS_V10_SCHEMA_VERSION",
    "W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER",
    "W64_CONSENSUS_V10_STAGES",
    "ConsensusFallbackControllerV10",
    "ConsensusV10Witness",
    "emit_consensus_v10_witness",
]
