"""W70 — Consensus Fallback Controller V16.

Strictly extends W69's
``coordpy.consensus_fallback_controller_v15``. V15 had a 24-stage
chain. V16 adds two new stages:

  repair_dominance_arbiter
  budget_primary_arbiter

placed between
``silent_corruption_plus_member_replacement_arbiter`` and
``best_parent``.

Honest scope (W70): ``W70-L-CONSENSUS-V16-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v15 import (
    ConsensusFallbackControllerV15,
    W69_CONSENSUS_V15_STAGES,
    W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER,
)
from .tiny_substrate_v3 import _sha256_hex


W70_CONSENSUS_V16_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v16.v1")
W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER: str = (
    "repair_dominance_arbiter")
W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER: str = (
    "budget_primary_arbiter")


def _build_v16_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W69_CONSENSUS_V15_STAGES:
        out.append(s)
        if (not inserted and s ==
                W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER):
            out.append(
                W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER)
            out.append(
                W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx, W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER)
        out.insert(
            idx, W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER)
    return tuple(out)


W70_CONSENSUS_V16_STAGES: tuple[str, ...] = _build_v16_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV16:
    inner_v15: ConsensusFallbackControllerV15
    repair_dominance_threshold: float = 0.5
    budget_primary_threshold: float = 0.5
    audit_v16: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            k_required: int = 2, cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            multi_branch_rejoin_threshold: float = 0.5,
            silent_corruption_threshold: float = 0.5,
            repair_dominance_threshold: float = 0.5,
            budget_primary_threshold: float = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV16":
        inner = ConsensusFallbackControllerV15.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            multi_branch_rejoin_threshold=float(
                multi_branch_rejoin_threshold),
            silent_corruption_threshold=float(
                silent_corruption_threshold),
            **inner_kwargs)
        return cls(
            inner_v15=inner,
            repair_dominance_threshold=float(
                repair_dominance_threshold),
            budget_primary_threshold=float(
                budget_primary_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W70_CONSENSUS_V16_SCHEMA_VERSION,
            "kind": "consensus_v16_controller",
            "inner_v15_cid": str(self.inner_v15.cid()),
            "stages": list(W70_CONSENSUS_V16_STAGES),
            "repair_dominance_threshold": float(round(
                self.repair_dominance_threshold, 12)),
            "budget_primary_threshold": float(round(
                self.budget_primary_threshold, 12)),
        })

    def decide_v16(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            repair_dominance_scores_per_parent: (
                Sequence[float] | None) = None,
            budget_primary_scores_per_parent: (
                Sequence[float] | None) = None,
            **v15_kwargs: Any,
    ) -> dict[str, Any]:
        v15_out = self.inner_v15.decide_v15(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v15_kwargs)
        v15_stage = str(v15_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 25: repair-dominance arbiter.
        if (v15_stage in terminal_stages
                and repair_dominance_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    repair_dominance_scores_per_parent):
                if (float(sc) >= float(
                        self.repair_dominance_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v16.append({
                    "stage": (
                        W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER),
                    "v15_terminal_stage": str(v15_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v16_promoted": True,
                    "rationale": "repair_dominance_applied",
                }
        # Stage 26: budget-primary arbiter.
        if (v15_stage in terminal_stages
                and budget_primary_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    budget_primary_scores_per_parent):
                if (float(sc) >= float(
                        self.budget_primary_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v16.append({
                    "stage": (
                        W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER),
                    "v15_terminal_stage": str(v15_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v16_promoted": True,
                    "rationale": "budget_primary_applied",
                }
        self.audit_v16.append({
            "stage": v15_stage, "v16_promoted": False})
        return v15_out


@dataclasses.dataclass(frozen=True)
class ConsensusV16Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    repair_dominance_stage_fired: int
    budget_primary_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "repair_dominance_stage_fired": int(
                self.repair_dominance_stage_fired),
            "budget_primary_stage_fired": int(
                self.budget_primary_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v16_witness",
            "witness": self.to_dict()})


def emit_consensus_v16_witness(
        controller: ConsensusFallbackControllerV16,
) -> ConsensusV16Witness:
    rd = sum(
        1 for e in controller.audit_v16
        if str(e.get("stage", ""))
            == W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER)
    bp = sum(
        1 for e in controller.audit_v16
        if str(e.get("stage", ""))
            == W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER)
    return ConsensusV16Witness(
        schema=W70_CONSENSUS_V16_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W70_CONSENSUS_V16_STAGES),
        n_decisions=int(len(controller.audit_v16)),
        repair_dominance_stage_fired=int(rd),
        budget_primary_stage_fired=int(bp),
    )


__all__ = [
    "W70_CONSENSUS_V16_SCHEMA_VERSION",
    "W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER",
    "W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER",
    "W70_CONSENSUS_V16_STAGES",
    "ConsensusFallbackControllerV16",
    "ConsensusV16Witness",
    "emit_consensus_v16_witness",
]
