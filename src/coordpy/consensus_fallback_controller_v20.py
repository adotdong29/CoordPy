"""W74 — Consensus Fallback Controller V20.

Strictly extends W73's
``coordpy.consensus_fallback_controller_v19``. V19 had a 32-stage
chain. V20 adds two new stages:

  compound_repair_arbiter
  compound_repair_after_delayed_repair_then_replacement_arbiter

placed between
``replacement_after_contradiction_then_rejoin_arbiter`` and
``best_parent``.

Honest scope (W74): ``W74-L-CONSENSUS-V20-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v19 import (
    ConsensusFallbackControllerV19,
    W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER,
    W73_CONSENSUS_V19_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W74_CONSENSUS_V20_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v20.v1")
W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_ARBITER: str = (
    "compound_repair_arbiter")
W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER: str = (
    "compound_repair_after_delayed_repair_then_replacement_arbiter")


def _build_v20_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W73_CONSENSUS_V19_STAGES:
        out.append(s)
        if (not inserted and s ==
                W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER):
            out.append(
                W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_ARBITER)
            out.append(
                W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx,
            W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER)
        out.insert(
            idx,
            W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_ARBITER)
    return tuple(out)


W74_CONSENSUS_V20_STAGES: tuple[str, ...] = _build_v20_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV20:
    inner_v19: ConsensusFallbackControllerV19
    compound_repair_threshold: float = 0.5
    compound_repair_drtr_threshold: float = 0.5
    audit_v20: list[dict[str, Any]] = dataclasses.field(
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
            restart_aware_threshold: float = 0.5,
            delayed_repair_threshold: float = 0.5,
            rejoin_pressure_threshold: float = 0.5,
            delayed_rejoin_threshold: float = 0.5,
            replacement_pressure_threshold: float = 0.5,
            replacement_after_ctr_threshold: float = 0.5,
            compound_repair_threshold: float = 0.5,
            compound_repair_drtr_threshold: float = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV20":
        inner = ConsensusFallbackControllerV19.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            multi_branch_rejoin_threshold=float(
                multi_branch_rejoin_threshold),
            silent_corruption_threshold=float(
                silent_corruption_threshold),
            repair_dominance_threshold=float(
                repair_dominance_threshold),
            budget_primary_threshold=float(
                budget_primary_threshold),
            restart_aware_threshold=float(
                restart_aware_threshold),
            delayed_repair_threshold=float(
                delayed_repair_threshold),
            rejoin_pressure_threshold=float(
                rejoin_pressure_threshold),
            delayed_rejoin_threshold=float(
                delayed_rejoin_threshold),
            replacement_pressure_threshold=float(
                replacement_pressure_threshold),
            replacement_after_ctr_threshold=float(
                replacement_after_ctr_threshold),
            **inner_kwargs)
        return cls(
            inner_v19=inner,
            compound_repair_threshold=float(
                compound_repair_threshold),
            compound_repair_drtr_threshold=float(
                compound_repair_drtr_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W74_CONSENSUS_V20_SCHEMA_VERSION,
            "kind": "consensus_v20_controller",
            "inner_v19_cid": str(self.inner_v19.cid()),
            "stages": list(W74_CONSENSUS_V20_STAGES),
            "compound_repair_threshold": float(round(
                self.compound_repair_threshold, 12)),
            "compound_repair_drtr_threshold": float(round(
                self.compound_repair_drtr_threshold, 12)),
        })

    def decide_v20(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            compound_repair_scores_per_parent: (
                Sequence[float] | None) = None,
            compound_repair_drtr_scores_per_parent: (
                Sequence[float] | None) = None,
            **v19_kwargs: Any,
    ) -> dict[str, Any]:
        v19_out = self.inner_v19.decide_v19(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v19_kwargs)
        v19_stage = str(v19_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 33: compound-repair arbiter.
        if (v19_stage in terminal_stages
                and compound_repair_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    compound_repair_scores_per_parent):
                if (float(sc) >= float(
                        self.compound_repair_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v20.append({
                    "stage": (
                        W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_ARBITER),
                    "v19_terminal_stage": str(v19_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v20_promoted": True,
                    "rationale": "compound_repair_applied",
                }
        # Stage 34: compound-repair-after-DR-then-replacement arbiter.
        if (v19_stage in terminal_stages
                and compound_repair_drtr_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    compound_repair_drtr_scores_per_parent):
                if (float(sc) >= float(
                        self.compound_repair_drtr_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v20.append({
                    "stage": (
                        W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER),
                    "v19_terminal_stage": str(v19_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v20_promoted": True,
                    "rationale": "compound_repair_drtr_applied",
                }
        self.audit_v20.append({
            "stage": v19_stage, "v20_promoted": False})
        return v19_out


@dataclasses.dataclass(frozen=True)
class ConsensusV20Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    compound_repair_stage_fired: int
    compound_repair_drtr_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "compound_repair_stage_fired": int(
                self.compound_repair_stage_fired),
            "compound_repair_drtr_stage_fired": int(
                self.compound_repair_drtr_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v20_witness",
            "witness": self.to_dict()})


def emit_consensus_v20_witness(
        controller: ConsensusFallbackControllerV20,
) -> ConsensusV20Witness:
    cr = sum(
        1 for e in controller.audit_v20
        if str(e.get("stage", ""))
            == W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_ARBITER)
    cd = sum(
        1 for e in controller.audit_v20
        if str(e.get("stage", ""))
            == W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER)
    return ConsensusV20Witness(
        schema=W74_CONSENSUS_V20_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W74_CONSENSUS_V20_STAGES),
        n_decisions=int(len(controller.audit_v20)),
        compound_repair_stage_fired=int(cr),
        compound_repair_drtr_stage_fired=int(cd),
    )


__all__ = [
    "W74_CONSENSUS_V20_SCHEMA_VERSION",
    "W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_ARBITER",
    "W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER",
    "W74_CONSENSUS_V20_STAGES",
    "ConsensusFallbackControllerV20",
    "ConsensusV20Witness",
    "emit_consensus_v20_witness",
]
