"""W75 — Consensus Fallback Controller V21.

Strictly extends W74's
``coordpy.consensus_fallback_controller_v20``. V20 had a 34-stage
chain. V21 adds two new stages:

  compound_chain_repair_arbiter
  compound_repair_after_replacement_then_rejoin_arbiter

placed between
``compound_repair_after_delayed_repair_then_replacement_arbiter``
and ``best_parent``.

Honest scope (W75): ``W75-L-CONSENSUS-V21-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v20 import (
    ConsensusFallbackControllerV20,
    W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER,
    W74_CONSENSUS_V20_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W75_CONSENSUS_V21_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v21.v1")
W75_CONSENSUS_V21_STAGE_COMPOUND_CHAIN_REPAIR_ARBITER: str = (
    "compound_chain_repair_arbiter")
W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER: str = (
    "compound_repair_after_replacement_then_rejoin_arbiter")


def _build_v21_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W74_CONSENSUS_V20_STAGES:
        out.append(s)
        if (not inserted and s ==
                W74_CONSENSUS_V20_STAGE_COMPOUND_REPAIR_AFTER_DRTR_ARBITER):
            out.append(
                W75_CONSENSUS_V21_STAGE_COMPOUND_CHAIN_REPAIR_ARBITER)
            out.append(
                W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx,
            W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER)
        out.insert(
            idx,
            W75_CONSENSUS_V21_STAGE_COMPOUND_CHAIN_REPAIR_ARBITER)
    return tuple(out)


W75_CONSENSUS_V21_STAGES: tuple[str, ...] = _build_v21_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV21:
    inner_v20: ConsensusFallbackControllerV20
    compound_chain_repair_threshold: float = 0.5
    compound_repair_rtr_threshold: float = 0.5
    audit_v21: list[dict[str, Any]] = dataclasses.field(
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
            compound_chain_repair_threshold: float = 0.5,
            compound_repair_rtr_threshold: float = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV21":
        inner = ConsensusFallbackControllerV20.init(
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
            compound_repair_threshold=float(
                compound_repair_threshold),
            compound_repair_drtr_threshold=float(
                compound_repair_drtr_threshold),
            **inner_kwargs)
        return cls(
            inner_v20=inner,
            compound_chain_repair_threshold=float(
                compound_chain_repair_threshold),
            compound_repair_rtr_threshold=float(
                compound_repair_rtr_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W75_CONSENSUS_V21_SCHEMA_VERSION,
            "kind": "consensus_v21_controller",
            "inner_v20_cid": str(self.inner_v20.cid()),
            "stages": list(W75_CONSENSUS_V21_STAGES),
            "compound_chain_repair_threshold": float(round(
                self.compound_chain_repair_threshold, 12)),
            "compound_repair_rtr_threshold": float(round(
                self.compound_repair_rtr_threshold, 12)),
        })

    def decide_v21(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            compound_chain_repair_scores_per_parent: (
                Sequence[float] | None) = None,
            compound_repair_rtr_scores_per_parent: (
                Sequence[float] | None) = None,
            **v20_kwargs: Any,
    ) -> dict[str, Any]:
        v20_out = self.inner_v20.decide_v20(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v20_kwargs)
        v20_stage = str(v20_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 35: compound-chain-repair arbiter.
        if (v20_stage in terminal_stages
                and compound_chain_repair_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    compound_chain_repair_scores_per_parent):
                if (float(sc) >= float(
                        self.compound_chain_repair_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v21.append({
                    "stage": (
                        W75_CONSENSUS_V21_STAGE_COMPOUND_CHAIN_REPAIR_ARBITER),
                    "v20_terminal_stage": str(v20_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W75_CONSENSUS_V21_STAGE_COMPOUND_CHAIN_REPAIR_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v21_promoted": True,
                    "rationale": "compound_chain_repair_applied",
                }
        # Stage 36: compound-repair-after-replacement-then-rejoin
        # arbiter.
        if (v20_stage in terminal_stages
                and compound_repair_rtr_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    compound_repair_rtr_scores_per_parent):
                if (float(sc) >= float(
                        self.compound_repair_rtr_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v21.append({
                    "stage": (
                        W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER),
                    "v20_terminal_stage": str(v20_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v21_promoted": True,
                    "rationale": "compound_repair_rtr_applied",
                }
        self.audit_v21.append({
            "stage": v20_stage, "v21_promoted": False})
        return v20_out


@dataclasses.dataclass(frozen=True)
class ConsensusV21Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    compound_chain_repair_stage_fired: int
    compound_repair_rtr_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "compound_chain_repair_stage_fired": int(
                self.compound_chain_repair_stage_fired),
            "compound_repair_rtr_stage_fired": int(
                self.compound_repair_rtr_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v21_witness",
            "witness": self.to_dict()})


def emit_consensus_v21_witness(
        controller: ConsensusFallbackControllerV21,
) -> ConsensusV21Witness:
    cr = sum(
        1 for e in controller.audit_v21
        if str(e.get("stage", ""))
            == W75_CONSENSUS_V21_STAGE_COMPOUND_CHAIN_REPAIR_ARBITER)
    cd = sum(
        1 for e in controller.audit_v21
        if str(e.get("stage", ""))
            == W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER)
    return ConsensusV21Witness(
        schema=W75_CONSENSUS_V21_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W75_CONSENSUS_V21_STAGES),
        n_decisions=int(len(controller.audit_v21)),
        compound_chain_repair_stage_fired=int(cr),
        compound_repair_rtr_stage_fired=int(cd),
    )


__all__ = [
    "W75_CONSENSUS_V21_SCHEMA_VERSION",
    "W75_CONSENSUS_V21_STAGE_COMPOUND_CHAIN_REPAIR_ARBITER",
    "W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER",
    "W75_CONSENSUS_V21_STAGES",
    "ConsensusFallbackControllerV21",
    "ConsensusV21Witness",
    "emit_consensus_v21_witness",
]
